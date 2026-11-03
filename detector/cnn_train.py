"""
Fine-tune the MobileNetV3-Small solar classifier.
==================================================
Data layout (torchvision ImageFolder):
    data/train/solar/*.jpg
    data/train/no_solar/*.jpg

Build that folder from your confirmed references first:
    python manage.py prepare_cnn_data

Then train (inside the venv):
    python -m detector.cnn_train --epochs 12

Only the classifier head + last feature block are trained (the ImageNet backbone
is frozen), so it fine-tunes fast on a modest dataset and on Apple-Silicon MPS.
The best-val checkpoint is saved to detector.cnn.MODEL_PATH, which auto-activates
it in the detection pipeline.
"""
import argparse
from pathlib import Path


class _TransformSubset:
    """Apply a transform on top of a (transform-less) ImageFolder Subset."""
    def __init__(self, subset, transform):
        self.subset = subset
        self.transform = transform

    def __len__(self):
        return len(self.subset)

    def __getitem__(self, i):
        img, label = self.subset[i]          # base ImageFolder has no transform → PIL
        return self.transform(img), label


def train(data_dir="data/train", out_path=None, epochs=12,
          batch=16, lr=1e-3, val_frac=0.2, seed=42):
    import random
    import numpy as np
    import torch, torch.nn as nn
    from torch.utils.data import DataLoader, random_split
    from torchvision import datasets, transforms
    from detector.cnn import build_model, _get_device, IMG_SIZE, MODEL_PATH

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    data_dir = Path(data_dir)
    out_path = Path(out_path) if out_path else MODEL_PATH
    device   = _get_device()

    data_dir = Path(data_dir)
    split_layout = all((data_dir / split).is_dir() for split in ("train", "val", "test"))
    if split_layout:
        train_base = datasets.ImageFolder(str(data_dir / "train"))
        val_base = datasets.ImageFolder(str(data_dir / "val"))
        test_base = datasets.ImageFolder(str(data_dir / "test"))
        if not (train_base.class_to_idx == val_base.class_to_idx == test_base.class_to_idx):
            raise SystemExit("Train, validation, and test folders must have matching classes.")
        n = len(train_base)
        print(f"device={device}  data={data_dir}  split=train/val/test "
              f"n={n}/{len(val_base)}/{len(test_base)} classes={train_base.class_to_idx}")
    else:
        base = datasets.ImageFolder(str(data_dir))   # transform=None → yields PIL images
        n = len(base)
        print(f"device={device}  data={data_dir}  n={n}  classes={base.class_to_idx}")
    if n < 4:
        raise SystemExit("Need at least 4 images. Run `python manage.py prepare_cnn_data` "
                         "and confirm more homes to grow the dataset.")

    if not split_layout:
        n_val = max(1, int(len(base) * val_frac))
        g = torch.Generator().manual_seed(seed)
        tr_sub, va_sub = random_split(base, [len(base) - n_val, n_val], generator=g)

    norm = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    train_tf = transforms.Compose([
        transforms.RandomResizedCrop(IMG_SIZE, scale=(0.8, 1.0), ratio=(0.95, 1.05)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.ColorJitter(0.2, 0.2, 0.2, 0.03),
        transforms.ToTensor(), norm,
    ])
    eval_tf = transforms.Compose([
        transforms.Resize(256), transforms.CenterCrop(IMG_SIZE),
        transforms.ToTensor(), norm,
    ])
    if split_layout:
        tr = DataLoader(_TransformSubset(train_base, train_tf), batch_size=batch, shuffle=True)
        va = DataLoader(_TransformSubset(val_base, eval_tf), batch_size=batch)
        te = DataLoader(_TransformSubset(test_base, eval_tf), batch_size=batch)
    else:
        tr = DataLoader(_TransformSubset(tr_sub, train_tf), batch_size=batch, shuffle=True)
        va = DataLoader(_TransformSubset(va_sub, eval_tf), batch_size=batch)
        te = None

    model = build_model().to(device)
    # Freeze backbone; train head + last feature block only
    for p in model.parameters():               p.requires_grad = False
    for p in model.classifier.parameters():    p.requires_grad = True
    for p in model.features[-1].parameters():  p.requires_grad = True

    opt  = torch.optim.Adam([p for p in model.parameters() if p.requires_grad], lr=lr)
    crit = nn.CrossEntropyLoss()

    best_metric, best_state, best_val = -1.0, None, None

    def evaluate(loader):
        model.eval()
        matrix = torch.zeros((2, 2), dtype=torch.int64)
        with torch.no_grad():
            for x, y in loader:
                pred = model(x.to(device)).argmax(1).cpu()
                for actual, guessed in zip(y, pred):
                    matrix[int(actual), int(guessed)] += 1
        tn, fp = matrix[0, 0].item(), matrix[0, 1].item()
        fn, tp = matrix[1, 0].item(), matrix[1, 1].item()
        total = max(tn + fp + fn + tp, 1)
        recall = tp / max(tp + fn, 1)
        specificity = tn / max(tn + fp, 1)
        precision = tp / max(tp + fp, 1)
        return {
            "n": total,
            "accuracy": (tn + tp) / total,
            "balanced_accuracy": (recall + specificity) / 2,
            "precision": precision,
            "recall": recall,
            "specificity": specificity,
            "confusion_matrix": [[tn, fp], [fn, tp]],
        }

    for ep in range(1, epochs + 1):
        model.train()
        for x, y in tr:
            x, y = x.to(device), y.to(device)
            opt.zero_grad()
            crit(model(x), y).backward()
            opt.step()

        metrics = evaluate(va)
        print(f"epoch {ep:2d}/{epochs}  val_bal_acc={metrics['balanced_accuracy']:.3f} "
              f"recall={metrics['recall']:.3f} specificity={metrics['specificity']:.3f}")
        if metrics["balanced_accuracy"] >= best_metric:
            best_metric = metrics["balanced_accuracy"]
            best_val = metrics
            best_state = {k: v.cpu() for k, v in model.state_dict().items()}

    model.load_state_dict(best_state)
    test_metrics = evaluate(te) if te is not None else None
    out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"model": best_state, "classes": ["no_solar", "solar"],
                "val_metrics": best_val, "test_metrics": test_metrics, "n": n}, out_path)
    print(f"saved {out_path}  best_val_bal_acc={best_metric:.3f}")
    if test_metrics:
        print(f"test={test_metrics}")
    return {"out": str(out_path), "val_metrics": best_val,
            "test_metrics": test_metrics, "n": n}


if __name__ == "__main__":
    import os, django
    os.environ.setdefault("DJANGO_SETTINGS_MODULE", "solarleads.settings")
    django.setup()
    ap = argparse.ArgumentParser()
    ap.add_argument("--data",   default="data/train")
    ap.add_argument("--out",    default=None, help="defaults to media/models/solar_cnn.pt")
    ap.add_argument("--epochs", type=int, default=12)
    ap.add_argument("--batch",  type=int, default=16)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    train(args.data, args.out, epochs=args.epochs, batch=args.batch,
          lr=args.lr, seed=args.seed)
