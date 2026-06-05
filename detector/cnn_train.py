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
    import torch, torch.nn as nn
    from torch.utils.data import DataLoader, random_split
    from torchvision import datasets, transforms
    from detector.cnn import build_model, _get_device, IMG_SIZE, MODEL_PATH

    data_dir = Path(data_dir)
    out_path = Path(out_path) if out_path else MODEL_PATH
    device   = _get_device()

    base = datasets.ImageFolder(str(data_dir))   # transform=None → yields PIL images
    print(f"device={device}  data={data_dir}  n={len(base)}  classes={base.class_to_idx}")
    if len(base) < 4:
        raise SystemExit("Need at least 4 images. Run `python manage.py prepare_cnn_data` "
                         "and confirm more homes to grow the dataset.")

    n_val = max(1, int(len(base) * val_frac))
    g = torch.Generator().manual_seed(seed)
    tr_sub, va_sub = random_split(base, [len(base) - n_val, n_val], generator=g)

    norm = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    train_tf = transforms.Compose([
        transforms.RandomResizedCrop(IMG_SIZE, scale=(0.6, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.ColorJitter(0.2, 0.2, 0.2, 0.03),
        transforms.ToTensor(), norm,
    ])
    eval_tf = transforms.Compose([
        transforms.Resize(256), transforms.CenterCrop(IMG_SIZE),
        transforms.ToTensor(), norm,
    ])
    tr = DataLoader(_TransformSubset(tr_sub, train_tf), batch_size=batch, shuffle=True)
    va = DataLoader(_TransformSubset(va_sub, eval_tf),  batch_size=batch)

    model = build_model().to(device)
    # Freeze backbone; train head + last feature block only
    for p in model.parameters():               p.requires_grad = False
    for p in model.classifier.parameters():    p.requires_grad = True
    for p in model.features[-1].parameters():  p.requires_grad = True

    opt  = torch.optim.Adam([p for p in model.parameters() if p.requires_grad], lr=lr)
    crit = nn.CrossEntropyLoss()

    best_acc, best_state = -1.0, None
    for ep in range(1, epochs + 1):
        model.train()
        for x, y in tr:
            x, y = x.to(device), y.to(device)
            opt.zero_grad()
            crit(model(x), y).backward()
            opt.step()

        model.eval()
        correct = total = 0
        with torch.no_grad():
            for x, y in va:
                x, y = x.to(device), y.to(device)
                correct += (model(x).argmax(1) == y).sum().item()
                total   += y.numel()
        acc = correct / max(total, 1)
        print(f"epoch {ep:2d}/{epochs}  val_acc={acc:.3f}")
        if acc >= best_acc:
            best_acc  = acc
            best_state = {k: v.cpu() for k, v in model.state_dict().items()}

    out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"model": best_state, "classes": ["no_solar", "solar"],
                "val_acc": best_acc, "n": len(base)}, out_path)
    print(f"saved {out_path}  best_val_acc={best_acc:.3f}")
    return {"out": str(out_path), "val_acc": best_acc, "n": len(base)}


if __name__ == "__main__":
    import os, django
    os.environ.setdefault("DJANGO_SETTINGS_MODULE", "solarleads.settings")
    django.setup()
    ap = argparse.ArgumentParser()
    ap.add_argument("--data",   default="data/train")
    ap.add_argument("--out",    default=None, help="defaults to media/models/solar_cnn.pt")
    ap.add_argument("--epochs", type=int, default=12)
    ap.add_argument("--batch",  type=int, default=16)
    args = ap.parse_args()
    train(args.data, args.out, epochs=args.epochs, batch=args.batch)
