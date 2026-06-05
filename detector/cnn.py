"""
CNN Solar Classifier — transfer-learned MobileNetV3-Small
=========================================================
The learned alternative to the heuristic 5-stage detector. It detects rooftop
solar directly from pixels, so it handles dark/black panels that the colour-based
heuristic misses (e.g. 3961 25th St).

Activation is automatic and safe:
  • If a trained model exists at MODEL_PATH → the pipeline uses it (authoritative).
  • If not → `is_available()` is False and the heuristic detector runs unchanged.

Training data comes from your confirmed references (the "Confirm Has/No Solar"
flywheel). Build the dataset and train with:
    python manage.py prepare_cnn_data
    python -m detector.cnn_train --epochs 12

Torch is imported lazily inside functions so importing this module (e.g. during
normal web requests) never pays the torch import cost unless a model is in use.
"""
from pathlib import Path
import logging
from django.conf import settings

logger = logging.getLogger(__name__)

MODEL_PATH = Path(settings.MEDIA_ROOT) / "models" / "solar_cnn.pt"
IMG_SIZE   = 224
CLASSES    = ["no_solar", "solar"]   # ImageFolder-sorted order → indices 0, 1

_model  = None    # lazily-loaded, cached
_device = None


# ── Device / model ─────────────────────────────────────────────────────────

def _get_device():
    import torch
    global _device
    if _device is None:
        if torch.backends.mps.is_available():
            _device = torch.device("mps")
        elif torch.cuda.is_available():
            _device = torch.device("cuda")
        else:
            _device = torch.device("cpu")
    return _device


def build_model(num_classes: int = 2):
    """MobileNetV3-Small backbone (ImageNet) with a fresh binary head."""
    import torch.nn as nn
    from torchvision.models import mobilenet_v3_small, MobileNet_V3_Small_Weights
    model = mobilenet_v3_small(weights=MobileNet_V3_Small_Weights.DEFAULT)
    in_features = model.classifier[-1].in_features
    model.classifier[-1] = nn.Linear(in_features, num_classes)
    return model


def eval_transform():
    """Deterministic transform for inference (and validation)."""
    from torchvision import transforms
    return transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(IMG_SIZE),   # the address point sits at image centre
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])


# ── Inference ──────────────────────────────────────────────────────────────

def is_available() -> bool:
    """True once a trained model has been saved to disk."""
    return MODEL_PATH.exists()


def _load():
    global _model
    if _model is not None:
        return _model
    if not MODEL_PATH.exists():
        return None
    import torch
    model = build_model()
    ckpt  = torch.load(MODEL_PATH, map_location="cpu")
    state = ckpt.get("model", ckpt) if isinstance(ckpt, dict) else ckpt
    model.load_state_dict(state)
    model.eval().to(_get_device())
    _model = model
    logger.info(f"Loaded CNN solar classifier from {MODEL_PATH}")
    return _model


def predict(pil_image) -> dict | None:
    """
    Classify a satellite image. Returns {'prob': float, 'has_solar': bool}
    where prob is P(solar), or None if no trained model is available.
    """
    model = _load()
    if model is None:
        return None
    import torch
    x = eval_transform()(pil_image.convert("RGB")).unsqueeze(0).to(_get_device())
    with torch.no_grad():
        prob = torch.softmax(model(x), dim=1)[0, CLASSES.index("solar")].item()
    return {"prob": round(prob, 4), "has_solar": prob >= 0.5}
