"""
CNN Solar Classifier — transfer-learned MobileNetV3-Small
=========================================================
The learned alternative to the heuristic 5-stage detector. It detects rooftop
solar directly from pixels, so it handles dark/black panels that the colour-based
heuristic misses (e.g. 3961 25th St).

When the bundled ONNX checkpoint exists, the pipeline uses it. PyTorch checkpoints
remain available for local experiments; without either artifact, the heuristic
detector runs unchanged.

Training data comes from your confirmed references (the "Confirm Has/No Solar"
flywheel). Build the dataset and train with:
    python manage.py prepare_cnn_data
    python -m detector.cnn_train --epochs 12

Inference uses ONNX Runtime so production does not need PyTorch or torchvision.
Training dependencies are imported lazily and remain optional.
"""
from pathlib import Path
import logging
from django.conf import settings

logger = logging.getLogger(__name__)

MODEL_PATH = Path(settings.MEDIA_ROOT) / "models" / "solar_cnn.pt"
ONNX_MODEL_PATH = Path(__file__).resolve().parent / "models" / "solar_cnn_queens2018.onnx"
IMG_SIZE   = 224
CLASSES    = ["no_solar", "solar"]   # ImageFolder-sorted order → indices 0, 1
# Selected on the independent validation split to maximize balanced accuracy.
# The exported model's softmax score is not calibrated probability.
SOLAR_DECISION_THRESHOLD = 0.42


def is_solar_probability(prob: float) -> bool:
    """Apply the validation-tuned decision cutoff to a model score."""
    return prob >= SOLAR_DECISION_THRESHOLD

_model  = None    # lazily-loaded, cached
_onnx_session = None
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
    return ONNX_MODEL_PATH.exists() or MODEL_PATH.exists()


def _load():
    global _model, _onnx_session
    if _onnx_session is not None:
        return "onnx"
    if _model is not None:
        return _model
    if ONNX_MODEL_PATH.exists():
        import onnxruntime as ort
        options = ort.SessionOptions()
        options.intra_op_num_threads = 1
        options.inter_op_num_threads = 1
        _onnx_session = ort.InferenceSession(
            str(ONNX_MODEL_PATH), sess_options=options, providers=["CPUExecutionProvider"]
        )
        logger.info(f"Loaded ONNX solar classifier from {ONNX_MODEL_PATH}")
        return "onnx"
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


def _onnx_input(pil_image):
    """PIL/NumPy equivalent of Resize(256), CenterCrop(224), ImageNet normalize."""
    import numpy as np
    from PIL import Image

    image = pil_image.convert("RGB")
    width, height = image.size
    scale = 256 / min(width, height)
    resized = (round(width * scale), round(height * scale))
    image = image.resize(resized, resample=Image.Resampling.BILINEAR)
    left = round((resized[0] - IMG_SIZE) / 2)
    top = round((resized[1] - IMG_SIZE) / 2)
    image = image.crop((left, top, left + IMG_SIZE, top + IMG_SIZE))
    array = np.asarray(image, dtype=np.float32) / 255.0
    mean = np.asarray([0.485, 0.456, 0.406], dtype=np.float32)
    std = np.asarray([0.229, 0.224, 0.225], dtype=np.float32)
    return ((array - mean) / std).transpose(2, 0, 1)[None, ...].astype(np.float32)


def predict(pil_image) -> dict | None:
    """
    Classify a satellite image. Returns {'prob': float, 'has_solar': bool}
    where prob is P(solar), or None if no trained model is available.
    """
    loaded = _load()
    if loaded is None:
        return None
    if loaded == "onnx":
        import numpy as np
        logits = _onnx_session.run(None, {"image": _onnx_input(pil_image)})[0][0]
        probabilities = np.exp(logits - np.max(logits))
        probabilities /= probabilities.sum()
        prob = float(probabilities[CLASSES.index("solar")])
    else:
        import torch
        x = eval_transform()(pil_image.convert("RGB")).unsqueeze(0).to(_get_device())
        with torch.no_grad():
            prob = torch.softmax(loaded(x), dim=1)[0, CLASSES.index("solar")].item()
    return {"prob": round(prob, 4), "has_solar": is_solar_probability(prob)}
