"""Export a trained MobileNet checkpoint for lightweight production inference."""
import argparse
from pathlib import Path


def export(checkpoint, output):
    import onnx
    import torch
    from detector.cnn import build_model, IMG_SIZE

    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    model = build_model()
    state = payload.get("model", payload) if isinstance(payload, dict) else payload
    model.load_state_dict(state)
    model.eval()
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    example = torch.zeros(1, 3, IMG_SIZE, IMG_SIZE, dtype=torch.float32)
    torch.onnx.export(
        model, example, str(output), input_names=["image"], output_names=["logits"],
        dynamic_axes={"image": {0: "batch"}, "logits": {0: "batch"}},
        opset_version=17, dynamo=False,
    )
    onnx.checker.check_model(str(output))
    print(f"Exported {output} ({output.stat().st_size:,} bytes)")


if __name__ == "__main__":
    import os
    import django
    os.environ.setdefault("DJANGO_SETTINGS_MODULE", "solarleads.settings")
    django.setup()
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoint")
    parser.add_argument("output")
    args = parser.parse_args()
    export(args.checkpoint, args.output)
