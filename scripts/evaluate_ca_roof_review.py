"""Evaluate the rooftop in-frame gate on the independent NAIP review set.

The city-level test split is deliberately unavailable to this script. The
labels are visual review labels and must be quality-checked before they are
treated as ground truth.
"""
import argparse
import csv
import os
import sys
from collections import Counter
from pathlib import Path

import cv2
import numpy as np
from PIL import Image


ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "ca_roof_review_v2"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", choices=("train", "validation"), required=True)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    sys.path.insert(0, str(ROOT))
    os.environ.setdefault("DJANGO_SETTINGS_MODULE", "solarleads.settings")
    import django
    django.setup()
    from detector.rooftop_annotator import _annotate_rooftop

    with (DATA / "manifest.csv").open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    eligible = [row for row in rows
                if row["split"] == args.split
                and row["roof_label"] in {"roof", "nonroof"}]

    counts = Counter()
    detail = []
    for row in eligible:
        image = np.asarray(Image.open(DATA / row["image_file"]).convert("RGB"))
        _, result = _annotate_rooftop(cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
        predicted = "roof" if result["in_frame"] else "nonroof"
        counts[(row["roof_label"], predicted)] += 1
        detail.append({
            "sample_id": row["sample_id"], "city": row["city"],
            "actual": row["roof_label"], "predicted": predicted,
            "correct": str(predicted == row["roof_label"]).lower(),
            "method": result.get("roof_detection_method", ""),
            "reason": result.get("reason", ""),
        })

    tp, fn = counts["roof", "roof"], counts["roof", "nonroof"]
    fp, tn = counts["nonroof", "roof"], counts["nonroof", "nonroof"]
    n = len(eligible)
    print(f"Split: {args.split}; labeled chips: {n}; uncertain excluded: "
          f"{sum(row['split'] == args.split and row['roof_label'] == 'uncertain' for row in rows)}")
    print(f"Confusion (TP/FN/FP/TN): {tp}/{fn}/{fp}/{tn}")
    print(f"Accuracy: {(tp + tn) / n:.3f}; precision: {tp / (tp + fp) if tp + fp else 0:.3f}; "
          f"recall: {tp / (tp + fn) if tp + fn else 0:.3f}; "
          f"specificity: {tn / (tn + fp) if tn + fp else 0:.3f}")
    for row in detail:
        if row["correct"] == "false":
            print(f"ERROR {row['sample_id']}: actual={row['actual']} "
                  f"predicted={row['predicted']} method={row['method']} "
                  f"reason={row['reason']}")

    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=detail[0].keys())
            writer.writeheader()
            writer.writerows(detail)


if __name__ == "__main__":
    main()
