"""Evaluate residential-roof detection against parcel-marked SF targets.

Uses the clean NAIP chips in `raw/`, not marked review images. Test predictions
are intentionally not supported until the detector is frozen.
"""
import argparse
import csv
import json
import os
import sys
from collections import Counter
from pathlib import Path

import cv2
import numpy as np
from PIL import Image


ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "sf_parcel_roof_review"
IMAGE_SIZE = 640
# Match the nominal in-app Google Maps zoom-20 output scale near San Francisco.
PIXEL_SIZE_M = 0.12


def web_mercator(lat, lon):
    r = 6378137.0
    return r * np.deg2rad(lon), r * np.log(np.tan(np.pi / 4 + np.deg2rad(lat) / 2))


def parcel_mask(geometry, lon, lat):
    if geometry["type"] == "Polygon":
        polygons = [geometry["coordinates"]]
    else:
        polygons = geometry["coordinates"]
    cx, cy = web_mercator(lat, lon)
    mask = np.zeros((IMAGE_SIZE, IMAGE_SIZE), dtype=np.uint8)
    for polygon in polygons:
        rings = [polygon[0]]  # outer boundary; parcels have no interior holes in this layer
        for ring in rings:
            pts = []
            for px_lon, px_lat, *_ in ring:
                x, y = web_mercator(px_lat, px_lon)
                px = int(round(IMAGE_SIZE / 2 + (x - cx) / PIXEL_SIZE_M))
                py = int(round(IMAGE_SIZE / 2 - (y - cy) / PIXEL_SIZE_M))
                pts.append((px, py))
            if len(pts) >= 3:
                cv2.fillPoly(mask, [np.asarray(pts, dtype=np.int32)], 255)
    return mask


def highlighted_mask(original, annotated):
    delta = annotated.astype(np.int16) - original.astype(np.int16)
    # The roof fill changes BGR roughly by (+9,+64,+77); the detector outline
    # and label are brighter, so the parcel overlap check is robust to both.
    result = ((delta[:, :, 1] > 28) & (delta[:, :, 2] > 35)
              & (delta[:, :, 2] > delta[:, :, 0] * 2)).astype(np.uint8) * 255
    # Remove the center marker/crosshair from the highlighted-pixel measurement.
    cv2.circle(result, (IMAGE_SIZE // 2, IMAGE_SIZE // 2), 24, 0, -1)
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", choices=("train", "validation"), required=True)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--min-parcel-overlap", type=float, default=0.50)
    args = parser.parse_args()

    sys.path.insert(0, str(ROOT))
    os.environ.setdefault("DJANGO_SETTINGS_MODULE", "solarleads.settings")
    import django
    django.setup()
    from detector.rooftop_annotator import _annotate_rooftop

    rows = [r for r in csv.DictReader((DATA / "manifest.csv").open(newline=""))
            if r["split"] == args.split]
    feature_rows = json.loads((DATA / "parcels.geojson").read_text())["features"]
    geometries = {f["properties"]["sample_id"]: f["geometry"] for f in feature_rows}
    counts = Counter()
    details = []
    for row in rows:
        raw = np.asarray(Image.open(DATA / row["raw_image_file"]).convert("RGB"))
        bgr = cv2.cvtColor(raw, cv2.COLOR_RGB2BGR)
        annotated, verification = _annotate_rooftop(bgr)
        roi = parcel_mask(geometries[row["sample_id"]], float(row["target_lon"]),
                          float(row["target_lat"]))
        selected = highlighted_mask(bgr, annotated)
        selected_count = cv2.countNonZero(selected)
        inside_count = cv2.countNonZero(cv2.bitwise_and(selected, roi))
        overlap = inside_count / selected_count if selected_count else 0.0
        predicted = (verification["in_frame"] and selected_count > 0
                     and overlap >= args.min_parcel_overlap)
        actual = row["expected_target"] == "residential_house"
        counts[(actual, bool(predicted))] += 1
        details.append({
            "sample_id": row["sample_id"], "target_class": row["target_class"],
            "expected_residential_house": str(actual).lower(),
            "roof_in_frame": str(bool(verification["in_frame"])).lower(),
            "highlight_pixels_in_parcel": inside_count,
            "highlight_pixels_total": selected_count,
            "parcel_overlap_fraction": round(overlap, 4),
            "predicted_residential_house": str(bool(predicted)).lower(),
            "correct": str(bool(predicted) == actual).lower(),
            "detection_method": verification.get("roof_detection_method", ""),
            "reason": verification.get("reason", ""),
        })

    tp, fn = counts[True, True], counts[True, False]
    fp, tn = counts[False, True], counts[False, False]
    n = len(rows)
    print(f"Split: {args.split}; parcels: {n}; parcel-overlap threshold: {args.min_parcel_overlap:.2f}")
    print(f"Confusion (TP/FN/FP/TN): {tp}/{fn}/{fp}/{tn}")
    print(f"Accuracy: {(tp+tn)/n:.3f}; precision: {tp/(tp+fp) if tp+fp else 0:.3f}; "
          f"recall: {tp/(tp+fn) if tp+fn else 0:.3f}; specificity: {tn/(tn+fp) if tn+fp else 0:.3f}")
    for item in details:
        if item["correct"] == "false":
            print(f"ERROR {item['sample_id']}: target={item['target_class']} "
                  f"roof_in_frame={item['roof_in_frame']} "
                  f"parcel_overlap={item['parcel_overlap_fraction']}")
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=details[0].keys())
            writer.writeheader()
            writer.writerows(details)


if __name__ == "__main__":
    main()
