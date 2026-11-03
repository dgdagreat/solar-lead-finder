"""Prepare a spatially split rooftop classifier dataset from public 2018 labels.

The command downloads the 5.4 MB Queens PV labels archive, obtains small image
chips from New York State's 2018 orthophoto service, and makes train/val/test
ImageFolder directories. Google Maps imagery is not used.
"""
import csv
import hashlib
import io
import math
import random
import time
import zipfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import requests
import shapefile
from PIL import Image
from django.conf import settings
from django.core.management.base import BaseCommand, CommandError


LABELS_URL = "https://zenodo.org/records/15084216/files/NYQ18-labels.zip?download=1"
BUILDINGS_URL = "https://gisservices.its.ny.gov/arcgis/rest/services/BuildingFootprints/FeatureServer/2/query"
TILE_URL = "https://orthos.its.ny.gov/arcgis/rest/services/wms/2018/MapServer/tile/19/{row}/{col}"
TILE_RESOLUTION_M = 0.29858214164761665
WEB_MERCATOR_ORIGIN_M = 20037508.342789244
EVAL_POSITIVE_TILES = {
    "027177", "025205", "997210", "025177", "020215", "005200",
    "027180", "005195", "022202", "002200", "022225", "025182",
    "025225", "020205", "027182", "017212", "025222", "020207",
}
EVAL_NEGATIVE_IDS = [
    4676628, 4675874, 5120253, 4710191, 5120540, 5120275, 4675766,
    5121816, 5160745, 4650898, 5123480, 4622777, 5121557, 5122167,
    5150867, 4686022, 5121438, 4710344, 4718483, 5166502,
]


def _spatial_split(x, y):
    """Stable 70/15/15 split by 5,000-foot blocks to limit geographic leakage."""
    block = f"{math.floor(x / 5000)}:{math.floor(y / 5000)}".encode()
    bucket = int(hashlib.sha256(block).hexdigest()[:8], 16) % 20
    return "train" if bucket < 14 else ("val" if bucket < 17 else "test")


def _polygon_area(points):
    if len(points) < 4:
        return 0.0
    return abs(sum(points[i][0] * points[i + 1][1] - points[i + 1][0] * points[i][1]
                   for i in range(len(points) - 1))) / 2


def _arcgis_query(params, timeout=90):
    r = requests.get(BUILDINGS_URL, params=params, timeout=timeout)
    r.raise_for_status()
    data = r.json()
    if "error" in data:
        raise RuntimeError(data["error"])
    return data.get("features", [])


def _download_labels(work_dir):
    archive = work_dir / "NYQ18-labels.zip"
    if not archive.exists():
        with requests.get(LABELS_URL, stream=True, timeout=90) as response:
            response.raise_for_status()
            with archive.open("wb") as f:
                for chunk in response.iter_content(1024 * 1024):
                    if chunk:
                        f.write(chunk)
    label_dir = work_dir / "labels"
    label_dir.mkdir(exist_ok=True)
    with zipfile.ZipFile(archive) as zf:
        for name in zf.namelist():
            if name.startswith("shapefile/") and not name.endswith("/"):
                dest = label_dir / Path(name).name
                if not dest.exists():
                    dest.write_bytes(zf.read(name))
    return label_dir / "NYQ18-labels.shp"


def _get_buildings(rng, bounds):
    xmin, ymin, xmax, ymax = bounds
    # Small overlapping searches keep the public feature service response bounded.
    centers = [(rng.uniform(xmin, xmax), rng.uniform(ymin, ymax)) for _ in range(8)]
    boxes = set()
    for x, y in centers:
        boxes.add((round(x / 2500) * 2500, round(y / 2500) * 2500))
    features = {}
    def fetch_box(box):
        cx, cy = box
        try:
            return box, _arcgis_query({
                "where": "1=1", "outFields": "*",
                "geometry": f"{cx-2500},{cy-2500},{cx+2500},{cy+2500}",
                "geometryType": "esriGeometryEnvelope", "inSR": "2263",
                "outSR": "2263", "returnGeometry": "true", "f": "geojson",
                "resultRecordCount": "1200",
            })
        except Exception as exc:
            return box, exc

    with ThreadPoolExecutor(max_workers=6) as pool:
        futures = [pool.submit(fetch_box, box) for box in sorted(boxes)]
        for future in as_completed(futures):
            box, result = future.result()
            if isinstance(result, Exception):
                print(f"Building query skipped at {box}: {result}", flush=True)
                continue
            found = result
            for item in found:
                features[item.get("id", item.get("properties", {}).get("OBJECTID"))] = item
    return list(features.values())


def _eval_exclusion_boxes():
    try:
        features = _arcgis_query({
            "objectIds": ",".join(map(str, EVAL_NEGATIVE_IDS)),
            "outFields": "OBJECTID", "returnGeometry": "true",
            "outSR": "2263", "f": "geojson",
        })
        if len(features) != len(EVAL_NEGATIVE_IDS):
            raise RuntimeError(f"Resolved {len(features)} of {len(EVAL_NEGATIVE_IDS)} evaluation roofs")
        return [feature["geometry"]["coordinates"][0] for feature in features]
    except Exception as exc:
        raise CommandError(f"Could not exclude the previous evaluation roofs: {exc}") from exc


def _building_candidates(features, solar_boxes, holdout_rings, rng):
    pv = np.asarray(solar_boxes, dtype=np.float64)
    holdout = []
    for ring in holdout_rings:
        xs, ys = [p[0] for p in ring], [p[1] for p in ring]
        holdout.append((min(xs), min(ys), max(xs), max(ys)))
    candidates = []
    for item in features:
        rings = item.get("geometry", {}).get("coordinates", [])
        if not rings:
            continue
        points = rings[0]
        if len(points) < 5:
            continue
        xs, ys = [p[0] for p in points], [p[1] for p in points]
        box = (min(xs), min(ys), max(xs), max(ys))
        area = _polygon_area(points)
        if not (400 <= area <= 12000 and 15 <= box[2]-box[0] <= 120 and 15 <= box[3]-box[1] <= 120):
            continue
        # A negative roof must have no labeled PV on the building or in the
        # detector's central 150-foot analysis region.
        if len(pv):
            near = ((pv[:, 0] <= box[2] + 75) & (pv[:, 2] >= box[0] - 75) &
                    (pv[:, 1] <= box[3] + 75) & (pv[:, 3] >= box[1] - 75))
            if near.any():
                continue
        if any(not (box[2] + 200 < b[0] or box[0] - 200 > b[2] or
                    box[3] + 200 < b[1] or box[1] - 200 > b[3]) for b in holdout):
            continue
        x, y = sum(xs[:-1]) / max(len(xs)-1, 1), sum(ys[:-1]) / max(len(ys)-1, 1)
        candidates.append({"x": x, "y": y, "tile": "building", "source_id": item.get("id", "")})
    rng.shuffle(candidates)
    return candidates


def _positive_candidates(shp_path, per_split, rng):
    reader = shapefile.Reader(str(shp_path))
    candidates = []
    for shape, record in zip(reader.shapes(), reader.records()):
        tile = str(record[0]).strip()
        if tile in EVAL_POSITIVE_TILES:
            continue
        box = shape.bbox
        width, height = box[2] - box[0], box[3] - box[1]
        if not (8 <= width <= 180 and 8 <= height <= 180):
            continue
        x, y = (box[0] + box[2]) / 2, (box[1] + box[3]) / 2
        candidates.append({"x": x, "y": y, "tile": tile, "source_id": f"{tile}:{len(candidates)}"})
    rng.shuffle(candidates)
    selected = {s: [] for s in ("train", "val", "test")}
    occupied = {s: [] for s in selected}
    for item in candidates:
        split = _spatial_split(item["x"], item["y"])
        if len(selected[split]) >= per_split:
            continue
        if any((item["x"] - x)**2 + (item["y"] - y)**2 < 100**2 for x, y in occupied[split]):
            continue
        selected[split].append(item)
        occupied[split].append((item["x"], item["y"]))
        if all(len(selected[s]) >= per_split for s in selected):
            break
    return selected


def _negative_by_split(candidates, per_split):
    selected = {s: [] for s in ("train", "val", "test")}
    occupied = {s: [] for s in selected}
    for item in candidates:
        split = _spatial_split(item["x"], item["y"])
        if len(selected[split]) >= per_split:
            continue
        if any((item["x"] - x)**2 + (item["y"] - y)**2 < 100**2 for x, y in occupied[split]):
            continue
        selected[split].append(item)
        occupied[split].append((item["x"], item["y"]))
        if all(len(selected[s]) >= per_split for s in selected):
            break
    return selected


class Command(BaseCommand):
    help = "Build a spatially split CNN dataset from independently labeled Queens 2018 imagery."

    def add_arguments(self, parser):
        parser.add_argument("--per-split", type=int, default=300,
                            help="Examples per class in each split (default: 300).")
        parser.add_argument("--output", default="data/queens_external_v1")
        parser.add_argument("--workers", type=int, default=8)
        parser.add_argument("--seed", type=int, default=3719)

    def handle(self, *args, **opts):
        per_split = opts["per_split"]
        if per_split < 20:
            raise CommandError("Use at least 20 examples per split for a useful training run.")
        rng = random.Random(opts["seed"])
        out = Path(settings.BASE_DIR) / opts["output"]
        work_dir = Path(settings.BASE_DIR) / "data" / "source" / "queens2018"
        work_dir.mkdir(parents=True, exist_ok=True)
        shp_path = _download_labels(work_dir)
        reader = shapefile.Reader(str(shp_path))
        solar_boxes = [s.bbox for s in reader.shapes()]
        xmin = min(b[0] for b in solar_boxes); ymin = min(b[1] for b in solar_boxes)
        xmax = max(b[2] for b in solar_boxes); ymax = max(b[3] for b in solar_boxes)
        positives = _positive_candidates(shp_path, per_split, rng)
        missing = {s: per_split-len(positives[s]) for s in positives}
        if any(missing.values()):
            raise CommandError(f"Not enough positive roofs per spatial split: {missing}")
        self.stdout.write(f"Selected positive roofs: { {k: len(v) for k,v in positives.items()} }")

        buildings = _get_buildings(rng, (xmin, ymin, xmax, ymax))
        holdouts = _eval_exclusion_boxes()
        negatives = _negative_by_split(
            _building_candidates(buildings, solar_boxes, holdouts, rng), per_split)
        missing = {s: per_split-len(negatives[s]) for s in negatives}
        if any(missing.values()):
            raise CommandError(f"Not enough clear negative roofs per spatial split: {missing}")
        self.stdout.write(f"Selected negative roofs: { {k: len(v) for k,v in negatives.items()} }")

        samples = []
        for cls, groups in (("solar", positives), ("no_solar", negatives)):
            for split, items in groups.items():
                for index, item in enumerate(items):
                    samples.append((split, cls, index, item))
        from pyproj import Transformer
        to_mercator = Transformer.from_crs(2263, 3857, always_xy=True)
        to_wgs84 = Transformer.from_crs(3857, 4326, always_xy=True)
        tile_span = TILE_RESOLUTION_M * 256
        positions, tile_keys = {}, set()
        for sample in samples:
            item = sample[3]
            mx, my = to_mercator.transform(item["x"], item["y"])
            tx, ty = (mx + WEB_MERCATOR_ORIGIN_M) / tile_span, (WEB_MERCATOR_ORIGIN_M - my) / tile_span
            col, row = math.floor(tx), math.floor(ty)
            positions[(sample[0], sample[1], sample[2])] = (tx-col, ty-row, mx, my)
            tile_keys.update((row+dy, col+dx) for dy in (-1, 0, 1) for dx in (-1, 0, 1))

        tile_dir = work_dir / "tiles_z19"
        tile_dir.mkdir(exist_ok=True)
        def tile_path(key):
            return tile_dir / f"{key[0]}_{key[1]}.jpg"
        def fetch_tile(key):
            path = tile_path(key)
            if path.exists():
                return
            url = TILE_URL.format(row=key[0], col=key[1])
            last_error = None
            for attempt in range(3):
                try:
                    response = requests.get(url, timeout=45)
                    response.raise_for_status()
                    image = Image.open(io.BytesIO(response.content)).convert("RGB")
                    if image.size != (256, 256):
                        raise ValueError(f"Unexpected tile shape {image.size}")
                    image.save(path, quality=90)
                    return
                except Exception as exc:
                    last_error = exc
                    if attempt < 2:
                        time.sleep(1 + attempt)
            raise last_error

        self.stdout.write(f"Fetching {len(tile_keys)} cached 2018 aerial tiles …")
        failed_tiles = set()
        with ThreadPoolExecutor(max_workers=opts["workers"]) as pool:
            futures = {pool.submit(fetch_tile, key): key for key in tile_keys}
            for done, future in enumerate(as_completed(futures), 1):
                key = futures[future]
                try:
                    future.result()
                except Exception as exc:
                    failed_tiles.add(key)
                    self.stderr.write(f"Tile {key} failed: {exc}")
                if done % 100 == 0 or done == len(futures):
                    self.stdout.write(f"Fetched {done}/{len(futures)} tiles")

        out.mkdir(parents=True, exist_ok=True)
        manifest_path = out / "manifest.csv"
        with manifest_path.open("w", newline="") as manifest_file:
            writer = csv.DictWriter(manifest_file, fieldnames=[
                "path", "label", "split", "x_2263", "y_2263", "source_tile", "source_id",
            ])
            writer.writeheader()
            saved = {s: {"solar": 0, "no_solar": 0} for s in ("train", "val", "test")}

            for split, cls, index, item in samples:
                frac_x, frac_y, mx, my = positions[(split, cls, index)]
                center_col = math.floor((mx + WEB_MERCATOR_ORIGIN_M) / tile_span)
                center_row = math.floor((WEB_MERCATOR_ORIGIN_M - my) / tile_span)
                neighborhood = [(center_row+dy, center_col+dx)
                                for dy in (-1, 0, 1) for dx in (-1, 0, 1)]
                if any(key in failed_tiles for key in neighborhood):
                    continue
                mosaic = Image.new("RGB", (768, 768))
                for dy in (-1, 0, 1):
                    for dx in (-1, 0, 1):
                        with Image.open(tile_path((center_row+dy, center_col+dx))) as tile:
                            mosaic.paste(tile.convert("RGB"), ((dx+1)*256, (dy+1)*256))
                center_x, center_y = 256 + frac_x*256, 256 + frac_y*256
                latitude = to_wgs84.transform(mx, my)[1]
                ground_m = 240 * 0.304800609601219
                half = round(ground_m / math.cos(math.radians(latitude)) / TILE_RESOLUTION_M / 2)
                crop = mosaic.crop((round(center_x)-half, round(center_y)-half,
                                    round(center_x)+half, round(center_y)+half))
                if crop.size != (2*half, 2*half):
                    continue
                target_dir = out / split / cls
                target_dir.mkdir(parents=True, exist_ok=True)
                path = target_dir / f"{cls}_{index:04d}.jpg"
                crop.resize((640, 640), Image.Resampling.LANCZOS).save(
                    path, quality=92, optimize=True)
                writer.writerow({
                    "path": str(path.relative_to(out)), "label": cls, "split": split,
                    "x_2263": item["x"], "y_2263": item["y"],
                    "source_tile": item["tile"], "source_id": item["source_id"],
                })
                saved[split][cls] += 1

        notice = out / "NOTICE.md"
        notice.write_text(
            "# Queens 2018 roof classifier dataset\n\n"
            "Labels: Ranalli et al., *Segmentation Dataset of Labeled Photovoltaic "
            "Installations for Orthographic Aerial Imagery in Queens, New York*, "
            "Zenodo 15084216. The original 2018 imagery is supplied under CC BY 4.0; "
            "PV labels were manually created and reviewed by the study team.\n\n"
            "Images: New York State 2018 orthophoto ArcGIS MapServer level-19 tiles, "
            "assembled into 240-foot chips. Attribution: NYS ITS Geospatial Services / Westchester "
            "County GIS. This folder is generated locally and is excluded from git.\n\n"
            "The 20 previously evaluated positive source tiles and 20 previously "
            "evaluated negative building footprints are excluded. Train, validation, "
            "and test sets are separated by 5,000-foot geographic blocks.\n"
        )
        self.stdout.write(self.style.SUCCESS(f"Saved examples: {saved}"))
        if any(saved[s][c] != per_split for s in saved for c in ("solar", "no_solar")):
            raise CommandError("Some image cells failed; rerun after checking service errors.")
