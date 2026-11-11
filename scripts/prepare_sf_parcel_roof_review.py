"""Build a parcel-marked San Francisco residential-roof review set.

Positive candidates come from current single-unit residential parcels. Negative
controls come from commercial and open-space/parking parcels. Each aerial chip
marks the parcel boundary and its official address point (or parcel centroid
for controls). A land-use category is a sampling stratum, not visual roof
ground truth; roof labels remain blank for review.

Public sources:
  * DataSF San Francisco Land Use (c5ge-t6pj)
  * DataSF Addresses with Units (ramy-di5m)
  * USGS NAIP ImageServer
"""
import csv
import argparse
import json
import math
import random
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from io import BytesIO
from pathlib import Path

import requests
from PIL import Image, ImageDraw, ImageFont


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "data" / "sf_parcel_roof_review"
LANDUSE_URL = "https://data.sf.gov/resource/c5ge-t6pj.json"
ADDRESS_URL = "https://data.sf.gov/resource/ramy-di5m.json"
NAIP_URL = "https://imagery.nationalmap.gov/arcgis/rest/services/USGSNAIPImagery/ImageServer"
IMAGE_SIZE = 640
# Approximate the live Google Static Maps zoom-20 ground scale near SF. NAIP
# source imagery is coarser and is upsampled; this adds no new image detail.
PIXEL_SIZE_M = 0.12
HALF_EXTENT_M = IMAGE_SIZE * PIXEL_SIZE_M / 2
CITY_SPLITS = {
    "train": ("centroid_l >= 37.78", 37.78, 37.84),
    "validation": ("centroid_l >= 37.75 AND centroid_l < 37.78", 37.75, 37.78),
    "test": ("centroid_l < 37.75", 37.70, 37.75),
}
SPLIT_CODES = {"train": "TR", "validation": "VA", "test": "TE"}
PER_CLASS_PER_SPLIT = {"residential_single_unit": 10, "commercial": 5, "open_space_or_parking": 5}
RANDOM_SEED = 20261110


def web_mercator(lat, lon):
    r = 6378137.0
    return r * math.radians(lon), r * math.log(math.tan(math.pi / 4 + math.radians(lat) / 2))


def api_json(session, url, params):
    for attempt in range(5):
        try:
            response = session.get(url, params=params, timeout=45)
            if response.status_code == 429 or response.status_code >= 500:
                if attempt < 4:
                    time.sleep(1.5 * (attempt + 1))
                    continue
            response.raise_for_status()
            return response.json()
        except requests.RequestException:
            if attempt == 4:
                raise
            time.sleep(1.5 * (attempt + 1))
    raise RuntimeError(f"Request retries exhausted for {url}")


def point_in_ring(lon, lat, ring):
    inside = False
    j = len(ring) - 1
    for i, (xi, yi, *_) in enumerate(ring):
        xj, yj = ring[j][:2]
        if ((yi > lat) != (yj > lat)) and lon < (xj - xi) * (lat - yi) / (yj - yi + 1e-30) + xi:
            inside = not inside
        j = i
    return inside


def point_in_polygon(lon, lat, geometry):
    if geometry["type"] == "Polygon":
        polygons = [geometry["coordinates"]]
    elif geometry["type"] == "MultiPolygon":
        polygons = geometry["coordinates"]
    else:
        return False
    for polygon in polygons:
        if point_in_ring(lon, lat, polygon[0]) and not any(point_in_ring(lon, lat, ring) for ring in polygon[1:]):
            return True
    return False


def _candidate_query(session, split, category, limit=1000):
    lat_where, _, _ = CITY_SPLITS[split]
    base = f"geography_type='parcel' AND ({lat_where})"
    strata = {
        "residential_single_unit": "resunits=1 AND resunits_s=0 AND total_comm=0 AND open_space=false AND parking_lo=false AND garage=false",
        "commercial": "resunits=0 AND total_comm>0 AND open_space=false AND parking_lo=false",
        "open_space_or_parking": "resunits=0 AND total_comm=0 AND (open_space=true OR parking_lo=true)",
    }
    fields = "the_geom,ludb_id,centroid_l,centroid_1,resunits,total_comm,open_space,parking_lo,geography_type,mapblklot"
    return api_json(session, LANDUSE_URL, {
        "$select": fields, "$where": f"{base} AND ({strata[category]})",
        "$limit": limit, "$order": "ludb_id",
    })


def _address_for_parcel(session, parcel):
    geom = parcel["the_geom"]
    if geom["type"] == "Polygon":
        polygons = [geom["coordinates"]]
    else:
        polygons = geom["coordinates"]
    outer_points = [pt for poly in polygons for pt in poly[0]]
    xs, ys = [pt[0] for pt in outer_points], [pt[1] for pt in outer_points]
    pad = 0.00003
    rows = api_json(session, ADDRESS_URL, {
        "$select": "address,latitude,longitude,point",
        "$where": f"within_box(point,{max(ys)+pad},{min(xs)-pad},{min(ys)-pad},{max(xs)+pad})",
        "$limit": 100,
    })
    inside = []
    for row in rows:
        try:
            lon, lat = float(row["longitude"]), float(row["latitude"])
        except (TypeError, ValueError, KeyError):
            continue
        if point_in_polygon(lon, lat, geom):
            inside.append((lon, lat, row.get("address", "")))
    if not inside:
        return None
    center_lon = float(parcel["centroid_1"])
    center_lat = float(parcel["centroid_l"])
    return min(inside, key=lambda p: (p[0] - center_lon) ** 2 + (p[1] - center_lat) ** 2)


def _select_candidates(session):
    rng = random.Random(RANDOM_SEED)
    selected, seen = [], set()
    for split in CITY_SPLITS:
        for category, needed in PER_CLASS_PER_SPLIT.items():
            candidates = _candidate_query(session, split, category)
            rng.shuffle(candidates)
            accepted = 0
            for parcel in candidates:
                if accepted >= needed:
                    break
                pid = parcel["mapblklot"]
                if pid in seen:
                    continue
                geometry = parcel["the_geom"]
                center_lon, center_lat = float(parcel["centroid_1"]), float(parcel["centroid_l"])
                # Center each target on the parcel centroid. The yellow outline
                # makes the intended property visible even when the centroid
                # lands on a side yard rather than directly on its roof.
                target = (center_lon, center_lat, "")
                target_method = "parcel_centroid"
                seen.add(pid)
                accepted += 1
                selected.append({
                    "split": split, "target_class": category,
                    "expected_target": "residential_house" if category == "residential_single_unit" else "nonresidential",
                    "parcel_id": pid, "landuse_id": parcel["ludb_id"],
                    "parcel_geometry": geometry,
                    "target_lon": target[0], "target_lat": target[1],
                    "target_method": target_method,
                    "imagery_year": "", "source_item": "",
                })
            if accepted < needed:
                raise RuntimeError(f"Only found {accepted}/{needed} {category} candidates in {split}")
    return selected


def _fetch_naip(session, sample):
    x, y = web_mercator(sample["target_lat"], sample["target_lon"])
    bbox = f"{x-HALF_EXTENT_M},{y-HALF_EXTENT_M},{x+HALF_EXTENT_M},{y+HALF_EXTENT_M}"
    mosaic = {"mosaicMethod": "esriMosaicAttribute", "sortField": "Year", "sortValue": "3000", "ascending": False}
    params = {
        "bbox": bbox, "bboxSR": 3857, "imageSR": 3857,
        "size": f"{IMAGE_SIZE},{IMAGE_SIZE}", "format": "jpg", "f": "image",
        "mosaicRule": json.dumps(mosaic),
    }
    response = None
    for attempt in range(5):
        try:
            response = session.get(f"{NAIP_URL}/exportImage", params=params, timeout=90)
            if response.status_code == 429 or response.status_code >= 500:
                if attempt < 4:
                    time.sleep(1.5 * (attempt + 1))
                    continue
            response.raise_for_status()
            break
        except requests.RequestException:
            if attempt == 4:
                raise
            time.sleep(1.5 * (attempt + 1))
    image = Image.open(BytesIO(response.content)).convert("RGB")
    if image.size != (IMAGE_SIZE, IMAGE_SIZE):
        raise ValueError(f"Unexpected NAIP dimensions {image.size}")
    # Identify scene vintage for provenance.
    identify_data = api_json(session, f"{NAIP_URL}/identify", {
        "geometry": json.dumps({"x": x, "y": y, "spatialReference": {"wkid": 3857}}),
        "geometryType": "esriGeometryPoint", "sr": 3857, "returnGeometry": "false", "f": "json",
    })
    features = identify_data.get("catalogItems", {}).get("features", [])
    attrs = [f.get("attributes", {}) for f in features]
    attrs.sort(key=lambda item: (item.get("Year") or 0, item.get("Name") or ""), reverse=True)
    if attrs:
        sample["source_item"] = attrs[0].get("Name", "")
        sample["imagery_year"] = attrs[0].get("Year", "")
    return image


def _project(lon, lat, center_lon, center_lat):
    x, y = web_mercator(lat, lon)
    cx, cy = web_mercator(center_lat, center_lon)
    px = IMAGE_SIZE / 2 + (x - cx) / PIXEL_SIZE_M
    py = IMAGE_SIZE / 2 - (y - cy) / PIXEL_SIZE_M
    return px, py


def _draw_ring(draw, ring, center_lon, center_lat, color, width):
    points = [_project(pt[0], pt[1], center_lon, center_lat) for pt in ring]
    for a, b in zip(points, points[1:] + points[:1]):
        draw.line((a[0], a[1], b[0], b[1]), fill=color, width=width)


def _annotate(image, sample):
    image = image.copy()
    draw = ImageDraw.Draw(image)
    geom = sample["parcel_geometry"]
    polygons = [geom["coordinates"]] if geom["type"] == "Polygon" else geom["coordinates"]
    for poly in polygons:
        _draw_ring(draw, poly[0], sample["target_lon"], sample["target_lat"], (255, 230, 0), 3)
    cx = cy = IMAGE_SIZE / 2
    draw.ellipse((cx-10, cy-10, cx+10, cy+10), fill=(255, 25, 25), outline=(255, 255, 255), width=2)
    draw.line((cx-18, cy, cx+18, cy), fill=(255, 255, 255), width=2)
    draw.line((cx, cy-18, cx, cy+18), fill=(255, 255, 255), width=2)
    return image


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--reuse-existing-images", action="store_true",
                        help="Reuse annotated image chips already in OUT/images")
    args = parser.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "images").mkdir(exist_ok=True)
    (OUT / "raw").mkdir(exist_ok=True)
    query_session = requests.Session()
    query_session.headers["User-Agent"] = "SolarLeadFinder-SF-Parcel-Roof-Review/1.0"
    samples = _select_candidates(query_session)
    manifest_rows = []
    split_panels = {split: [] for split in CITY_SPLITS}
    parcels = {"type": "FeatureCollection", "features": []}

    for index, sample in enumerate(samples, 1):
        cls = {"residential_single_unit": "H", "commercial": "C", "open_space_or_parking": "O"}[sample["target_class"]]
        sample["sample_id"] = f"SF-{SPLIT_CODES[sample['split']]}-{cls}-{sum(1 for s in samples[:index] if s['split']==sample['split'] and s['target_class']==sample['target_class']):02d}"

    def prepare(sample):
        image_path = OUT / "images" / f"{sample['sample_id']}.jpg"
        raw_path = OUT / "raw" / f"{sample['sample_id']}.jpg"
        if args.reuse_existing_images and raw_path.exists():
            sample["image_file"] = f"images/{sample['sample_id']}.jpg"
            sample["raw_image_file"] = f"raw/{sample['sample_id']}.jpg"
            sample["source_item"] = ""
            sample["imagery_year"] = ""
            sample["source_service"] = NAIP_URL
            sample["resolution_m_per_pixel"] = PIXEL_SIZE_M
            sample["native_source_resolution_m_per_pixel"] = 0.3
            sample["image_width"] = IMAGE_SIZE
            sample["image_height"] = IMAGE_SIZE
            sample["roof_label"] = ""
            sample["review_confidence"] = ""
            sample["review_notes"] = ""
            return sample
        worker_session = requests.Session()
        worker_session.headers["User-Agent"] = "SolarLeadFinder-SF-Parcel-Roof-Review/1.0"
        image = _fetch_naip(worker_session, sample)
        image.save(raw_path, quality=95)
        _annotate(image, sample).save(image_path, quality=95)
        sample["image_file"] = f"images/{sample['sample_id']}.jpg"
        sample["raw_image_file"] = f"raw/{sample['sample_id']}.jpg"
        sample["source_service"] = NAIP_URL
        sample["resolution_m_per_pixel"] = PIXEL_SIZE_M
        sample["native_source_resolution_m_per_pixel"] = 0.3
        sample["image_width"] = IMAGE_SIZE
        sample["image_height"] = IMAGE_SIZE
        sample["roof_label"] = ""
        sample["review_confidence"] = ""
        sample["review_notes"] = ""
        worker_session.close()
        return sample

    completed = {}
    with ThreadPoolExecutor(max_workers=6) as pool:
        futures = {pool.submit(prepare, sample): sample["sample_id"] for sample in samples}
        for future in as_completed(futures):
            sample = future.result()
            completed[sample["sample_id"]] = sample
            print(f"Downloaded {sample['sample_id']} ({len(completed)}/{len(samples)})", flush=True)

    for sample in samples:
        sample = completed[sample["sample_id"]]
        parcel_feature = {"type": "Feature", "geometry": sample.pop("parcel_geometry"), "properties": {
            "sample_id": sample["sample_id"], "parcel_id": sample["parcel_id"],
            "target_class": sample["target_class"], "split": sample["split"],
        }}
        parcels["features"].append(parcel_feature)
        manifest_rows.append(sample)

        # Put sample id and target type directly on contact-sheet tiles.
        tile = Image.new("RGB", (660, 690), "white")
        shown = Image.open(OUT / sample["image_file"]).convert("RGB")
        shown.thumbnail((640, 640))
        tile.paste(shown, ((660-shown.width)//2, 35))
        d = ImageDraw.Draw(tile)
        d.text((10, 8), f"{sample['sample_id']} | {sample['target_class']}", fill="black")
        d.text((10, 20), "Yellow outline = parcel; red cross = parcel centroid", fill=(120, 0, 0))
        split_panels[sample["split"]].append(tile)
        print(f"{sample['sample_id']} {sample['target_class']} year={sample['imagery_year']}", flush=True)

    fields = ["sample_id", "split", "target_class", "expected_target", "parcel_id", "landuse_id",
              "target_lat", "target_lon", "target_method", "image_file",
              "raw_image_file",
              "imagery_year", "source_item", "source_service", "resolution_m_per_pixel",
              "native_source_resolution_m_per_pixel",
              "image_width", "image_height", "roof_label", "review_confidence", "review_notes"]
    with (OUT / "manifest.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(manifest_rows)
    (OUT / "parcels.geojson").write_text(json.dumps(parcels))
    for split, tiles in split_panels.items():
        cols = 3
        rows = math.ceil(len(tiles) / cols)
        sheet = Image.new("RGB", (cols * 660, rows * 690), "white")
        for i, tile in enumerate(tiles):
            sheet.paste(tile, ((i % cols) * 660, (i // cols) * 690))
        sheet.save(OUT / f"{split}_review.jpg", quality=92)
    readme = f"""# San Francisco parcel-marked residential-roof review\n\nGenerated {len(samples)} chips from official City of San Francisco land-use parcels and USGS NAIP aerial imagery. Files in `images/` show the selected parcel outlined in yellow and its centroid in red; matching clean imagery without review marks is in `raw/`.\n\nChips are rendered at a nominal 0.12 m/pixel to approximate the app's Google Maps zoom-20 scale near San Francisco. NAIP source imagery is about 0.3 m/pixel, so the requested output is upsampled; this does not add image detail.\n\nThe target class is a residential house roof. Residential candidates are one-unit residential parcels with no recorded commercial use; negative controls are commercial and open-space/parking parcels. Land-use strata identify the intended property type but are not visual roof labels. Fill `roof_label` only after reviewing the marked parcel and its NAIP image. Use `roof`, `nonroof`, or `uncertain`.\n\nThe split is geographic and assigned before labeling: north (train), central (validation), south (test). Keep test predictions sealed until the detector and validation decisions are frozen. `parcels.geojson` contains parcel outlines and IDs.\n\nImagery: USGS NAIP ImageServer. Parcel metadata: DataSF San Francisco Land Use.\n"""
    (OUT / "README.md").write_text(readme)


if __name__ == "__main__":
    main()
