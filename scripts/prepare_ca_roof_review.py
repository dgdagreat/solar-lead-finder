"""Download an independent California roof-review batch from public USGS NAIP.

The batch is a manually labeled roof-presence set, not a solar-status set.
Sample centers are split by city before review to prevent nearby image leakage.
Google Maps imagery is never requested or included.
"""
import csv
import json
import math
import time
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from io import BytesIO
from pathlib import Path

import requests
from PIL import Image, ImageDraw


SERVICE = (
    "https://imagery.nationalmap.gov/arcgis/rest/services/"
    "USGSNAIPImagery/ImageServer"
)
OUTPUT = Path(__file__).resolve().parents[1] / "data" / "ca_roof_review_v2"
PIXEL_SIZE_M = 0.3
IMAGE_SIZE = 640
HALF_EXTENT_M = PIXEL_SIZE_M * IMAGE_SIZE / 2
MOSAIC_RULE = {
    "mosaicMethod": "esriMosaicAttribute",
    "sortField": "Year",
    "sortValue": "3000",
    "ascending": False,
}

# Each city is absent from the prior four-city California pilot. Four locations
# are residential sampling candidates and four are park/road/open-space controls.
# These strata are sampling intent only: human image review assigns roof_label.
CITY_SAMPLES = {
    "Riverside": {
        "split": "train",
        "base": (33.9680, -117.3630),
        "residential_offsets": [
            (-0.0014, -0.0014), (-0.0014, 0.0014),
            (0.0014, -0.0014), (0.0014, 0.0014),
        ],
        "controls": [
            (33.9820, -117.3770), (33.9870, -117.3630),
            (33.9520, -117.3920), (33.9400, -117.3650),
        ],
    },
    "Bakersfield": {
        "split": "train",
        "base": (35.3730, -119.0190),
        "residential_offsets": [
            (-0.0015, -0.0015), (-0.0015, 0.0015),
            (0.0015, -0.0015), (0.0015, 0.0015),
        ],
        "controls": [
            (35.4010, -119.0830), (35.4090, -119.0400),
            (35.3500, -119.0720), (35.3900, -119.0090),
        ],
    },
    "San Jose": {
        "split": "validation",
        "base": (37.3020, -121.9000),
        "residential_offsets": [
            (-0.0014, -0.0014), (-0.0014, 0.0014),
            (0.0014, -0.0014), (0.0014, 0.0014),
        ],
        "controls": [
            (37.3280, -121.8870), (37.3370, -121.9000),
            (37.3150, -121.9220), (37.2920, -121.8780),
        ],
    },
    "Long Beach": {
        "split": "validation",
        "base": (33.8430, -118.1840),
        "residential_offsets": [
            (-0.0014, -0.0014), (-0.0014, 0.0014),
            (0.0014, -0.0014), (0.0014, 0.0014),
        ],
        "controls": [
            (33.8130, -118.0900), (33.7700, -118.1700),
            (33.8300, -118.2050), (33.8580, -118.1800),
        ],
    },
    "Oakland": {
        "split": "test",
        "base": (37.8240, -122.2600),
        "residential_offsets": [
            (-0.0014, -0.0014), (-0.0014, 0.0014),
            (0.0014, -0.0014), (0.0014, 0.0014),
        ],
        "controls": [
            (37.8020, -122.2570), (37.8200, -122.2700),
            (37.8350, -122.2500), (37.8080, -122.2300),
        ],
    },
    "San Diego": {
        "split": "test",
        "base": (32.7470, -117.1500),
        "residential_offsets": [
            (-0.0014, -0.0014), (-0.0014, 0.0014),
            (0.0014, -0.0014), (0.0014, 0.0014),
        ],
        "controls": [
            (32.7350, -117.1500), (32.7700, -117.2000),
            (32.7550, -117.1350), (32.7280, -117.1650),
        ],
    },
}


def _web_mercator(lat, lon):
    radius = 6378137.0
    x = radius * math.radians(lon)
    y = radius * math.log(math.tan(math.pi / 4 + math.radians(lat) / 2))
    return x, y


def _samples():
    result = []
    for city, config in CITY_SAMPLES.items():
        base_lat, base_lon = config["base"]
        prefix = "".join(part[0] for part in city.split()).upper()
        for index, (dlat, dlon) in enumerate(config["residential_offsets"], 1):
            result.append({
                "sample_id": f"{prefix}-R-{index:02d}", "city": city,
                "split": config["split"], "sampling_stratum": "residential_candidate",
                "latitude": base_lat + dlat, "longitude": base_lon + dlon,
            })
        for index, (lat, lon) in enumerate(config["controls"], 1):
            result.append({
                "sample_id": f"{prefix}-C-{index:02d}", "city": city,
                "split": config["split"], "sampling_stratum": "nonroof_control_candidate",
                "latitude": lat, "longitude": lon,
            })
    return result


def _identify(session, x, y):
    params = {
        "geometry": json.dumps({"x": x, "y": y, "spatialReference": {"wkid": 3857}}),
        "geometryType": "esriGeometryPoint", "sr": 3857,
        "returnGeometry": "false", "mosaicRule": json.dumps(MOSAIC_RULE), "f": "json",
    }
    try:
        response = _get_with_retry(session, f"{SERVICE}/identify", params=params, timeout=45)
    except requests.RequestException:
        # Some image-service scenes fail catalog identification under an explicit
        # mosaic rule even though exportImage succeeds. The default identify
        # still returns the selected source item and acquisition year.
        params.pop("mosaicRule", None)
        response = _get_with_retry(session, f"{SERVICE}/identify", params=params, timeout=45)
    response.raise_for_status()
    data = response.json()
    features = data.get("catalogItems", {}).get("features", [])
    attributes = [item.get("attributes", {}) for item in features]
    if not attributes:
        attributes = [data.get("properties", {})]
    attributes.sort(key=lambda item: (item.get("Year") or 0, item.get("Name") or ""), reverse=True)
    best = attributes[0]
    return best.get("Name", ""), best.get("Year", "")


def _download_sample(sample):
    lat, lon = sample["latitude"], sample["longitude"]
    x, y = _web_mercator(lat, lon)
    bbox = f"{x-HALF_EXTENT_M},{y-HALF_EXTENT_M},{x+HALF_EXTENT_M},{y+HALF_EXTENT_M}"
    session = requests.Session()
    session.headers["User-Agent"] = "SolarLeadFinder-California-Roof-Review/1.0"

    response = _get_with_retry(
        session, f"{SERVICE}/exportImage", params={
            "bbox": bbox, "bboxSR": 3857, "imageSR": 3857,
            "size": f"{IMAGE_SIZE},{IMAGE_SIZE}", "format": "jpg", "f": "image",
            "mosaicRule": json.dumps(MOSAIC_RULE),
        }, timeout=90,
    )
    response.raise_for_status()
    image = Image.open(BytesIO(response.content)).convert("RGB")
    if image.size != (IMAGE_SIZE, IMAGE_SIZE):
        raise ValueError(f"Unexpected image dimensions for {sample['sample_id']}: {image.size}")

    image_dir = OUTPUT / "images"
    image_dir.mkdir(parents=True, exist_ok=True)
    relative_path = f"images/{sample['sample_id']}.jpg"
    image.save(OUTPUT / relative_path, quality=95)
    source_item, imagery_year = _identify(session, x, y)
    sample.update({
        "image_file": relative_path,
        "source_item": source_item,
        "imagery_year": imagery_year,
        "source_service": SERVICE,
        "retrieved_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "image_size": f"{IMAGE_SIZE}x{IMAGE_SIZE}",
        "ground_sample_distance_m": PIXEL_SIZE_M,
        "roof_label": "",
        "review_confidence": "",
        "review_notes": "",
    })
    return sample


def _get_with_retry(session, url, params, timeout):
    for attempt in range(4):
        try:
            response = session.get(url, params=params, timeout=timeout)
        except requests.RequestException:
            if attempt == 3:
                raise
            time.sleep(1.5 * (attempt + 1))
            continue
        if response.status_code < 500 and response.status_code != 429:
            response.raise_for_status()
            return response
        if attempt == 3:
            response.raise_for_status()
        time.sleep(1.5 * (attempt + 1))
    raise RuntimeError("USGS request retry limit reached")


def _write_manifest(rows):
    fields = [
        "sample_id", "city", "split", "sampling_stratum", "latitude", "longitude",
        "roof_label", "review_confidence", "review_notes", "image_file", "image_size",
        "ground_sample_distance_m", "imagery_year", "source_item", "source_service",
        "retrieved_utc",
    ]
    with (OUTPUT / "manifest.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in fields})


def _write_contact_sheet(rows):
    thumb_w, thumb_h, label_h, columns = 160, 160, 30, 6
    rows_count = math.ceil(len(rows) / columns)
    sheet = Image.new("RGB", (columns * thumb_w, rows_count * (thumb_h + label_h)), "white")
    draw = ImageDraw.Draw(sheet)
    for index, row in enumerate(rows):
        image = Image.open(OUTPUT / row["image_file"])
        image.thumbnail((thumb_w, thumb_h))
        x, y = (index % columns) * thumb_w, (index // columns) * (thumb_h + label_h)
        sheet.paste(image, (x, y))
        draw.text((x + 4, y + thumb_h + 3), row["sample_id"], fill="black")
    sheet.save(OUTPUT / "contact_sheet.jpg", quality=92)


def main():
    OUTPUT.mkdir(parents=True, exist_ok=True)
    samples = _samples()
    failures_path = OUTPUT / "download_failures.csv"
    previous_rows = []
    if ("--refresh" not in sys.argv and failures_path.exists()
            and (OUTPUT / "manifest.csv").exists()):
        with failures_path.open(newline="", encoding="utf-8") as handle:
            failed_ids = {row["sample_id"] for row in csv.DictReader(handle)}
        with (OUTPUT / "manifest.csv").open(newline="", encoding="utf-8") as handle:
            previous_rows = list(csv.DictReader(handle))
        if failed_ids:
            samples = [sample for sample in samples if sample["sample_id"] in failed_ids]
            print(f"Retrying {len(samples)} failed samples; existing chips are retained.")
        else:
            samples = []
    elif "--refresh" not in sys.argv and (OUTPUT / "manifest.csv").exists():
        print("Manifest already exists; pass --refresh to download the full batch again.")
        return
    rows, failures = [], []
    with ThreadPoolExecutor(max_workers=3) as pool:
        future_map = {pool.submit(_download_sample, sample): sample for sample in samples}
        for future in as_completed(future_map):
            sample = future_map[future]
            try:
                row = future.result()
                rows.append(row)
                print(f"Downloaded {row['sample_id']} ({row['imagery_year']})", flush=True)
            except Exception as error:
                failures.append((sample["sample_id"], str(error)))
                print(f"Failed {sample['sample_id']}: {error}", flush=True)
    combined = {row["sample_id"]: row for row in previous_rows}
    combined.update({row["sample_id"]: row for row in rows})
    rows = sorted(combined.values(), key=lambda row: row["sample_id"])
    _write_manifest(rows)
    if rows:
        _write_contact_sheet(rows)
    readme = OUTPUT / "README.md"
    readme.write_text(
        "# California roof review v2\n\n"
        "This is a manually reviewed roof-presence set from USGS NAIP aerial imagery. "
        "It is geographically separate from the earlier Fresno, Modesto, Oxnard, "
        "and Stockton pilot. It does not label solar status. The sampling stratum "
        "is not ground truth; review each image at its center and record `roof`, "
        "`nonroof`, or `uncertain` in `manifest.csv`. Do not force a label where "
        "the image is ambiguous. City splits are assigned before review: Riverside "
        "and Bakersfield train; San Jose and Long Beach validation; Oakland and "
        "San Diego test. Keep the test labels sealed until detector decisions are frozen.\n\n"
        "Imagery source: USGS National Map NAIP ImageServer, natural-color aerial "
        "orthophotos (public domain); the manifest records service item and year. "
        "No Google Maps imagery is used.\n\n"
        f"Generated {len(rows)} of {len(samples)} samples.\n",
        encoding="utf-8",
    )
    if failures:
        failures_path.write_text(
            "sample_id,error\n" + "".join(f'"{sid}","{err.replace(chr(34), chr(39))}"\n' for sid, err in failures),
            encoding="utf-8",
        )
    elif failures_path.exists():
        failures_path.unlink()
    print(f"Wrote {len(rows)} samples to {OUTPUT}")
    if failures:
        print(f"{len(failures)} downloads failed; see {OUTPUT / 'download_failures.csv'}")


if __name__ == "__main__":
    main()
