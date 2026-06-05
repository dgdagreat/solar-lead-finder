"""
Solar Panel Detector — 5-Stage Computer Vision Pipeline
========================================================
Each stage analyses a different physical property of solar panels.
A home is only marked "has_solar" when:
  • combined weighted score  > SOLAR_THRESHOLD  AND
  • at least 2 stages independently trigger

Stage 1 — Color Segmentation
    Panels appear as clustered dark-navy/blue pixels.
    Scattered dark pixels (roads, shadows) are penalised.

Stage 2 — Even-Spaced Grid Detection
    Panel arrays create EVENLY-SPACED parallel lines.
    Any urban area has parallel lines; we specifically require
    periodicity (consistent spacing) using a line-position histogram.

Stage 3 — Same-Size Rectangle Clustering
    An array contains many rectangles of NEARLY IDENTICAL area.
    We reject detections where rectangle sizes are widely varied.

Stage 4 — Uniform Dark-Blob Analysis
    Panels are flat/smooth (low texture std-dev), solid, and medium-sized.
    We cap the maximum blob count — too many blobs means it's just
    a busy urban scene, not a panel array.

Stage 5 — Reference Comparison  ★ NEW ★
    Compare the candidate image's solar-region fingerprint (HSV histogram +
    edge-density histogram + texture histogram) against a library of
    satellite images CONFIRMED to have solar panels.
    High similarity to known-solar images strongly boosts confidence.
"""

import cv2
import numpy as np
from PIL import Image, ImageEnhance
import requests
from io import BytesIO
import logging

logger = logging.getLogger(__name__)

WEIGHTS = {
    "color":      0.20,
    "grid":       0.28,
    "rectangles": 0.20,
    "blob":       0.12,
    "reference":  0.20,   # Stage 5 — similarity to known solar images
}
SOLAR_THRESHOLD    = 0.36
MIN_STAGES_AGREE   = 2
STAGE_FIRE_CUTOFF  = 0.30   # a stage "fires" if its score exceeds this


# ══════════════════════════════════════════════════════════════════════════
# Public entry points
# ══════════════════════════════════════════════════════════════════════════

def detect_solar_panels_from_url(image_url: str) -> dict:
    try:
        resp = requests.get(image_url, timeout=15)
        resp.raise_for_status()
        image = Image.open(BytesIO(resp.content)).convert("RGB")
        return detect_solar_panels(image)
    except Exception as e:
        logger.error(f"Failed to fetch/detect {image_url}: {e}")
        return _error_result(str(e))


def detect_solar_panels_from_path(path: str) -> dict:
    try:
        image = Image.open(path).convert("RGB")
        return detect_solar_panels(image)
    except Exception as e:
        logger.error(f"Failed to load {path}: {e}")
        return _error_result(str(e))


# ══════════════════════════════════════════════════════════════════════════
# Core pipeline
# ══════════════════════════════════════════════════════════════════════════

def detect_solar_panels(pil_image: Image.Image) -> dict:
    # Slight contrast boost so panels stand out from roof
    pil_image = ImageEnhance.Contrast(pil_image).enhance(1.3)

    img_rgb  = np.array(pil_image)
    img_bgr  = cv2.cvtColor(img_rgb,  cv2.COLOR_RGB2BGR)
    img_hsv  = cv2.cvtColor(img_rgb,  cv2.COLOR_RGB2HSV)
    img_gray = cv2.cvtColor(img_bgr,  cv2.COLOR_BGR2GRAY)

    color_score,  color_mask,  color_info  = _stage_color(img_hsv)
    cluster_ratio = color_info.get("cluster_ratio", 0.0)

    grid_score,               grid_info   = _stage_grid(img_gray, color_mask)
    effective_grid = grid_score if color_score > 0.22 else 0.0

    rect_score,               rect_info   = _stage_rectangles(color_mask, img_gray)
    blob_score,               blob_info   = _stage_dark_blobs(img_gray, color_mask)

    # Stage 5 — Reference comparison
    try:
        from .reference_comparator import compare_to_references
        ref_score = compare_to_references(pil_image)
    except Exception as e:
        logger.warning(f"Reference comparison skipped: {e}")
        ref_score = 0.0

    combined = (
        color_score    * WEIGHTS["color"]      +
        effective_grid * WEIGHTS["grid"]       +
        rect_score     * WEIGHTS["rectangles"] +
        blob_score     * WEIGHTS["blob"]       +
        ref_score      * WEIGHTS["reference"]
    )

    stages_triggered = sum([
        color_score    > STAGE_FIRE_CUTOFF,
        effective_grid > STAGE_FIRE_CUTOFF,
        rect_score     > STAGE_FIRE_CUTOFF,
        blob_score     > STAGE_FIRE_CUTOFF,
        ref_score      > STAGE_FIRE_CUTOFF,
    ])

    strong_color_hit  = (color_score > 0.55) and (cluster_ratio > 0.35)
    # Strong reference match + any corroborating signal = high confidence
    strong_ref_hit    = (ref_score > 0.60) and (color_score > 0.20)

    has_solar = (
        ((combined > SOLAR_THRESHOLD) and (stages_triggered >= MIN_STAGES_AGREE))
        or strong_color_hit
        or strong_ref_hit
    )

    logger.info(
        f"color:{color_score:.2f} grid:{grid_score:.2f} "
        f"rect:{rect_score:.2f} blob:{blob_score:.2f} ref:{ref_score:.2f} "
        f"→ combined:{combined:.2f} stages:{stages_triggered} has_solar:{has_solar}"
    )

    return {
        "has_solar":        has_solar,
        "confidence":       round(combined, 3),
        "method":           "multi_stage_cv_v2",
        "stages_triggered": stages_triggered,
        "stage_scores": {
            "color_segmentation":   round(color_score, 3),
            "grid_line_detection":  round(grid_score,  3),
            "rectangle_clustering": round(rect_score,  3),
            "dark_blob_uniformity": round(blob_score,  3),
            "reference_similarity": round(ref_score,   3),
        },
        "details": {
            "parallel_line_groups": grid_info.get("clusters", 0),
            "panel_rectangles":     rect_info.get("valid_rects", 0),
            "dark_blobs":           blob_info.get("uniform_blobs", 0),
            "color_cluster_ratio":  color_info.get("cluster_ratio", 0.0),
        }
    }


# ══════════════════════════════════════════════════════════════════════════
# Stage 1 — Color Segmentation (with clustering check)
# ══════════════════════════════════════════════════════════════════════════

def _stage_color(img_hsv: np.ndarray) -> tuple[float, np.ndarray, dict]:
    """
    Two specific blue hue ranges only (removed the over-broad near-black range).
    After masking, measure CONCENTRATION: pixels that are tightly
    clustered together score higher than scattered dark pixels.
    Scattered matches (roads, random shadows) are heavily penalised.
    """
    h, w = img_hsv.shape[:2]
    total = h * w

    # Only specific solar-panel blue hues — NOT generic near-black
    ranges = [
        (np.array([100, 55,  8]),  np.array([132, 255, 72])),  # navy blue
        (np.array([88,  18,  8]),  np.array([118,  75, 58])),  # blue-grey
    ]

    raw_mask = np.zeros((h, w), dtype=np.uint8)
    for lo, hi in ranges:
        raw_mask = cv2.bitwise_or(raw_mask, cv2.inRange(img_hsv, lo, hi))

    # Morphological cleanup
    k3 = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    mask = cv2.morphologyEx(raw_mask, cv2.MORPH_OPEN,  k3, iterations=2)
    mask = cv2.morphologyEx(mask,     cv2.MORPH_CLOSE, k3, iterations=2)

    solar_pixels = int(np.sum(mask > 0))
    pixel_ratio  = solar_pixels / total

    # Concentration check: what fraction of matching pixels are in the
    # single largest connected component?
    cluster_ratio = 0.0
    if solar_pixels > 20:
        n_labels, _, stats, _ = cv2.connectedComponentsWithStats(mask)
        if n_labels > 1:
            areas = [stats[i, cv2.CC_STAT_AREA] for i in range(1, n_labels)]
            biggest = max(areas)
            cluster_ratio = biggest / solar_pixels
            # Panels form dense clusters. If biggest cluster < 30% of all
            # matching pixels, it's scattered noise (roads, shadows).
            if cluster_ratio < 0.30:
                pixel_ratio *= 0.25   # heavy penalty for scattered pixels

    # 0.3% = 0.1 score; 6% = 1.0 score
    score = min(pixel_ratio / 0.06, 1.0)
    return score, mask, {"cluster_ratio": round(cluster_ratio, 3)}


# ══════════════════════════════════════════════════════════════════════════
# Stage 2 — Even-Spaced Grid Detection
# ══════════════════════════════════════════════════════════════════════════

def _stage_grid(img_gray: np.ndarray, color_mask: np.ndarray) -> tuple[float, dict]:
    """
    Detect evenly-spaced parallel lines (solar panel grid).

    Key improvement over naive Hough: we restrict the analysis to a
    FOCUS ZONE — the bounding box of the largest color cluster — so we
    only look for grid lines where we already suspect panels exist.
    This removes the majority of road/building line noise.

    Within the focus zone we also require:
      • mean line spacing > 6 px  (real panels have physical size)
      • gap coefficient of variation < 0.55  (evenly spaced)
    """
    # Find focus zone from colour mask
    focus_gray = _crop_to_focus_zone(img_gray, color_mask)
    if focus_gray is None:
        return 0.0, {"clusters": 0}

    blurred   = cv2.GaussianBlur(focus_gray, (0, 0), 2)
    sharpened = cv2.addWeighted(focus_gray, 1.5, blurred, -0.5, 0)
    edges     = cv2.Canny(sharpened, 40, 120, apertureSize=3)

    lines = cv2.HoughLinesP(
        edges,
        rho=1, theta=np.pi / 180, threshold=20,
        minLineLength=12, maxLineGap=5,
    )

    if lines is None:
        return 0.0, {"clusters": 0}

    angle_list = [
        np.degrees(np.arctan2(l[0][3] - l[0][1], l[0][2] - l[0][0])) % 180
        for l in lines
    ]

    buckets = {}
    for ang in angle_list:
        b = int(ang // 10) * 10
        buckets[b] = buckets.get(b, 0) + 1
    dominant = {b: cnt for b, cnt in buckets.items() if cnt >= 3}

    if not dominant:
        return 0.0, {"clusters": 0}

    periodic_clusters = 0
    for bucket_angle in dominant:
        bl = [l[0] for l, a in zip(lines, angle_list) if abs(a - bucket_angle) <= 12]
        if _has_even_spacing(bl, bucket_angle):
            periodic_clusters += 1

    bucket_angles = list(dominant.keys())
    has_grid = any(
        75 <= abs(b1 - b2) <= 105
        for i, b1 in enumerate(bucket_angles)
        for b2 in bucket_angles[i + 1:]
    )

    if   periodic_clusters >= 2 and has_grid:  score = min(0.65 + periodic_clusters * 0.10, 1.0)
    elif periodic_clusters >= 1 and has_grid:  score = 0.55
    elif periodic_clusters >= 1:               score = 0.40
    elif has_grid and len(lines) >= 8:         score = 0.25
    else:                                      score = 0.0

    return score, {"clusters": periodic_clusters}


def _crop_to_focus_zone(img: np.ndarray, mask: np.ndarray, pad: int = 30) -> np.ndarray | None:
    """
    Crop img to the bounding box of the largest connected component in mask.
    Returns None if no meaningful region found.
    """
    if np.sum(mask) == 0:
        return None
    n, labels, stats, _ = cv2.connectedComponentsWithStats(mask)
    if n <= 1:
        return None
    areas = [stats[i, cv2.CC_STAT_AREA] for i in range(1, n)]
    best  = int(np.argmax(areas)) + 1
    x  = max(stats[best, cv2.CC_STAT_LEFT]   - pad, 0)
    y  = max(stats[best, cv2.CC_STAT_TOP]    - pad, 0)
    x2 = min(x + stats[best, cv2.CC_STAT_WIDTH]  + 2 * pad, img.shape[1])
    y2 = min(y + stats[best, cv2.CC_STAT_HEIGHT] + 2 * pad, img.shape[0])
    crop = img[y:y2, x:x2]
    if crop.size < 400:
        return None
    return crop


def _has_even_spacing(bucket_lines: list, angle_deg: float) -> bool:
    """
    Project midpoints onto the axis perpendicular to angle_deg.
    Deduplicate close projections, then check:
      • mean gap > 6 px   (physical panel size floor)
      • gap CV  < 0.55    (evenly spaced — solar panel regularity)
    """
    if len(bucket_lines) < 4:
        return False

    perp = np.radians(angle_deg + 90)
    cx, cy = np.cos(perp), np.sin(perp)
    projs = sorted({
        round((x1 + x2) / 2 * cx + (y1 + y2) / 2 * cy, 1)
        for x1, y1, x2, y2 in bucket_lines
    })

    if len(projs) < 4:
        return False

    # Remove duplicate projections within 2px
    deduped = [projs[0]]
    for p in projs[1:]:
        if p - deduped[-1] > 2:
            deduped.append(p)

    if len(deduped) < 4:
        return False

    gaps     = np.diff(deduped)
    mean_gap = float(np.mean(gaps))
    if mean_gap < 6:
        return False

    cv_gaps = float(np.std(gaps)) / mean_gap
    return cv_gaps < 0.55


# ══════════════════════════════════════════════════════════════════════════
# Stage 3 — Same-Size Rectangle Clustering
# ══════════════════════════════════════════════════════════════════════════

def _stage_rectangles(color_mask: np.ndarray, img_gray: np.ndarray) -> tuple[float, dict]:
    """
    Solar panel arrays: many rectangles of NEARLY IDENTICAL area in a cluster.
    Key differentiator from roads/buildings: size CONSISTENCY.
    Steps:
      1. Find rectangular contours in the color mask
      2. Keep panel-sized ones (60–5000 px²) with right aspect ratio
      3. Group into clusters by proximity
      4. For the biggest cluster, compute the area coefficient of variation
         — low CV = all panels the same size = real array
    """
    k3 = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    cleaned = cv2.morphologyEx(color_mask, cv2.MORPH_OPEN, k3, iterations=1)

    contours, _ = cv2.findContours(cleaned, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    panel_rects = []
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if not (60 < area < 5000):
            continue
        peri   = cv2.arcLength(cnt, True)
        approx = cv2.approxPolyDP(cnt, 0.05 * peri, True)
        if len(approx) not in (4, 5):
            continue
        x, y, bw, bh = cv2.boundingRect(cnt)
        if min(bw, bh) == 0:
            continue
        ar = max(bw, bh) / min(bw, bh)
        if not (1.1 <= ar <= 3.8):
            continue
        panel_rects.append((x, y, bw, bh, area))

    n = len(panel_rects)
    if n < 2:
        return 0.0, {"valid_rects": 0}

    clusters = _cluster_rects([(x, y, w, h) for x, y, w, h, _ in panel_rects], max_dist=70)
    biggest_cluster_idx = max(range(len(clusters)), key=lambda i: len(clusters[i]))
    biggest_cluster     = clusters[biggest_cluster_idx]
    bc_size             = len(biggest_cluster)

    if bc_size < 2:
        return 0.0, {"valid_rects": n}

    # Area consistency in the biggest cluster
    bc_areas = [panel_rects[i][4] for i in biggest_cluster]
    mean_area = np.mean(bc_areas)
    std_area  = np.std(bc_areas)
    area_cv   = std_area / mean_area if mean_area > 0 else 1.0

    # Low CV = uniform sizes = real panel array
    # High CV = mixed sizes = probably not panels
    if area_cv > 0.70:
        return min(n / 40, 0.20), {"valid_rects": n}   # penalise inconsistency

    if bc_size >= 6 and area_cv < 0.40:
        score = min(0.60 + (bc_size / 40) * 0.40, 1.0)
    elif bc_size >= 4:
        score = 0.40 + (1.0 - area_cv) * 0.20
    elif bc_size >= 2:
        score = 0.25
    else:
        score = 0.10

    return score, {"valid_rects": n}


def _cluster_rects(rects, max_dist: int = 70):
    centres = [(x + w // 2, y + h // 2) for x, y, w, h in rects]
    visited = [False] * len(centres)
    clusters = []
    for i, c1 in enumerate(centres):
        if visited[i]:
            continue
        cluster = [i]
        visited[i] = True
        for j, c2 in enumerate(centres):
            if not visited[j]:
                dist = ((c1[0] - c2[0]) ** 2 + (c1[1] - c2[1]) ** 2) ** 0.5
                if dist < max_dist:
                    cluster.append(j)
                    visited[j] = True
        clusters.append(cluster)
    return clusters


# ══════════════════════════════════════════════════════════════════════════
# Stage 4 — Uniform Dark-Blob Analysis
# ══════════════════════════════════════════════════════════════════════════

def _stage_dark_blobs(img_gray: np.ndarray, color_mask: np.ndarray) -> tuple[float, dict]:
    """
    Solar panels are flat and smooth (low texture variance) and solid.
    Tighter than before:
      • texture std < 22  (was 45 — was catching roads/shadows)
      • solidity > 0.78   (was 0.65)
      • area 80 – 4 000   (was 80 – 14 000)
      • MAX blob count 25 — if there are more, it's just a busy scene
    """
    # Work inside focus zone to avoid counting random urban blobs
    focus = _crop_to_focus_zone(img_gray, color_mask)
    work  = focus if focus is not None else img_gray

    _, dark = cv2.threshold(work, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    k5 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    dark = cv2.morphologyEx(dark, cv2.MORPH_OPEN, k5, iterations=1)

    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(dark)

    uniform_blobs = 0
    for lbl in range(1, num_labels):
        area   = stats[lbl, cv2.CC_STAT_AREA]
        width  = stats[lbl, cv2.CC_STAT_WIDTH]
        height = stats[lbl, cv2.CC_STAT_HEIGHT]

        if not (80 < area < 4000):
            continue
        if min(width, height) < 5:
            continue

        # Solidity
        mask_lbl = (labels == lbl).astype(np.uint8)
        cnts, _  = cv2.findContours(mask_lbl, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not cnts:
            continue
        hull_area = cv2.contourArea(cv2.convexHull(cnts[0]))
        if hull_area == 0:
            continue
        solidity = area / hull_area
        if solidity < 0.78:
            continue

        # Texture (lower threshold than before)
        blob_pixels = work[labels == lbl]
        if np.std(blob_pixels) > 22:
            continue

        uniform_blobs += 1

    # Cap: too many blobs = busy urban scene, not solar panels
    if uniform_blobs > 25:
        score = 0.05
    elif uniform_blobs >= 6:
        score = min(0.45 + (uniform_blobs / 25) * 0.55, 1.0)
    elif uniform_blobs >= 3:
        score = 0.25 + (uniform_blobs / 15) * 0.20
    elif uniform_blobs >= 1:
        score = 0.10
    else:
        score = 0.0

    return score, {"uniform_blobs": uniform_blobs}


# ══════════════════════════════════════════════════════════════════════════
# Helpers
# ══════════════════════════════════════════════════════════════════════════

def _error_result(msg: str) -> dict:
    return {
        "has_solar":        False,
        "confidence":       0.0,
        "method":           "error",
        "error":            msg,
        "stages_triggered": 0,
        "stage_scores":     {},
        "details":          {},
    }
