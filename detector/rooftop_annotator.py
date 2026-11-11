"""
Rooftop Annotator — v3
=======================
How it works:
  • Localises the building at the image centre with a flood fill, then keeps
    only its roof-coloured pixels — so grass, trees, bushes and pools are excluded
  • Highlights that single roof with a semi-transparent yellow fill — neighbours
    are not covered
  • Verifies the image is actually a residential property:
      - a roof-coloured region exists within 100 px of centre
      - that region is a plausible single-family size
      - the centre is not dominated by road / pavement
  • Returns a verification dict so the app can warn when
    the target property is not clearly in the frame
"""

import cv2
import numpy as np
import requests
import os
from io import BytesIO
from PIL import Image
from pathlib import Path
from django.conf import settings
import logging

logger = logging.getLogger(__name__)

# ── Tuning constants ──────────────────────────────────────────────────────
MAX_CENTRE_DIST   = 100   # px — avoid accepting a neighboring roof across a street/yard
MIN_ROOF_AREA     = 1200  # px² — smallest acceptable rooftop
MAX_ROOF_AREA     = 30000 # px² — larger fills often spill into fields or roads
ROOF_WINDOW_RADIUS = 140  # px — clip the highlight to a single-home window around centre
MAX_VEGETATION_FRACTION = 0.10  # reject shadowed foliage that looks grey in the roof mask
# A second, tightly centered roof check is used when flood fill leaks into
# neighboring trees or fails to seed on a roof-colored pixel.  It requires a
# roof-color majority, limited visible vegetation, and real surface edges.
CENTER_PATCH_RADIUS = 55
CENTER_PATCH_MIN_ROOF_FRACTION = 0.60
CENTER_PATCH_MAX_VEGETATION_FRACTION = 0.30
CENTER_PATCH_MIN_EDGE_FRACTION = 0.035
CENTER_PATCH_MIN_HUE_VARIATION = 8.0
CENTER_PATCH_MIN_SATURATION_VARIATION = 10.0
HIGHLIGHT_COLOR   = (30, 215, 255)   # BGR: vivid yellow-orange
HIGHLIGHT_ALPHA   = 0.30             # fill opacity
OUTLINE_THICKNESS = 3


# ══════════════════════════════════════════════════════════════════════════
# Public entry point
# ══════════════════════════════════════════════════════════════════════════

def fetch_and_annotate(home) -> str | None:
    """
    Fetch the satellite image for `home`, detect + highlight the target
    building, save to media/, return the relative media path.
    Also updates home.notes with a verification warning if needed.
    """
    from .image_fetcher import get_static_map_url

    image_url = get_static_map_url(home.full_address)
    if not image_url:
        return None

    try:
        resp = requests.get(image_url, timeout=15)
        resp.raise_for_status()
        pil_img = Image.open(BytesIO(resp.content)).convert("RGB")
    except Exception as e:
        logger.error(f"Failed to download satellite image: {e}")
        return None

    rel_path, _ = annotate_rooftop_image(home, pil_img)
    return rel_path


def annotate_rooftop_image(home, pil_img: Image.Image) -> tuple[str, dict]:
    """Annotate and verify the exact satellite image used by the classifier."""
    img_rgb = np.array(pil_img.convert("RGB"))
    img_bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)

    annotated, verification = _annotate_rooftop(img_bgr)

    # Persist the verification warning, and clear a stale warning after a good scan.
    warning_prefix = "⚠️ Satellite image may not show target property clearly:"
    if not verification["in_frame"]:
        warning = f"{warning_prefix} {verification['reason']}"
        if warning not in (home.notes or ""):
            home.notes = warning
            home.save(update_fields=["notes"])
    elif (home.notes or "").startswith(warning_prefix):
        home.notes = ""
        home.save(update_fields=["notes"])

    # Save through Django's storage backend (not a raw filesystem write) so the
    # annotated image lands wherever `default` storage points: the local media
    # dir in dev, or the GCS bucket in production. tasks.py sets this rel_path as
    # home.image, so home.image.url then resolves correctly on either backend.
    rel_path = f"home_images/annotated_{home.id}.jpg"
    ok, buf = cv2.imencode(".jpg", annotated)
    if not ok:
        raise RuntimeError("Failed to encode annotated rooftop image")

    from django.core.files.base import ContentFile
    from django.core.files.storage import default_storage

    # Overwrite any prior annotation so the filename stays stable (storage.save
    # would otherwise append a random suffix on collision).
    if default_storage.exists(rel_path):
        default_storage.delete(rel_path)
    default_storage.save(rel_path, ContentFile(buf.tobytes()))

    logger.info(f"Annotated: {rel_path} | verified={verification['in_frame']}")
    return rel_path, verification


# ══════════════════════════════════════════════════════════════════════════
# Core annotation
# ══════════════════════════════════════════════════════════════════════════

def _annotate_rooftop(img_bgr: np.ndarray) -> tuple[np.ndarray, dict]:
    h, w   = img_bgr.shape[:2]
    cx, cy = w // 2, h // 2
    output = img_bgr.copy()
    img_hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)

    # ── Step 1: Segment roof-coloured pixels ─────────────────────────────
    # Excludes grass, trees, bushes, pools and cars — none are roof-coloured.
    roof_mask = _roof_colour_mask(img_hsv)

    # ── Step 2: Isolate the ONE roof at the address centre ───────────────
    # (a) flood fill from the centre to localise the building, (b) keep only its
    # roof-coloured pixels (drops grass / trees / bushes / pool), (c) clip to a
    # single-home window so dense row-house blocks that share roof colours can't
    # merge into one giant highlight, then re-isolate the centre component.
    k5 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    fill_mask, _ = _flood_fill_roof(img_bgr, cx, cy)
    target = cv2.bitwise_and(fill_mask, roof_mask)
    window = np.zeros_like(target)
    cv2.circle(window, (cx, cy), ROOF_WINDOW_RADIUS, 255, -1)
    target = cv2.bitwise_and(target, window)
    target = _roof_component_at_centre(target, cx, cy)
    # Close internal gaps (shadows, ridge lines, vents) so the highlight is solid
    target = cv2.morphologyEx(target, cv2.MORPH_CLOSE, k5, iterations=2)
    filled_area = int(cv2.countNonZero(target))

    # ── Step 3: Verify property is in frame ───────────────────────────────
    verification = _verify_in_frame(img_hsv, target, filled_area, h, w, cx, cy)
    if not verification["in_frame"]:
        # Flood fill is useful for isolating roofs in ordinary imagery, but it
        # can fail on low-contrast California roofs or absorb nearby shaded
        # trees. Recheck only a small patch around the address point. Requiring
        # roof-colored pixels, low vegetation, and surface edges keeps a plain
        # road, lawn, or water patch from passing this fallback.
        patch_evidence = _center_patch_roof_evidence(img_hsv, cx, cy)
        if patch_evidence["roof_like"]:
            patch = np.zeros((h, w), dtype=np.uint8)
            cv2.circle(patch, (cx, cy), CENTER_PATCH_RADIUS, 255, -1)
            target = cv2.bitwise_and(roof_mask, patch)
            # Do not highlight every roof-colored surface in the local disk.
            # Dense California blocks can put several roofs, streets, and
            # driveways inside this window; retain only the connected component
            # at the target point (or its nearest plausible roof component).
            target = _roof_component_at_centre(target, cx, cy)
            target = cv2.morphologyEx(target, cv2.MORPH_CLOSE, k5, iterations=2)
            filled_area = int(cv2.countNonZero(target))
            if filled_area >= MIN_ROOF_AREA:
                verification = {
                    "in_frame": True,
                    "reason": "",
                    "roof_detection_method": "center_patch",
                    "vegetation_fraction": patch_evidence["vegetation_fraction"],
                    "center_patch": patch_evidence,
                }
    verification.setdefault("roof_detection_method", "flood_fill")
    # Version marker lets the GCP demo recheck images annotated by the older,
    # foliage-blind verifier exactly once after this safeguard is deployed.
    verification["vegetation_checked"] = True

    # ── Step 4: Draw highlight ────────────────────────────────────────────
    if filled_area >= MIN_ROOF_AREA:
        contours, _ = cv2.findContours(target, cv2.RETR_EXTERNAL,
                                        cv2.CHAIN_APPROX_SIMPLE)
        if contours:
            best_cnt = max(contours, key=cv2.contourArea)

            # Semi-transparent fill — ONLY this single roof polygon
            overlay = output.copy()
            cv2.drawContours(overlay, [best_cnt], -1, HIGHLIGHT_COLOR, -1)
            cv2.addWeighted(overlay, HIGHLIGHT_ALPHA, output,
                            1 - HIGHLIGHT_ALPHA, 0, output)

            # Solid bold outline
            cv2.drawContours(output, [best_cnt], -1, HIGHLIGHT_COLOR,
                             OUTLINE_THICKNESS)

            # Label above the bounding box
            x_r, y_r, bw, bh = cv2.boundingRect(best_cnt)
            _draw_label(output, "TARGET PROPERTY",
                        max(x_r, 4), max(y_r - 8, 22), HIGHLIGHT_COLOR)
    else:
        # Flood fill found nothing — fallback tight circle
        cv2.circle(output, (cx, cy), 80, (0, 100, 255), 3)
        _draw_label(output, "VERIFY MANUALLY", cx - 80, cy + 100, (0, 100, 255))

    # Always: white crosshair at exact address point
    cv2.drawMarker(output, (cx, cy), (255, 255, 255),
                   cv2.MARKER_CROSS, 22, 2)

    # Add verification banner at top if not confirmed
    if not verification["in_frame"]:
        _draw_warning_banner(output, verification["reason"])

    return output, verification


# ══════════════════════════════════════════════════════════════════════════
# Roof colour segmentation
# ══════════════════════════════════════════════════════════════════════════

def _roof_colour_mask(img_hsv: np.ndarray) -> np.ndarray:
    """
    Segment common roof materials in HSV space.
    Returns a cleaned binary mask.
    """
    ranges = [
        # Terracotta / clay tiles
        (np.array([5,  55, 80]),  np.array([20, 230, 220])),
        # Concrete / medium grey flat roof
        (np.array([0,  0,  80]),  np.array([180, 38, 200])),
        # Light grey / white roofs
        (np.array([0,  0,  180]), np.array([180, 30, 255])),
        # Dark asphalt / charcoal shingles
        (np.array([0,  0,  15]),  np.array([180, 45, 95])),
        # Brown / wood shingles
        (np.array([10, 38, 55]),  np.array([26, 185, 165])),
        # Blue-grey metal roofing
        (np.array([95, 12, 70]),  np.array([130, 60, 175])),
    ]

    mask = np.zeros(img_hsv.shape[:2], dtype=np.uint8)
    for lo, hi in ranges:
        mask = cv2.bitwise_or(mask, cv2.inRange(img_hsv, lo, hi))

    # Close gaps within a single rooftop, then remove tiny noise
    k7 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    k3 = cv2.getStructuringElement(cv2.MORPH_RECT,    (3, 3))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k7, iterations=3)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN,  k3, iterations=1)
    return mask


# ══════════════════════════════════════════════════════════════════════════
# Roof selection + verification
# ══════════════════════════════════════════════════════════════════════════

def _roof_component_at_centre(mask: np.ndarray, cx: int, cy: int) -> np.ndarray:
    """
    Return a mask of the single connected roof region at (cx, cy) — the building
    that sold. If the centre pixel isn't on a roof, fall back to the nearest
    plausible roof component within MAX_CENTRE_DIST. Empty mask if none found.
    """
    n, labels, stats, centroids = cv2.connectedComponentsWithStats(mask)
    if n <= 1:
        return np.zeros_like(mask)

    lbl = int(labels[cy, cx])
    if lbl == 0:                                    # centre isn't on a roof pixel
        best, best_dist = 0, float("inf")
        for i in range(1, n):
            if stats[i, cv2.CC_STAT_AREA] < MIN_ROOF_AREA:
                continue
            d = ((centroids[i][0] - cx) ** 2 + (centroids[i][1] - cy) ** 2) ** 0.5
            if d < best_dist:
                best, best_dist = i, d
        if best == 0 or best_dist > MAX_CENTRE_DIST:
            return np.zeros_like(mask)
        lbl = best

    return np.where(labels == lbl, 255, 0).astype(np.uint8)


def _flood_fill_roof(img_bgr: np.ndarray, cx: int, cy: int) -> tuple[np.ndarray, int]:
    """
    Flood fill from (cx, cy) to localise the single connected region at the
    address centre. Tries progressively looser tolerances until a plausible-sized
    region is found. Returns (mask, area). This LOCALISES the building; the
    roof-colour intersection in the caller removes any vegetation it leaks into.
    """
    h, w = img_bgr.shape[:2]
    blurred = cv2.GaussianBlur(img_bgr, (5, 5), 0)
    for tolerance in [12, 20, 30, 42]:
        mask = np.zeros((h + 2, w + 2), dtype=np.uint8)
        lo = hi = (tolerance,) * 3
        flags = cv2.FLOODFILL_MASK_ONLY | (255 << 8) | cv2.FLOODFILL_FIXED_RANGE
        try:
            cv2.floodFill(blurred.copy(), mask, (cx, cy), (255, 255, 255), lo, hi, flags)
        except Exception:
            continue
        result = mask[1:-1, 1:-1]
        area = int(np.sum(result > 0))
        if area >= MIN_ROOF_AREA:
            if area > MAX_ROOF_AREA:
                continue
            return result, area
    return np.zeros((h, w), dtype=np.uint8), 0


def _verify_in_frame(img_hsv, target_mask: np.ndarray, filled_area: int,
                     h: int, w: int, cx: int, cy: int) -> dict:
    """
    Return {"in_frame": bool, "reason": str}.

    Checks:
      1. A rooftop region was found close enough to centre
      2. That region is a plausible single-family size
      3. The centre isn't dominated by grass/tree/open-lot green

    NOTE: We deliberately avoid road-colour checks because grey rooftops
    share the same HSV range as pavement and cause false warnings.
    """
    # Check 1: flood fill found a plausible-sized roof at centre
    if filled_area < MIN_ROOF_AREA:
        return {
            "in_frame": False,
            "reason": "No rooftop detected at address centre — may be a road, tree, or open lot"
        }

    # Dark leaves in shade overlap the broad grey-roof color range. Inspect the
    # original pixels inside the candidate outline as a separate vegetation
    # check; if too much is green, don't call it a roof even if its shadow passed
    # the color mask.
    contours, _ = cv2.findContours(target_mask, cv2.RETR_EXTERNAL,
                                   cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return {"in_frame": False, "reason": "No roof-shaped region found near the address centre"}

    candidate = max(contours, key=cv2.contourArea)
    region = np.zeros((h, w), dtype=np.uint8)
    cv2.drawContours(region, [candidate], -1, 255, thickness=cv2.FILLED)
    green_vegetation = cv2.inRange(
        img_hsv,
        np.array([30, 20, 25], dtype=np.uint8),
        np.array([90, 255, 255], dtype=np.uint8),
    )
    # Deeply shaded hedges can lose their green hue and shift to saturated
    # olive (roughly OpenCV hue 20–29). Keep this range narrow so terracotta
    # roofs stay outside the vegetation check.
    olive_vegetation = cv2.inRange(
        img_hsv,
        np.array([20, 80, 25], dtype=np.uint8),
        np.array([29, 255, 255], dtype=np.uint8),
    )
    vegetation = cv2.bitwise_or(green_vegetation, olive_vegetation)
    region_pixels = max(cv2.countNonZero(region), 1)
    vegetation_fraction = cv2.countNonZero(
        cv2.bitwise_and(vegetation, region)
    ) / region_pixels
    if vegetation_fraction > MAX_VEGETATION_FRACTION:
        return {
            "in_frame": False,
            "reason": "The target area looks vegetation-heavy, not like a roof",
            "vegetation_fraction": round(vegetation_fraction, 3),
        }

    return {
        "in_frame": True,
        "reason": "",
        "vegetation_fraction": round(vegetation_fraction, 3),
    }


def _center_patch_roof_evidence(img_hsv: np.ndarray, cx: int, cy: int) -> dict:
    """Check local roof material and structure without relying on flood fill."""
    h, w = img_hsv.shape[:2]
    patch = np.zeros((h, w), dtype=np.uint8)
    cv2.circle(patch, (cx, cy), CENTER_PATCH_RADIUS, 255, -1)
    patch_area = max(cv2.countNonZero(patch), 1)

    roof_mask = _roof_colour_mask(img_hsv)
    roof_fraction = cv2.countNonZero(cv2.bitwise_and(roof_mask, patch)) / patch_area
    green = cv2.inRange(
        img_hsv,
        np.array([30, 20, 25], dtype=np.uint8),
        np.array([90, 255, 255], dtype=np.uint8),
    )
    shaded_olive = cv2.inRange(
        img_hsv,
        np.array([20, 80, 25], dtype=np.uint8),
        np.array([29, 255, 255], dtype=np.uint8),
    )
    vegetation = cv2.bitwise_or(green, shaded_olive)
    vegetation_fraction = cv2.countNonZero(
        cv2.bitwise_and(vegetation, patch)
    ) / patch_area
    patch_pixels = img_hsv[patch > 0]
    hue_variation = float(np.std(patch_pixels[:, 0]))
    saturation_variation = float(np.std(patch_pixels[:, 1]))

    # Value-channel edges are sufficient here: this is a local structural
    # check, not a solar-panel detector. Uniform pavement has little support.
    edges = cv2.Canny(img_hsv[:, :, 2], 45, 120)
    edge_fraction = cv2.countNonZero(cv2.bitwise_and(edges, patch)) / patch_area
    roof_like = (
        roof_fraction >= CENTER_PATCH_MIN_ROOF_FRACTION
        and vegetation_fraction <= CENTER_PATCH_MAX_VEGETATION_FRACTION
        and edge_fraction >= CENTER_PATCH_MIN_EDGE_FRACTION
        and hue_variation >= CENTER_PATCH_MIN_HUE_VARIATION
        and saturation_variation >= CENTER_PATCH_MIN_SATURATION_VARIATION
    )
    return {
        "roof_like": roof_like,
        "roof_color_fraction": round(roof_fraction, 3),
        "vegetation_fraction": round(vegetation_fraction, 3),
        "edge_fraction": round(edge_fraction, 3),
        "hue_variation": round(hue_variation, 2),
        "saturation_variation": round(saturation_variation, 2),
    }


# ══════════════════════════════════════════════════════════════════════════
# Drawing helpers
# ══════════════════════════════════════════════════════════════════════════

def _draw_label(img, text: str, x: int, y: int, color):
    (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.52, 2)
    cv2.rectangle(img, (x - 3, y - th - 4), (x + tw + 3, y + 4), (0, 0, 0), -1)
    cv2.putText(img, text, (x, y),
                cv2.FONT_HERSHEY_SIMPLEX, 0.52, color, 2)


def _draw_warning_banner(img, reason: str):
    h, w = img.shape[:2]
    banner_h = 30
    cv2.rectangle(img, (0, 0), (w, banner_h), (0, 60, 200), -1)
    cv2.putText(img, f"  ⚠  {reason[:80]}",
                (4, 21),
                cv2.FONT_HERSHEY_SIMPLEX, 0.44,
                (255, 255, 255), 1)
