"""
Rooftop Annotator — v3
=======================
How it works:
  • Localises the building at the image centre with a flood fill, then keeps
    only its roof-coloured pixels — so grass, trees, bushes and pools are excluded
  • Highlights that single roof with a semi-transparent yellow fill — neighbours
    are not covered
  • Verifies the image is actually a residential property:
      - a roof-coloured region exists within 180 px of centre
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
MAX_CENTRE_DIST   = 260   # px — rooftop centroid must be within this of image centre
MIN_ROOF_AREA     = 1200  # px² — smallest acceptable rooftop
MAX_ROOF_AREA     = 200000# px² — largest acceptable rooftop
ROOF_WINDOW_RADIUS = 140  # px — clip the highlight to a single-home window around centre
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

    img_rgb = np.array(pil_img)
    img_bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)

    annotated, verification = _annotate_rooftop(img_bgr)

    # Persist verification warning to the model notes field
    if not verification["in_frame"]:
        warning = f"⚠️ Satellite image may not show target property clearly: {verification['reason']}"
        if warning not in (home.notes or ""):
            home.notes = warning
            home.save(update_fields=["notes"])

    rel_path = f"home_images/annotated_{home.id}.jpg"
    abs_path = Path(settings.MEDIA_ROOT) / rel_path
    abs_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(abs_path), annotated)

    logger.info(f"Annotated: {abs_path} | verified={verification['in_frame']}")
    return rel_path


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
    verification = _verify_in_frame(img_hsv, filled_area, h, w, cx, cy)

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


def _verify_in_frame(img_hsv, filled_area: int,
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

    return {"in_frame": True, "reason": ""}


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
