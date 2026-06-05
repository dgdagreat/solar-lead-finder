"""
Reference Comparator — Stage 5
================================
Compares the satellite image of a candidate home against a library of
images confirmed to have solar panels.

How it works
------------
For each reference image (and each candidate) we extract a 3-part
"solar fingerprint" from the pixels that pass the colour mask:

  1. HSV colour histogram  – the exact shade of the solar-panel blue/black
  2. Edge-density histogram – the regular grid pattern of panel boundaries
  3. Texture histogram      – Local Binary Pattern describing surface smoothness

We compute the Bhattacharyya distance between the candidate fingerprint
and every reference fingerprint, then convert the best match to a score
0 → 1.

Reference images are fetched once from Google Maps, saved to disk, and
re-used on every subsequent run (no repeated API calls).

Adding new references
---------------------
Call  add_reference(address)  with any address you have verified has
solar panels.  The image is fetched, fingerprinted, and persisted.
You can also promote a Home DB record:
    from detector.reference_comparator import add_reference
    add_reference(home.full_address)
"""

import cv2
import numpy as np
import requests
import pickle
import logging
from pathlib import Path
from io import BytesIO
from PIL import Image
from django.conf import settings

logger = logging.getLogger(__name__)

# ── Paths ─────────────────────────────────────────────────────────────────
MEDIA_ROOT    = Path(settings.MEDIA_ROOT)
REF_DIR       = MEDIA_ROOT / "solar_references"
NEG_DIR       = MEDIA_ROOT / "nosolar_references"
LIBRARY_FILE  = REF_DIR / "fingerprints.pkl"

# Stage 5 is now a TWO-CLASS margin classifier:
#   score = REF_BIAS + REF_GAIN * (best_solar_sim − best_nosolar_sim)
# Tuned empirically so confirmed-solar homes land >0.5 and no-solar homes <0.5.
REF_GAIN = 2.5
REF_BIAS = 0.5

# ── Seed addresses — California homes with documented rooftop solar ────────
# (high-solar-adoption neighbourhoods; verified visually)
SEED_ADDRESSES = [
    # Visually verified to have solar panels visible at Google Maps zoom 20
    "19848 Archwood St, Reseda, CA 91335",       # clear panel array on flat roof
    "9013 Sawyer St, Los Angeles, CA 90035",     # dark blue panels on grey roof
    "4820 Dunman Ave, Woodland Hills, CA 91364", # panels detected by CV pipeline
    "5618 Saint Clair Ave, Valley Village, CA 91607",
    "3133 Corte Canela, Camarillo, CA 93012",
    "1822 Sinaloa Rd, Simi Valley, CA 93065",
    "11259 Collins St, North Hollywood, CA 91601",
]


# ══════════════════════════════════════════════════════════════════════════
# Public API
# ══════════════════════════════════════════════════════════════════════════

def reference_scores(pil_image) -> dict:
    """
    Raw two-class comparison: how similar the candidate is to the best
    SOLAR reference vs the best NO-SOLAR reference.
    """
    library = _load_library()
    solar   = [r for r in library if r.get("label", "solar") == "solar"]
    nosolar = [r for r in library if r.get("label") == "no_solar"]

    fp = _fingerprint(pil_image) if library else None
    if fp is None:
        return {"best_solar": 0.0, "best_nosolar": 0.0,
                "n_solar": len(solar), "n_nosolar": len(nosolar), "has_fp": False}

    best_solar   = max((_compare_fingerprints(fp, r["fingerprint"]) for r in solar),   default=0.0)
    best_nosolar = max((_compare_fingerprints(fp, r["fingerprint"]) for r in nosolar), default=0.0)
    return {"best_solar": best_solar, "best_nosolar": best_nosolar,
            "n_solar": len(solar), "n_nosolar": len(nosolar), "has_fp": True}


def compare_to_references(pil_image) -> float:
    """
    Two-class margin score, 0.0–1.0:
      >0.5  → more like confirmed SOLAR than confirmed no-solar
      <0.5  → more like confirmed NO-SOLAR
    Falls back to plain best-solar similarity until negative references exist.
    """
    s = reference_scores(pil_image)
    if not s["has_fp"] or (s["n_solar"] == 0 and s["n_nosolar"] == 0):
        return 0.0
    if s["n_nosolar"] == 0:
        return s["best_solar"]          # no negatives yet → legacy behaviour

    margin = s["best_solar"] - s["best_nosolar"]
    score  = REF_BIAS + REF_GAIN * margin
    logger.debug(f"ref margin: solar={s['best_solar']:.3f} "
                 f"nosolar={s['best_nosolar']:.3f} margin={margin:+.3f} → {score:.3f}")
    return max(0.0, min(1.0, score))


def add_reference(address: str, label: str = "solar") -> bool:
    """
    Fetch the satellite image for `address`, fingerprint it, and add it to the
    persistent reference library under `label` ("solar" or "no_solar").
    Returns True on success.
    """
    from .image_fetcher import get_static_map_url

    url = get_static_map_url(address)
    if not url:
        return False

    try:
        resp = requests.get(url, timeout=15)
        resp.raise_for_status()
        pil_img = Image.open(BytesIO(resp.content)).convert("RGB")
    except Exception as e:
        logger.error(f"Failed to fetch reference image for {address}: {e}")
        return False

    fp = _fingerprint(pil_img)
    if fp is None:
        logger.warning(f"No recognisable region to fingerprint: {address}")
        return False

    library = _load_library()

    # Avoid duplicates (same address + same label)
    if any(r["address"] == address and r.get("label", "solar") == label for r in library):
        logger.info(f"Reference already exists [{label}]: {address}")
        return True

    # Save the image for inspection
    out_dir = REF_DIR if label == "solar" else NEG_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    safe_name = address.replace(" ", "_").replace(",", "")[:60]
    cv2.imwrite(str(out_dir / f"{safe_name}.jpg"),
                cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR))

    library.append({"address": address, "fingerprint": fp, "label": label})
    _save_library(library)
    logger.info(f"Reference added [{label}] ({len(library)} total): {address}")
    return True


def build_reference_library(force: bool = False):
    """
    Populate the library with SEED_ADDRESSES + any DB homes confirmed to
    have solar panels (solar_confidence >= 45%).
    Skips addresses already in the library unless force=True.
    """
    if force:
        if LIBRARY_FILE.exists():
            LIBRARY_FILE.unlink()

    library  = _load_library()
    existing = {r["address"] for r in library}

    added = 0
    # 1. Seed addresses
    for addr in SEED_ADDRESSES:
        if addr not in existing:
            if add_reference(addr):
                added += 1

    # 2. Confirmed solar homes from DB
    try:
        from homes.models import Home
        confirmed = Home.objects.filter(
            solar_status="has_solar",
            solar_confidence__gte=45
        )
        for home in confirmed:
            if home.full_address not in existing:
                if add_reference(home.full_address):
                    added += 1
    except Exception as e:
        logger.warning(f"Could not load DB confirmed solar homes: {e}")

    # 3. Negative (no-solar) references from confidently-no-solar DB homes
    added += seed_negatives_from_db()

    stats = library_stats()
    logger.info(f"Reference library ready — {stats['total']} entries "
                f"({stats['solar']} solar / {stats['no_solar']} no-solar, {added} new)")
    return stats


def add_from_db_confirmed():
    """
    Convenience: add all DB homes with solar_confidence >= 45 % to library.
    Call this after a batch scan to grow the reference set over time.
    """
    try:
        from homes.models import Home
        confirmed = Home.objects.filter(
            solar_status="has_solar", solar_confidence__gte=45
        )
        added = sum(1 for h in confirmed if add_reference(h.full_address))
        logger.info(f"Added {added} confirmed solar homes from DB to reference library")
        return added
    except Exception as e:
        logger.error(f"add_from_db_confirmed failed: {e}")
        return 0


def seed_negatives_from_db(limit: int = 10) -> int:
    """
    Add confidently-no-solar DB homes as NEGATIVE references. We pick homes
    whose CV features are near-zero (no colour cluster, no grid, no panel
    rectangles) — these are safe negatives even though the labels are
    machine-generated. Human "Confirm No Solar" clicks add stronger negatives.
    """
    try:
        from homes.models import Home
    except Exception as e:
        logger.warning(f"seed_negatives_from_db: cannot import Home: {e}")
        return 0

    existing = {(r["address"], r.get("label", "solar")) for r in _load_library()}
    candidates = []
    for h in Home.objects.filter(solar_status="no_solar").exclude(scan_detail=None):
        ss = (h.scan_detail or {}).get("stage_scores", {})
        color = ss.get("color_segmentation", 1.0)
        grid  = ss.get("grid_line_detection", 1.0)
        rect  = ss.get("rectangle_clustering", 1.0)
        if color < 0.05 and grid < 0.30 and rect < 0.10:
            if (h.full_address, "no_solar") not in existing:
                candidates.append((color + grid + rect, h))

    candidates.sort(key=lambda t: t[0])      # lowest-feature first = safest
    added = 0
    for _, h in candidates[:limit]:
        if add_reference(h.full_address, label="no_solar"):
            added += 1
    logger.info(f"Seeded {added} negative reference(s) from DB")
    return added


def library_stats() -> dict:
    lib = _load_library()
    return {
        "total":    len(lib),
        "solar":    sum(1 for r in lib if r.get("label", "solar") == "solar"),
        "no_solar": sum(1 for r in lib if r.get("label") == "no_solar"),
    }


def library_size() -> int:
    return len(_load_library())


# ══════════════════════════════════════════════════════════════════════════
# Fingerprinting
# ══════════════════════════════════════════════════════════════════════════

def _fingerprint(pil_image) -> dict | None:
    """
    Extract a 3-part fingerprint from the solar-panel coloured region.
    Returns None if the image has no recognisable solar-coloured pixels.
    """
    img_rgb  = np.array(pil_image)
    img_bgr  = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)
    img_hsv  = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2HSV)
    img_gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)

    # Build colour mask (same ranges as solar_detector Stage 1)
    mask = _solar_colour_mask(img_hsv)
    pixel_count = int(np.sum(mask > 0))

    if pixel_count < 200:          # too few matching pixels to fingerprint
        return None

    # ── Part 1: HSV colour histogram of masked pixels ─────────────────────
    hsv_hist = _masked_hist(img_hsv, mask, bins=32)

    # ── Part 2: Edge-density histogram (grid pattern) ─────────────────────
    edges = cv2.Canny(img_gray, 40, 120)
    edges_masked = cv2.bitwise_and(edges, edges, mask=mask)
    edge_hist = _masked_hist_gray(edges_masked, bins=32)

    # ── Part 3: Texture histogram (LBP-like: variance in 3x3 patches) ─────
    texture = _local_variance_map(img_gray)
    tex_masked = cv2.bitwise_and(texture, texture, mask=mask)
    tex_hist = _masked_hist_gray(tex_masked, bins=32)

    return {
        "hsv_hist":   hsv_hist,
        "edge_hist":  edge_hist,
        "tex_hist":   tex_hist,
        "pixel_count": pixel_count,
    }


def _compare_fingerprints(a: dict, b: dict) -> float:
    """
    Compare two fingerprints.  Returns similarity score 0–1.
    Uses Bhattacharyya distance: 0 = same, 1 = completely different.
    """
    try:
        d_hsv  = cv2.compareHist(a["hsv_hist"],  b["hsv_hist"],
                                  cv2.HISTCMP_BHATTACHARYYA)
        d_edge = cv2.compareHist(a["edge_hist"], b["edge_hist"],
                                  cv2.HISTCMP_BHATTACHARYYA)
        d_tex  = cv2.compareHist(a["tex_hist"],  b["tex_hist"],
                                  cv2.HISTCMP_BHATTACHARYYA)

        # Weighted combination — colour most important, edges next
        combined_dist = d_hsv * 0.50 + d_edge * 0.30 + d_tex * 0.20
        return max(0.0, 1.0 - combined_dist)
    except Exception:
        return 0.0


# ══════════════════════════════════════════════════════════════════════════
# Histogram helpers
# ══════════════════════════════════════════════════════════════════════════

def _solar_colour_mask(img_hsv: np.ndarray) -> np.ndarray:
    ranges = [
        (np.array([100, 55,  8]),  np.array([132, 255, 72])),   # navy blue
        (np.array([88,  18,  8]),  np.array([118,  75, 58])),   # blue-grey
    ]
    mask = np.zeros(img_hsv.shape[:2], dtype=np.uint8)
    for lo, hi in ranges:
        mask = cv2.bitwise_or(mask, cv2.inRange(img_hsv, lo, hi))
    k3 = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN,  k3, iterations=1)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k3, iterations=2)
    return mask


def _masked_hist(img_hsv: np.ndarray, mask: np.ndarray,
                 bins: int = 32) -> np.ndarray:
    """Compute a normalised joint H-S histogram over masked pixels."""
    hist = cv2.calcHist([img_hsv], [0, 1], mask,
                        [bins, bins], [0, 180, 0, 256])
    cv2.normalize(hist, hist, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX)
    return hist.astype(np.float32)


def _masked_hist_gray(img_gray: np.ndarray, bins: int = 32) -> np.ndarray:
    hist = cv2.calcHist([img_gray], [0], None,
                        [bins], [0, 256])
    cv2.normalize(hist, hist, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX)
    return hist.astype(np.float32)


def _local_variance_map(img_gray: np.ndarray) -> np.ndarray:
    """Compute local 3×3 variance as a proxy for texture roughness."""
    img_f   = img_gray.astype(np.float32)
    mean    = cv2.blur(img_f, (3, 3))
    mean_sq = cv2.blur(img_f ** 2, (3, 3))
    var     = np.clip(mean_sq - mean ** 2, 0, None)
    var_u8  = np.sqrt(var).astype(np.uint8)
    return var_u8


# ══════════════════════════════════════════════════════════════════════════
# Library persistence
# ══════════════════════════════════════════════════════════════════════════

def _load_library() -> list:
    if LIBRARY_FILE.exists():
        try:
            with open(LIBRARY_FILE, "rb") as f:
                lib = pickle.load(f)
            for r in lib:                 # backfill label for legacy entries
                r.setdefault("label", "solar")
            return lib
        except Exception:
            pass
    return []


def _save_library(library: list):
    REF_DIR.mkdir(parents=True, exist_ok=True)
    with open(LIBRARY_FILE, "wb") as f:
        pickle.dump(library, f)
