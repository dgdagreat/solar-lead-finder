"""
Tasks for fetching satellite images and running the solar detection pipeline.
process_home_sync runs directly in the web request (no Redis/Celery needed).
"""
from homes.models import Home
from .image_fetcher import get_static_map_url
from .solar_detector import detect_solar_panels_from_url
from .rooftop_annotator import fetch_and_annotate
import logging

logger = logging.getLogger(__name__)


def process_home_sync(home_id: int):
    """
    Full pipeline for one home:
      1. Fetch satellite image + annotate the target building
      2. Run 5-stage CV solar detector
      3. Save result + stage breakdown to DB
    """
    try:
        home = Home.objects.get(id=home_id)
    except Home.DoesNotExist:
        logger.error(f"Home #{home_id} not found.")
        return

    logger.info(f"Processing home #{home_id}: {home.full_address}")

    # Step 1 — fetch satellite image URL (for solar detection)
    image_url = get_static_map_url(home.full_address)
    if not image_url:
        home.solar_status = 'error'
        home.solar_confidence = 0
        home.stages_triggered = 0
        home.scan_detail = {
            "stage_scores": {},
            "details": {},
            "error": "Satellite scanning is not configured. Set GOOGLE_STREET_VIEW_API_KEY.",
        }
        home.save()
        return {"home_id": home_id, "status": home.solar_status}

    home.image_url = image_url

    # Step 2 — annotate the building in the image and save locally
    try:
        annotated_path = fetch_and_annotate(home)
    except Exception:
        # Annotation is a visual aid. It must not prevent the actual solar
        # scan from running if image decoding or OpenCV processing fails.
        logger.exception(f"Rooftop annotation failed for home #{home_id}; continuing scan")
        annotated_path = None
    if annotated_path:
        home.image = annotated_path   # serve the annotated version in the UI

    home.save()

    # Step 3 — run solar panel detector on the raw satellite image
    result, raw_image = detect_solar_panels_from_url(image_url, include_image=True)

    # Step 3b — learned override. Reuse the decoded scan image to avoid a second
    # satellite API request. Without a bundled model, keep the heuristic result.
    try:
        from . import cnn as learned_cnn
        if raw_image is not None and learned_cnn.is_available():
            pred = learned_cnn.predict(raw_image)
            if pred:
                result.setdefault("details", {})["cnn_prob"] = pred["prob"]
                result["has_solar"]  = pred["has_solar"]
                result["confidence"] = pred["prob"]
                result["method"]     = "mobilenetv3_cnn"
                result["stages_triggered"] = 0
                logger.info(f"CNN override [mobilenetv3]: "
                            f"P(solar)={pred['prob']:.3f} → has_solar={pred['has_solar']}")
    except Exception as e:
        logger.warning(f"CNN stage skipped: {e}")

    # Step 4 — save result
    if result.get("method") == "error":
        home.solar_status = 'error'
    elif result["has_solar"]:
        home.solar_status = 'has_solar'
        home.is_lead = False
    else:
        home.solar_status = 'no_solar'
        home.is_lead = True  # No solar = potential lead

    home.solar_confidence  = round((result.get("confidence") or 0) * 100, 1)
    home.stages_triggered  = (
        None if result.get("method") == "mobilenetv3_cnn"
        else result.get("stages_triggered", 0)
    )
    home.scan_detail       = {
        "method": result.get("method"),
        "stage_scores": result.get("stage_scores", {}),
        "details":      result.get("details", {}),
    }
    if result.get("method") == "error":
        home.scan_detail["error"] = result.get("error", "Satellite image scan failed.")
    home.save()

    if result.get("method") == "mobilenetv3_cnn":
        logger.info(f"Home #{home_id} → {home.solar_status} | model_score={home.solar_confidence}%")
    else:
        logger.info(
            f"Home #{home_id} → {home.solar_status} | "
            f"confidence={home.solar_confidence}% | stages={home.stages_triggered}/5"
        )
    return {"home_id": home_id, "status": home.solar_status}
