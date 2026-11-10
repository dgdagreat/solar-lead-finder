"""
Tasks for fetching satellite images and running the solar detection pipeline.
process_home_sync runs directly in the web request (no Redis/Celery needed).
"""
from homes.models import Home
from .image_fetcher import get_static_map_url
from .solar_detector import detect_solar_panels_from_url
from .rooftop_annotator import annotate_rooftop_image
import logging

logger = logging.getLogger(__name__)


def process_home_sync(home_id: int):
    """
    Full pipeline for one home:
      1. Fetch one satellite image and run roof verification on it
      2. Classify solar only when the target roof is verified
      3. Save result + verification details to DB
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

    home.save()

    # Step 2 — fetch once, then use this same image for roof verification and solar detection.
    result, raw_image = detect_solar_panels_from_url(image_url, include_image=True)

    roof_verification = None
    if raw_image is not None:
        try:
            annotated_path, roof_verification = annotate_rooftop_image(home, raw_image)
            if annotated_path:
                home.image = annotated_path
        except Exception:
            logger.exception(f"Roof verification failed for home #{home_id}")
            roof_verification = {
                "in_frame": False,
                "reason": "Could not verify that the image shows the target roof.",
            }
            home.notes = f"⚠️ {roof_verification['reason']}"
        home.save()

    # Fail closed: without a verified target roof, a tree, bush, road, or
    # neighboring roof must never be turned into a solar label or sales lead.
    if result.get("method") != "error" and not (roof_verification or {}).get("in_frame"):
        reason = (roof_verification or {}).get(
            "reason", "Satellite image did not provide a verifiable target roof."
        )
        result = {
            "method": "roof_unverified",
            "has_solar": False,
            "confidence": None,
            "stages_triggered": None,
            "stage_scores": {},
            "details": {},
            "roof_verification": {"in_frame": False, "reason": reason},
        }

    # Step 3 — learned override, only after the target roof was verified.
    try:
        from . import cnn as learned_cnn
        if (result.get("method") != "error" and raw_image is not None
                and (roof_verification or {}).get("in_frame")
                and learned_cnn.is_available()):
            pred = learned_cnn.predict(raw_image)
            if pred:
                result.setdefault("details", {})["cnn_prob"] = pred["prob"]
                result["confidence"] = pred["prob"]
                cnn_supported = learned_cnn.is_solar_supported(
                    pred["prob"], result.get("stage_scores", {})
                )
                if pred["has_solar"] and not cnn_supported:
                    # Medium CNN scores need independent panel-like grid or
                    # repeated-rectangle evidence; shadows tend to lack it.
                    result["method"] = "solar_review"
                    result["review_reason"] = (
                        "Solar scan needs review: the image classifier found a "
                        "possible match, but the image lacks enough repeated "
                        "panel-grid or rectangle structure. Shadows can resemble "
                        "panels. Verify the roof image manually."
                    )
                    logger.info(
                        "CNN solar candidate lacks structure: P(solar)=%.3f grid=%.3f rectangles=%.3f; review required",
                        pred["prob"],
                        result.get("stage_scores", {}).get("grid_line_detection", 0.0),
                        result.get("stage_scores", {}).get("rectangle_clustering", 0.0),
                    )
                elif not pred["has_solar"] and result.get("has_solar"):
                    # The two independent classifiers disagree; neither a
                    # solar label nor a lead is safe to assign automatically.
                    result["method"] = "solar_review"
                    result["review_reason"] = (
                        "Solar scan needs review: the image classifier and "
                        "panel-structure checks disagree. Verify the roof image manually."
                    )
                else:
                    result["has_solar"] = cnn_supported
                    result["method"] = "mobilenetv3_cnn"
                    result["stages_triggered"] = 0
                    logger.info(f"CNN agrees with physical checks: "
                                f"P(solar)={pred['prob']:.3f} → has_solar={pred['has_solar']}")
    except Exception as e:
        logger.warning(f"CNN stage skipped: {e}")

    # Step 4 — save result
    if result.get("method") == "error":
        home.solar_status = 'error'
        home.is_lead = False
    elif result.get("method") == "roof_unverified":
        home.solar_status = 'unverified'
        home.is_lead = False
    elif result.get("method") == "solar_review":
        home.solar_status = 'unverified'
        home.is_lead = False
    elif result["has_solar"]:
        home.solar_status = 'has_solar'
        home.is_lead = False
    else:
        home.solar_status = 'no_solar'
        home.is_lead = True  # No solar = potential lead

    confidence = result.get("confidence")
    home.solar_confidence = round(confidence * 100, 1) if confidence is not None else None
    home.stages_triggered  = (
        None if result.get("method") == "mobilenetv3_cnn"
        else result.get("stages_triggered", 0)
    )
    home.scan_detail       = {
        "method": result.get("method"),
        "stage_scores": result.get("stage_scores", {}),
        "details":      result.get("details", {}),
    }
    if result.get("review_reason"):
        home.scan_detail["review_reason"] = result["review_reason"]
    if result.get("roof_verification") or roof_verification:
        home.scan_detail["roof_verification"] = (
            result.get("roof_verification") or roof_verification
        )
    if result.get("method") == "error":
        home.scan_detail["error"] = result.get("error", "Satellite image scan failed.")
    home.save()

    if result.get("method") in ("mobilenetv3_cnn", "solar_review"):
        logger.info(f"Home #{home_id} → {home.solar_status} | model_score={home.solar_confidence}%")
    else:
        logger.info(
            f"Home #{home_id} → {home.solar_status} | "
            f"confidence={home.solar_confidence}% | stages={home.stages_triggered}/5"
        )
    return {"home_id": home_id, "status": home.solar_status}
