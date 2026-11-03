from pathlib import Path
from unittest.mock import patch

import numpy as np
from PIL import Image

from django.test import TestCase
from homes.models import Home

from detector.solar_detector import (
    detect_solar_panels, detect_solar_panels_from_path,
    detect_solar_panels_from_url, _error_result)
from detector import reference_comparator
from detector.tasks import process_home_sync

FIXTURES = Path(__file__).resolve().parent / "test_data"


class DetectorTests(TestCase):
    def test_blank_image_reports_no_solar(self):
        # Solid green (grass) — no panels, no roof, no blue clusters
        img = Image.new("RGB", (300, 300), (60, 140, 60))
        result = detect_solar_panels(img)

        self.assertFalse(result["has_solar"])
        self.assertEqual(result["method"], "multi_stage_cv_v2")
        for stage in ("color_segmentation", "grid_line_detection", "rectangle_clustering",
                      "dark_blob_uniformity", "reference_similarity"):
            self.assertIn(stage, result["stage_scores"])
        self.assertGreaterEqual(result["stages_triggered"], 0)

    def test_error_result_schema(self):
        result = _error_result("boom")
        self.assertEqual(result["method"], "error")
        self.assertFalse(result["has_solar"])
        self.assertEqual(result["error"], "boom")

    @patch("detector.solar_detector.requests.get")
    def test_http_scan_error_reports_status_without_exposing_api_key(self, get):
        response = get.return_value
        response.ok = False
        response.status_code = 403
        response.text = "denied key=sample-secret"

        result = detect_solar_panels_from_url(
            "https://maps.example/staticmap?key=sample-secret"
        )

        self.assertEqual(result["method"], "error")
        self.assertIn("HTTP 403", result["error"])
        self.assertNotIn("sample-secret", result["error"])
        self.assertIn("[redacted]", result["error"])

    @patch("detector.reference_comparator.compare_to_references", return_value=0.0)
    @patch("detector.solar_detector._stage_dark_blobs", return_value=(0.0, {}))
    @patch("detector.solar_detector._stage_rectangles", return_value=(0.0, {}))
    @patch("detector.solar_detector._stage_grid", return_value=(0.0, {}))
    @patch("detector.solar_detector._stage_color")
    def test_single_strong_color_signal_does_not_bypass_agreement_rule(
        self, stage_color, *_
    ):
        stage_color.return_value = (
            0.95,
            np.zeros((64, 64), dtype=np.uint8),
            {"cluster_ratio": 0.9},
        )
        result = detect_solar_panels(Image.new("RGB", (64, 64), (80, 80, 80)))

        self.assertFalse(result["has_solar"])
        self.assertEqual(result["stages_triggered"], 1)

    @patch("detector.reference_comparator.compare_to_references", return_value=0.0)
    @patch("detector.solar_detector._stage_dark_blobs", return_value=(0.0, {}))
    @patch("detector.solar_detector._stage_rectangles", return_value=(0.0, {}))
    @patch("detector.solar_detector._stage_grid", return_value=(0.8, {}))
    @patch("detector.solar_detector._stage_color")
    def test_two_agreeing_stages_can_flag_solar(
        self, stage_color, *_
    ):
        stage_color.return_value = (
            0.8,
            np.zeros((64, 64), dtype=np.uint8),
            {"cluster_ratio": 0.8},
        )
        result = detect_solar_panels(Image.new("RGB", (64, 64), (80, 80, 80)))

        self.assertTrue(result["has_solar"])
        self.assertEqual(result["stages_triggered"], 2)


class ReferenceComparatorTests(TestCase):
    def test_score_in_unit_range(self):
        # A solid image has no fingerprintable solar region -> safe 0.0..1.0 score
        img = Image.new("RGB", (200, 200), (200, 200, 200))
        score = reference_comparator.compare_to_references(img)
        self.assertGreaterEqual(score, 0.0)
        self.assertLessEqual(score, 1.0)


class ScanTaskTests(TestCase):
    def setUp(self):
        self.home = Home.objects.create(
            address="17 Example St",
            city="Testville",
            state="CA",
            zip_code="90000",
        )

    @patch("detector.tasks.get_static_map_url", return_value=None)
    def test_missing_maps_key_is_saved_as_actionable_scan_error(self, _get_url):
        process_home_sync(self.home.id)
        self.home.refresh_from_db()

        self.assertEqual(self.home.solar_status, "error")
        self.assertIn("GOOGLE_STREET_VIEW_API_KEY", self.home.scan_detail["error"])

    @patch("detector.tasks.detect_solar_panels_from_url")
    @patch("detector.tasks.fetch_and_annotate", side_effect=RuntimeError("bad image"))
    @patch("detector.tasks.get_static_map_url", return_value="https://maps.example/tile")
    def test_annotation_failure_does_not_stop_detection_and_error_is_saved(
        self, _get_url, _annotate, detect
    ):
        detect.return_value = {
            "method": "error",
            "has_solar": False,
            "confidence": 0.0,
            "stages_triggered": 0,
            "stage_scores": {},
            "details": {},
            "error": "Satellite imagery service returned HTTP 403",
        }

        process_home_sync(self.home.id)
        self.home.refresh_from_db()

        detect.assert_called_once_with("https://maps.example/tile")
        self.assertEqual(self.home.solar_status, "error")
        self.assertEqual(
            self.home.scan_detail["error"],
            "Satellite imagery service returned HTTP 403",
        )


class RealImageTests(TestCase):
    """Regression tests on real satellite rooftop images (see test_data/)."""

    def test_solar_rooftop_is_detected(self):
        result = detect_solar_panels_from_path(str(FIXTURES / "solar_rooftop.jpg"))
        self.assertTrue(result["has_solar"],
                        f"expected has_solar=True, got conf={result['confidence']}")

    def test_no_solar_rooftop_is_not_flagged(self):
        result = detect_solar_panels_from_path(str(FIXTURES / "no_solar_rooftop.jpg"))
        self.assertFalse(result["has_solar"],
                         f"expected has_solar=False, got conf={result['confidence']}")
