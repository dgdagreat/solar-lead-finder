from PIL import Image

from django.test import TestCase

from detector.solar_detector import detect_solar_panels, _error_result
from detector import reference_comparator


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


class ReferenceComparatorTests(TestCase):
    def test_score_in_unit_range(self):
        # A solid image has no fingerprintable solar region -> safe 0.0..1.0 score
        img = Image.new("RGB", (200, 200), (200, 200, 200))
        score = reference_comparator.compare_to_references(img)
        self.assertGreaterEqual(score, 0.0)
        self.assertLessEqual(score, 1.0)
