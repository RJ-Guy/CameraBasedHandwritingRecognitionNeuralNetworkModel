"""
test_ingestion.py - Automated Unit Tests for Phase 1: Optical Ingestion & Document Flattening

Verifies:
1. Valid JPEG/PNG optical loading and preservation of pristine copy.
2. Missing and corrupted file exception guarantees (FileNotFoundError, ValueError).
3. 4-point quadrilateral coordinate ordering (TL, TR, BR, BL).
4. Document boundary detection on synthetic tilted documents.
5. Perspective warp rectification and homography matrix computation.
6. Graceful bypass behavior when boundaries are absent or warp is disabled.
7. Manual corner coordinate fallback support.
"""

import sys
import tempfile
import unittest
from pathlib import Path
import cv2
import numpy as np

# Ensure Model package is discoverable on python path
TEST_DIR = Path(__file__).resolve().parent
MODEL_DIR = TEST_DIR.parent
if str(MODEL_DIR) not in sys.path:
    sys.path.insert(0, str(MODEL_DIR))

from ingestion import (
    DocumentIngestionEngine,
    detect_document_corners,
    ingest_image,
    load_image,
    order_quad_points,
    warp_perspective,
)
import config


class TestPhase1Ingestion(unittest.TestCase):
    """Test suite covering Phase 1 Ingestion specifications."""

    def setUp(self):
        self.engine = DocumentIngestionEngine(enable_warp=True, min_area_ratio=0.20)
        self.temp_dir = tempfile.TemporaryDirectory()
        self.temp_path = Path(self.temp_dir.name)

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_missing_file_raises_file_not_found(self):
        """Verify FileNotFoundError is raised with descriptive message for nonexistent file."""
        nonexistent = self.temp_path / "does_not_exist.jpg"
        with self.assertRaises(FileNotFoundError) as ctx:
            load_image(nonexistent)
        self.assertIn("not found", str(ctx.exception).lower())

    def test_corrupted_file_raises_value_error(self):
        """Verify ValueError is raised when encountering corrupted/zero-byte files."""
        corrupted_file = self.temp_path / "corrupted_image.png"
        corrupted_file.write_bytes(b"CORRUPTED_NON_IMAGE_HEADER_BYTES_12345")
        with self.assertRaises(ValueError) as ctx:
            load_image(corrupted_file)
        self.assertIn("failed to decode", str(ctx.exception).lower())

    def test_load_valid_image_and_untouched_preservation(self):
        """Verify valid image loading and ensure original copy is an independent deep copy."""
        # Create a valid test image
        test_img = np.zeros((100, 150, 3), dtype=np.uint8)
        test_img[10:90, 10:140] = [200, 200, 200]
        test_file = self.temp_path / "valid_test.png"
        cv2.imwrite(str(test_file), test_img)

        original, raw = load_image(test_file)
        self.assertEqual(original.shape, (100, 150, 3))
        self.assertEqual(raw.shape, (100, 150, 3))
        self.assertTrue(np.array_equal(original, raw))

        # Mutate raw to verify deep copy independence
        raw[0, 0] = [255, 0, 0]
        self.assertFalse(np.array_equal(original, raw))
        self.assertEqual(original[0, 0, 0], 0)

    def test_order_quad_points_deterministic(self):
        """Verify quadrilateral points are ordered [TL, TR, BR, BL] regardless of input order."""
        # Standard rectangle: TL=(10,10), TR=(90,10), BR=(90,80), BL=(10,80)
        true_tl = [10.0, 10.0]
        true_tr = [90.0, 10.0]
        true_br = [90.0, 80.0]
        true_bl = [10.0, 80.0]

        # Scramble points
        scrambled = np.array([true_br, true_tl, true_bl, true_tr], dtype=np.float32)
        ordered = order_quad_points(scrambled)

        np.testing.assert_allclose(ordered[0], true_tl, err_msg="Top-Left mismatch")
        np.testing.assert_allclose(ordered[1], true_tr, err_msg="Top-Right mismatch")
        np.testing.assert_allclose(ordered[2], true_br, err_msg="Bottom-Right mismatch")
        np.testing.assert_allclose(ordered[3], true_bl, err_msg="Bottom-Left mismatch")

    def test_synthetic_document_corner_detection_and_warp(self):
        """Verify automatic paper boundary detection and perspective warp on synthetic tilted sheet."""
        synth_img, true_corners = DocumentIngestionEngine.generate_synthetic_document_sample(
            width=1280, height=720
        )
        detected = detect_document_corners(synth_img, min_area_ratio=0.15)

        self.assertIsNotNone(detected, "Failed to detect synthetic document quadrilateral")
        self.assertEqual(detected.shape, (4, 2))

        # Each detected corner should be within 15 pixels of true corner
        ordered_true = order_quad_points(true_corners)
        for i in range(4):
            dist = np.linalg.norm(detected[i] - ordered_true[i])
            self.assertLess(
                dist, 15.0, f"Corner {i} deviation ({dist:.2f}px) exceeds tolerance"
            )

        # Verify warp execution
        flattened, M = warp_perspective(synth_img, detected)
        self.assertIsNotNone(flattened)
        self.assertEqual(M.shape, (3, 3))
        self.assertGreater(flattened.shape[0], 200)
        self.assertGreater(flattened.shape[1], 300)

    def test_graceful_bypass_on_uniform_image(self):
        """Verify graceful bypass when no 4-corner document quad is present."""
        # Uniform gray image (no paper edges)
        uniform_img = np.full((400, 600, 3), 128, dtype=np.uint8)

        original, ingested, meta = ingest_image(uniform_img, enable_warp=True)

        self.assertFalse(meta["warp_applied"], "Warp should be bypassed on uniform image")
        self.assertIsNone(meta["corners"])
        self.assertIsNone(meta["M"])
        self.assertEqual(ingested.shape, original.shape)
        self.assertTrue(np.array_equal(original, ingested))

    def test_warp_bypass_when_disabled_in_config(self):
        """Verify that disabling warp immediately passes the full frame untouched."""
        synth_img, _ = DocumentIngestionEngine.generate_synthetic_document_sample()
        original, ingested, meta = ingest_image(synth_img, enable_warp=False)

        self.assertFalse(meta["warp_applied"])
        self.assertEqual(ingested.shape, synth_img.shape)
        self.assertTrue(np.array_equal(ingested, synth_img))

    def test_manual_perspective_corners(self):
        """Verify explicit manual corner overrides."""
        test_img = np.full((500, 500, 3), 100, dtype=np.uint8)
        manual = [(50, 50), (450, 60), (440, 450), (60, 440)]

        original, ingested, meta = ingest_image(test_img, manual_corners=manual)

        self.assertTrue(meta["warp_applied"])
        self.assertIsNotNone(meta["corners"])
        self.assertIsNotNone(meta["M"])
        self.assertGreater(ingested.shape[0], 0)
        self.assertGreater(ingested.shape[1], 0)

    def test_samples_directory_image_loads_cleanly(self):
        """Verify loading existing camera capture samples from data/samples."""
        sample_path = config.SAMPLES_DIR / "sample_camera_capture.png"
        if sample_path.exists():
            original, ingested, meta = ingest_image(sample_path)
            self.assertIsNotNone(original)
            self.assertIsNotNone(ingested)
            self.assertIn("warp_applied", meta)
            self.assertEqual(original.shape[2], 3)


if __name__ == "__main__":
    unittest.main()
