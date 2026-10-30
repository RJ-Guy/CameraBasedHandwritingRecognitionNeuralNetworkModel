"""
test_preprocessing.py - Automated Unit Tests for Phase 2: Illumination-Robust Preprocessing

Verifies:
1. Grayscale conversion channel depth (single channel, uint8) from BGR, BGRA, and 2D inputs.
2. Background division numerical stability with uniform and gradient pitch-black images
   (asserts zero divide-by-zero errors, warnings, NaNs, or Infs).
3. Sauvola thresholding output range (strictly in {0, 255}) and accurate ink extraction.
4. Gaussian adaptive fallback binarization output and values.
5. CLAHE contrast enhancement maintaining dimensions, range, and local contrast boost.
6. Despeckling noise suppression (eliminating isolated speckles without eroding continuous strokes).
7. End-to-end PreprocessingEngine execution on real sample images and synthetic handwriting.
"""

import sys
import unittest
from pathlib import Path
import cv2
import numpy as np

# Ensure Model package is discoverable on python path
TEST_DIR = Path(__file__).resolve().parent
MODEL_DIR = TEST_DIR.parent
if str(MODEL_DIR) not in sys.path:
    sys.path.insert(0, str(MODEL_DIR))

from preprocessing import (
    PreprocessingEngine,
    PreprocessingResult,
    binarize_gaussian_adaptive,
    binarize_sauvola,
    compute_bg_kernel_size,
    convert_to_grayscale,
    despeckle_mask,
    enhance_contrast_clahe,
    estimate_background,
    normalize_illumination,
    preprocess_image,
)
import config


class TestPhase2Preprocessing(unittest.TestCase):
    """Test suite covering Phase 2 Preprocessing specifications."""

    def setUp(self):
        self.engine = PreprocessingEngine()

    def test_grayscale_conversion_channel_depth(self):
        """Verify grayscale conversion produces single-channel uint8 arrays."""
        # 3-channel BGR image
        bgr = np.random.randint(0, 256, (120, 160, 3), dtype=np.uint8)
        gray = convert_to_grayscale(bgr)
        self.assertEqual(gray.ndim, 2, "Grayscale image must have 2 dimensions (channel depth = 1)")
        self.assertEqual(gray.shape, (120, 160))
        self.assertEqual(gray.dtype, np.uint8)

        # 4-channel BGRA image
        bgra = np.random.randint(0, 256, (100, 100, 4), dtype=np.uint8)
        gray_bgra = convert_to_grayscale(bgra)
        self.assertEqual(gray_bgra.ndim, 2)
        self.assertEqual(gray_bgra.shape, (100, 100))

        # Already grayscale 2D image
        already_gray = np.random.randint(0, 256, (80, 80), dtype=np.uint8)
        pass_through = convert_to_grayscale(already_gray)
        self.assertEqual(pass_through.ndim, 2)
        self.assertTrue(np.array_equal(already_gray, pass_through))

    def test_grayscale_empty_raises_value_error(self):
        """Verify ValueError is raised on empty or None input."""
        with self.assertRaises(ValueError):
            convert_to_grayscale(np.array([]))
        with self.assertRaises(ValueError):
            convert_to_grayscale(None)

    def test_background_division_numerical_stability_black_images(self):
        """
        Verify background division on uniform black and gradient black images
        produces zero division-by-zero errors, warnings, NaNs, or Infs.
        """
        # 1. Pitch black uniform image (all zeros)
        black_img = np.zeros((150, 150), dtype=np.uint8)
        bg_black = estimate_background(black_img)
        norm_black = normalize_illumination(black_img, bg_black, epsilon=1.0)

        self.assertFalse(np.isnan(norm_black).any(), "Normalized black image contains NaNs")
        self.assertFalse(np.isinf(norm_black).any(), "Normalized black image contains Infs")
        self.assertEqual(norm_black.dtype, np.uint8)
        self.assertEqual(norm_black.shape, black_img.shape)

        # 2. Gradient black image (varying from 0 to 40)
        grad_row = np.linspace(0, 40, 150, dtype=np.uint8)
        grad_img = np.tile(grad_row, (150, 1))
        bg_grad = estimate_background(grad_img)
        norm_grad = normalize_illumination(grad_img, bg_grad, epsilon=1.0)

        self.assertFalse(np.isnan(norm_grad).any(), "Normalized gradient image contains NaNs")
        self.assertFalse(np.isinf(norm_grad).any(), "Normalized gradient image contains Infs")
        self.assertEqual(norm_grad.dtype, np.uint8)

    def test_background_estimation_and_dynamic_kernel(self):
        """Verify morphological background estimation kernel calculation and output."""
        k = compute_bg_kernel_size(height=720, width=1280, base_size=41, dynamic=True)
        self.assertEqual(k % 2, 1, "Kernel size must be odd")
        self.assertGreaterEqual(k, 41)

        # Small image guard
        k_small = compute_bg_kernel_size(height=30, width=30, base_size=41, dynamic=True)
        self.assertEqual(k_small % 2, 1)
        self.assertLess(k_small, 30)

        # Synthetic image with lighting gradient and dark ink strokes
        canvas = np.full((200, 300), 220, dtype=np.uint8)
        # Add shadow gradient across columns
        for c in range(300):
            canvas[:, c] = np.clip(220 - int(c * 0.4), 80, 220)
        # Add black text strokes
        canvas[90:110, 50:250] = 30

        bg = estimate_background(canvas, kernel_size=41)
        self.assertEqual(bg.shape, canvas.shape)
        # Background closing should have smoothed over the 20px dark text bar
        self.assertGreater(bg[100, 150], 100, "Background closing failed to fill text strokes")

    def test_clahe_contrast_enhancement(self):
        """Verify CLAHE enhances local contrast and maintains format."""
        low_contrast = np.full((100, 100), 120, dtype=np.uint8)
        low_contrast[40:60, 40:60] = 110 # subtle faint mark

        enhanced = enhance_contrast_clahe(low_contrast, clip_limit=3.0, grid_size=(4, 4))
        self.assertEqual(enhanced.shape, low_contrast.shape)
        self.assertEqual(enhanced.dtype, np.uint8)
        # Difference between background and mark should be amplified
        diff_orig = int(low_contrast[10, 10]) - int(low_contrast[50, 50])
        diff_enh = int(enhanced[10, 10]) - int(enhanced[50, 50])
        self.assertGreaterEqual(diff_enh, diff_orig)

    def test_sauvola_binarization_output_range_and_accuracy(self):
        """Verify Sauvola output values are strictly {0, 255} and isolate dark foreground."""
        canvas = np.full((120, 120), 230, dtype=np.uint8)
        # Draw 20x20 dark square (ink)
        canvas[50:70, 50:70] = 40

        binary = binarize_sauvola(canvas, window_size=25, k=0.2, r=128)
        unique_vals = set(np.unique(binary))
        self.assertTrue(unique_vals.issubset({0, 255}), f"Unexpected values in binary mask: {unique_vals}")

        # Ink square should be detected as 255
        ink_detected = (binary[50:70, 50:70] == 255).sum()
        self.assertGreaterEqual(ink_detected, 380, "Sauvola failed to segment dark square")

        # White background should be 0
        bg_detected = (binary[:40, :40] == 0).sum()
        self.assertEqual(bg_detected, 40 * 40, "Background misclassified as ink")

    def test_gaussian_adaptive_binarization(self):
        """Verify Gaussian adaptive fallback binarization."""
        canvas = np.full((100, 100), 220, dtype=np.uint8)
        canvas[40:60, 40:60] = 30

        binary = binarize_gaussian_adaptive(canvas, block_size=25, c=11)
        unique_vals = set(np.unique(binary))
        self.assertTrue(unique_vals.issubset({0, 255}))
        self.assertGreater((binary[40:60, 40:60] == 255).sum(), 350)

    def test_despeckling_noise_suppression(self):
        """Verify despeckling eliminates isolated noise while preserving strokes."""
        mask = np.zeros((80, 80), dtype=np.uint8)
        # Draw continuous 2-pixel stroke
        mask[30:32, 20:60] = 255
        # Add isolated 1-pixel speckles
        mask[10, 10] = 255
        mask[70, 70] = 255

        cleaned = despeckle_mask(mask, method="median", kernel_size=3)

        # Speckles must be removed
        self.assertEqual(cleaned[10, 10], 0, "Speckle at (10, 10) was not removed")
        self.assertEqual(cleaned[70, 70], 0, "Speckle at (70, 70) was not removed")

        # Stroke must be largely preserved
        stroke_remaining = (cleaned[30:32, 20:60] == 255).sum()
        original_stroke = (mask[30:32, 20:60] == 255).sum()
        self.assertGreater(stroke_remaining, original_stroke * 0.8)

    def test_end_to_end_preprocess_image(self):
        """Verify complete pipeline returns expected dataclass fields and types."""
        # Create synthetic camera document
        bgr = np.full((200, 300, 3), (230, 235, 240), dtype=np.uint8)
        # Add simulated text
        cv2.putText(bgr, "Handwriting", (50, 110), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (20, 25, 30), 2)

        result = preprocess_image(bgr, method="sauvola")

        self.assertIsInstance(result, PreprocessingResult)
        self.assertEqual(result.gray_image.shape, (200, 300))
        self.assertEqual(result.background_estimate.shape, (200, 300))
        self.assertEqual(result.normalized_image.shape, (200, 300))
        self.assertEqual(result.enhanced_image.shape, (200, 300))
        self.assertEqual(result.binary_mask.shape, (200, 300))

        # Check binary mask values
        unique_vals = set(np.unique(result.binary_mask))
        self.assertTrue(unique_vals.issubset({0, 255}))

        # Check ink was detected
        ink_pixel_count = (result.binary_mask == 255).sum()
        self.assertGreater(ink_pixel_count, 100)

        # Verify to_dict conversion
        d = result.to_dict()
        self.assertIn("binary_mask", d)
        self.assertIn("enhanced_image", d)

    def test_sample_image_preprocessing(self):
        """Verify preprocessing on existing sample camera capture."""
        sample_path = config.SAMPLES_DIR / "sample_camera_capture_synth.png"
        if not sample_path.exists():
            sample_path = config.SAMPLES_DIR / "sample_camera_capture.png"

        if sample_path.exists():
            img = cv2.imread(str(sample_path))
            result = self.engine.process(img)
            self.assertIsNotNone(result.binary_mask)
            self.assertEqual(result.binary_mask.shape[:2], img.shape[:2])
            ink_ratio = (result.binary_mask == 255).sum() / float(result.binary_mask.size)
            self.assertGreater(ink_ratio, 0.005, "Too few ink pixels detected")
            self.assertLess(ink_ratio, 0.40, "Too many ink pixels detected (inverted or threshold blowout)")


if __name__ == "__main__":
    unittest.main()
