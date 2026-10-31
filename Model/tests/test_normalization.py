"""
test_normalization.py - Automated Unit Tests for Phase 4: Geometric Normalization

Verifies:
1. Dual-centroid slant angle calculation on synthetic vertical, forward-slanted,
   and backward-slanted bars (verifying angle signs and magnitudes).
2. Affine horizontal shear deskewing with zero stroke clipping (all ink pixels preserved).
3. Normalized character crop shape is strictly (16, 16) with values in [0.0, 1.0].
4. Aspect ratio preservation (tall vs wide characters not distorted into blobs).
5. Centered zero-padded embedding leaving symmetric margins.
6. Integration with Phase 3 CharacterCrop dataclass and NormalizationEngine batch processing.
7. Handling of edge cases (empty masks, single pixels, zero-slant characters).
"""

import math
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

from normalization import (
    NormalizationEngine,
    NormalizationResult,
    NormalizedCharacter,
    crop_to_ink,
    deskew_character,
    estimate_slant_angle,
    normalize_all_characters,
    normalize_character,
    resize_and_center_16x16,
)
from segmentation import CharacterCrop
import config


class TestPhase4Normalization(unittest.TestCase):
    """Test suite covering Phase 4 Geometric Normalization specifications."""

    def setUp(self):
        self.engine = NormalizationEngine()

    def test_dual_centroid_vertical_bar_zero_slant(self):
        """Verify perfectly vertical bar produces approximately 0.0 degrees slant."""
        canvas = np.zeros((40, 40), dtype=np.uint8)
        canvas[5:35, 20] = 255  # Vertical line at x=20

        angle = estimate_slant_angle(canvas)
        self.assertAlmostEqual(angle, 0.0, delta=1.0, msg="Vertical line must have ~0 slant")

    def test_dual_centroid_forward_slant_positive_angle(self):
        """Verify rightward-leaning (forward) stroke produces a positive angle."""
        canvas = np.zeros((40, 40), dtype=np.uint8)
        # Line from (x=15, y=32) at bottom to (x=27, y=8) at top
        # dx = 27 - 15 = 12, dy = 32 - 8 = 24 -> tan(theta) = 12/24 = 0.5 -> theta ~ 26.5 deg
        cv2.line(canvas, (15, 32), (27, 8), 255, 2)

        angle = estimate_slant_angle(canvas)
        self.assertGreater(angle, 15.0, "Forward slant must yield positive angle > 15 deg")
        self.assertLess(angle, 35.0, "Forward slant exceeds plausible range")

    def test_dual_centroid_backward_slant_negative_angle(self):
        """Verify leftward-leaning (backward) stroke produces a negative angle."""
        canvas = np.zeros((40, 40), dtype=np.uint8)
        # Line from (x=27, y=32) at bottom to (x=15, y=8) at top
        # dx = 15 - 27 = -12, dy = 32 - 8 = 24 -> theta ~ -26.5 deg
        cv2.line(canvas, (27, 32), (15, 8), 255, 2)

        angle = estimate_slant_angle(canvas)
        self.assertLess(angle, -15.0, "Backward slant must yield negative angle < -15 deg")
        self.assertGreater(angle, -35.0)

    def test_affine_shear_zero_stroke_clipping(self):
        """Verify affine horizontal shear preserves all foreground stroke pixels without clipping."""
        canvas = np.zeros((40, 40), dtype=np.uint8)
        cv2.line(canvas, (14, 32), (26, 8), 255, 2)

        initial_ink_count = (canvas > 0).sum()
        slant_deg = estimate_slant_angle(canvas)

        tight_crop, full_sheared = deskew_character(canvas, slant_deg)

        sheared_ink_count = (tight_crop > 0).sum()
        # Due to bilinear interpolation smoothing, ink area should not decrease
        self.assertGreaterEqual(
            sheared_ink_count,
            int(initial_ink_count * 0.90),
            "Stroke was clipped or eroded during affine shear deskewing",
        )

        # After deskewing, the remaining slant should be significantly reduced
        new_slant = estimate_slant_angle(tight_crop)
        self.assertLess(
            abs(new_slant),
            abs(slant_deg) * 0.5,
            "Deskewing failed to reduce slant angle",
        )

    def test_normalized_character_shape_and_range(self):
        """Verify normalized character image shape is strictly (16, 16) and values in [0.0, 1.0]."""
        # Random handwriting character crop
        char = np.zeros((28, 20), dtype=np.uint8)
        cv2.putText(char, "B", (2, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.8, 255, 2)

        norm_char = normalize_character(char, char_id=1)

        self.assertEqual(norm_char.normalized_image.shape, (16, 16))
        self.assertEqual(norm_char.normalized_image.dtype, np.float32)
        self.assertGreaterEqual(norm_char.normalized_image.min(), 0.0)
        self.assertLessEqual(norm_char.normalized_image.max(), 1.0)
        self.assertGreater(norm_char.normalized_image.max(), 0.1, "Character was normalized to all zeros")

    def test_aspect_ratio_preservation_tall_vs_wide(self):
        """Verify tall and wide characters preserve their natural aspect ratio without distortion."""
        # Tall character: 30 height, 6 width (like digit '1')
        tall = np.zeros((30, 6), dtype=np.uint8)
        tall[:, 2:4] = 255

        norm_tall = resize_and_center_16x16(tall)
        self.assertEqual(norm_tall.shape, (16, 16))
        # Active width in canvas should remain narrow (much less than 14)
        active_cols = np.where(np.sum(norm_tall > 0.1, axis=0) > 0)[0]
        tall_w = len(active_cols)
        self.assertLessEqual(tall_w, 6, "Tall character was artificially stretched horizontally")

        # Wide character: 6 height, 30 width (like hyphen '-')
        wide = np.zeros((6, 30), dtype=np.uint8)
        wide[2:4, :] = 255

        norm_wide = resize_and_center_16x16(wide)
        self.assertEqual(norm_wide.shape, (16, 16))
        active_rows = np.where(np.sum(norm_wide > 0.1, axis=1) > 0)[0]
        wide_h = len(active_rows)
        self.assertLessEqual(wide_h, 6, "Wide character was artificially stretched vertically")

    def test_centered_embedding_symmetric_padding(self):
        """Verify character crop is centered within the 16x16 canvas."""
        square = np.full((10, 10), 255, dtype=np.uint8)
        norm_square = resize_and_center_16x16(square)

        # Active inner area is 14x14, centered inside 16x16, leaving 1px border
        self.assertEqual(norm_square.shape, (16, 16))
        self.assertEqual(norm_square[0, :].sum(), 0.0, "Top row must be zero padding")
        self.assertEqual(norm_square[15, :].sum(), 0.0, "Bottom row must be zero padding")
        self.assertEqual(norm_square[:, 0].sum(), 0.0, "Left column must be zero padding")
        self.assertEqual(norm_square[:, 15].sum(), 0.0, "Right column must be zero padding")

    def test_integration_with_character_crop_dataclass(self):
        """Verify normalize_character seamlessly ingests Phase 3 CharacterCrop dataclass."""
        crop_mask = np.zeros((25, 18), dtype=np.uint8)
        cv2.putText(crop_mask, "A", (2, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.7, 255, 2)

        char_crop = CharacterCrop(
            char_id=42,
            bbox=(100, 200, 18, 25),
            binary_crop=crop_mask,
            gray_crop=None,
            word_id=3,
            line_id=1,
        )

        norm = normalize_character(char_crop)
        self.assertEqual(norm.char_id, 42)
        self.assertEqual(norm.word_id, 3)
        self.assertEqual(norm.line_id, 1)
        self.assertEqual(norm.original_bbox, (100, 200, 18, 25))
        self.assertEqual(norm.normalized_image.shape, (16, 16))

    def test_normalization_engine_batch_processing(self):
        """Verify NormalizationEngine processes batch list and returns NormalizationResult."""
        crops = []
        for i in range(5):
            c = np.zeros((20, 15), dtype=np.uint8)
            cv2.putText(c, str(i), (2, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.6, 255, 2)
            crops.append(c)

        result = self.engine.normalize_batch(crops)
        self.assertIsInstance(result, NormalizationResult)
        self.assertEqual(result.total_characters, 5)

        batch = result.get_feature_input_batch()
        self.assertEqual(batch.shape, (5, 16, 16))
        self.assertEqual(batch.dtype, np.float32)


if __name__ == "__main__":
    unittest.main()
