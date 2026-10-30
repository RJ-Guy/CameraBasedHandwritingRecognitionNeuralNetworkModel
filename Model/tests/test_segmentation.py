"""
test_segmentation.py - Automated Unit Tests for Phase 3: Hierarchical Segmentation

Verifies:
1. Synthetic multi-line document segmentation.
2. Strict top-to-bottom line ordering (y1 < y2).
3. Strict left-to-right word and character ordering (x1 < x2).
4. Noise speckle filtering (< 15 px^2 dust suppression).
5. Scratch and horizontal ruling line rejection (extreme aspect ratio filtering).
6. Diacritic and dot association (unifying dot of 'i' with main stem).
7. Ligature splitting for wide joined characters.
8. Diagnostic debug overlay image rendering (proper dimensions, Blue/Green/Red boxes).
9. Robust error handling on empty or corrupted binary masks.
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

from segmentation import (
    CharacterCrop,
    Line,
    SegmentationEngine,
    SegmentationResult,
    Word,
    compute_horizontal_projection,
    detect_lines,
    detect_words,
    extract_characters,
    merge_diacritics,
    render_segmentation_overlay,
    segment_document,
    split_wide_ligature,
)
import config


def create_synthetic_multiline_doc(
    lines_text: list = None,
    width: int = 600,
    height: int = 300,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Creates a synthetic binary mask and grayscale canvas with multiple lines
    and distinct words for deterministic unit testing.
    """
    if lines_text is None:
        lines_text = ["HELLO WORLD", "CV OCR 2026", "TEST PIPELINE"]

    gray = np.full((height, width), 245, dtype=np.uint8)
    binary = np.zeros((height, width), dtype=np.uint8)

    y_positions = [60, 140, 220]

    for line_idx, text in enumerate(lines_text):
        y = y_positions[line_idx]
        # Draw on grayscale
        cv2.putText(gray, text, (50, y), cv2.FONT_HERSHEY_SIMPLEX, 1.0, 20, 2, cv2.LINE_AA)
        # Draw on binary mask (255 = ink)
        cv2.putText(binary, text, (50, y), cv2.FONT_HERSHEY_SIMPLEX, 1.0, 255, 2, cv2.LINE_AA)

    return binary, gray


class TestPhase3Segmentation(unittest.TestCase):
    """Test suite covering Phase 3 Segmentation specifications."""

    def setUp(self):
        self.engine = SegmentationEngine()

    def test_synthetic_multiline_segmentation_and_counts(self):
        """Verify segmentation on multi-line text extracts expected line and word counts."""
        binary, gray = create_synthetic_multiline_doc()
        result = self.engine.segment(binary, gray)

        self.assertIsInstance(result, SegmentationResult)
        self.assertEqual(result.total_lines, 3, f"Expected 3 lines, got {result.total_lines}")
        self.assertGreaterEqual(result.total_words, 5, "Expected at least 5 words across 3 lines")
        self.assertGreaterEqual(result.total_characters, 15, "Expected characters for HELLO WORLD...")

    def test_top_to_bottom_line_ordering(self):
        """Verify detected lines are strictly ordered top-to-bottom (y1 < y2)."""
        binary, gray = create_synthetic_multiline_doc()
        result = self.engine.segment(binary, gray)

        prev_y = -1
        for line in result.lines:
            self.assertGreater(line.y, prev_y, f"Line y={line.y} not greater than previous y={prev_y}")
            prev_y = line.y

    def test_left_to_right_word_and_char_ordering(self):
        """Verify words and characters are strictly ordered left-to-right within lines/words."""
        binary, gray = create_synthetic_multiline_doc()
        result = self.engine.segment(binary, gray)

        for line in result.lines:
            prev_word_x = -1
            for word in line.words:
                self.assertGreater(word.x, prev_word_x, "Words within line must be ordered left-to-right")
                prev_word_x = word.x

                prev_char_x = -1
                for char in word.characters:
                    self.assertGreaterEqual(char.x, prev_char_x, "Characters within word must be ordered left-to-right")
                    prev_char_x = char.x

    def test_noise_speckle_filtering(self):
        """Verify microscopic dust speckles (< 15 px^2) are discarded."""
        binary = np.zeros((100, 200), dtype=np.uint8)
        # Valid character 'T' (area > 30)
        cv2.putText(binary, "T", (50, 60), cv2.FONT_HERSHEY_SIMPLEX, 1.0, 255, 2)

        # Isolated tiny dust speckles (< 15 pixels)
        binary[15:17, 20:22] = 255  # 4 pixels
        binary[80, 150] = 255        # 1 pixel
        binary[30:33, 170:173] = 255 # 9 pixels

        result = self.engine.segment(binary)
        all_chars = result.get_all_characters()

        # Only the valid character should survive
        for char in all_chars:
            ink_count = (char.binary_crop == 255).sum()
            self.assertGreaterEqual(ink_count, config.MIN_CHAR_AREA, "Small speckle was not filtered")

    def test_ruling_line_scratch_filtering(self):
        """Verify extreme aspect ratio components (ruling lines and scratches) are rejected."""
        binary = np.zeros((100, 300), dtype=np.uint8)
        # Valid character 'A'
        cv2.putText(binary, "A", (50, 70), cv2.FONT_HERSHEY_SIMPLEX, 1.0, 255, 2)

        # Horizontal ruling line: 1px high, 120px wide (aspect ratio = 120 > 8.0)
        binary[85, 20:140] = 255

        # Vertical scratch: 70px high, 1px wide (aspect ratio = 0.014 < 1/8)
        binary[10:80, 200] = 255

        result = self.engine.segment(binary)
        all_chars = result.get_all_characters()

        for char in all_chars:
            self.assertLessEqual(char.aspect_ratio, config.MAX_CHAR_ASPECT_RATIO)
            self.assertGreaterEqual(char.aspect_ratio, 1.0 / config.MAX_CHAR_ASPECT_RATIO)

    def test_diacritic_merging_dot_on_i(self):
        """Verify diacritics (like dot on 'i') are merged into the main character stem."""
        # Standalone boxes for dot and stem
        dot_box = (10, 10, 6, 6)
        stem_box = (10, 22, 6, 25)

        merged = merge_diacritics([dot_box, stem_box], max_vertical_gap=15)
        self.assertEqual(len(merged), 1, "Dot and stem should be merged into 1 unified box")
        mx, my, mw, mh = merged[0]
        self.assertEqual(mx, 10)
        self.assertEqual(my, 10)
        self.assertEqual(mw, 6)
        self.assertEqual(mh, 37)

    def test_ligature_splitting_wide_joined_characters(self):
        """Verify wide joined cursive characters are split at the thinned stroke valley."""
        # Create two loops connected by a thin 1px bridge
        joined = np.zeros((24, 48), dtype=np.uint8)
        cv2.circle(joined, (12, 12), 9, 255, 2)
        cv2.circle(joined, (36, 12), 9, 255, 2)
        joined[12, 18:30] = 255  # Thin 1-pixel bridge

        sub_boxes = split_wide_ligature(joined, aspect_ratio_threshold=1.5)
        self.assertEqual(len(sub_boxes), 2, "Wide joined character should be split into 2 sub-boxes")
        self.assertLess(sub_boxes[0][2], 48)
        self.assertGreater(sub_boxes[1][0], 0)

    def test_render_debug_overlay_visual(self):
        """Verify diagnostic visual overlay canvas generation and color coding."""
        binary, gray = create_synthetic_multiline_doc()
        result = self.engine.segment(binary, gray)

        overlay = render_segmentation_overlay(gray, result)
        self.assertEqual(overlay.ndim, 3)
        self.assertEqual(overlay.shape[2], 3)
        self.assertEqual(overlay.shape[:2], binary.shape[:2])

        # Verify colors were drawn (presence of pure blue/green/red in overlay)
        has_blue = (overlay[:, :, 0] == 255).any()
        has_green = (overlay[:, :, 1] == 255).any()
        has_red = (overlay[:, :, 2] == 255).any()
        self.assertTrue(has_blue, "Line bounding box overlay missing")
        self.assertTrue(has_green, "Word bounding box overlay missing")
        self.assertTrue(has_red, "Character bounding box overlay missing")

    def test_empty_image_error_handling(self):
        """Verify empty input binary mask raises descriptive ValueError."""
        with self.assertRaises(ValueError):
            segment_document(np.array([]))
        with self.assertRaises(ValueError):
            segment_document(None)


if __name__ == "__main__":
    unittest.main()
