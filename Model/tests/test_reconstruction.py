"""
test_reconstruction.py - Unit Test Suite for Phase 7 (Reconstruction & Visualization)

Tests:
1. Spatial text assembly with mock token predictions.
2. Space insertion between words and newline insertion between lines.
3. Paragraph gap detection with double newlines ('\\n\\n').
4. Lossless UTF-8 plain text file export and byte integrity.
5. Bounding box enclosing geometry calculations.
6. Inverse homography bounding box projection.
7. Visual annotated document rendering (HUD overlay and telemetry banner).
8. ReconstructionEngine and VisualizationEngine object-oriented interfaces.
9. End-to-end run_pipeline execution on sample camera capture.
"""

import os
from pathlib import Path
import tempfile
import unittest
import cv2
import numpy as np

# Ensure Model directory is in sys.path
import sys
MODEL_DIR = Path(__file__).resolve().parent.parent
if str(MODEL_DIR) not in sys.path:
    sys.path.insert(0, str(MODEL_DIR))

import config
from predict import PredictionResult
from reconstruction import (
    compute_enclosing_bbox,
    reconstruct_text,
    export_text_file,
    reconstruct_and_export,
    ReconstructionResult,
    ReconstructionEngine,
)
from visualization import (
    render_annotated_document,
    render_side_by_side_comparison,
    transform_bbox_inverse,
    VisualizationEngine,
)
from main import run_pipeline, PipelineResult


class TestPhase7Reconstruction(unittest.TestCase):
    """Automated test cases for Phase 7 Reconstruction & Visual Export."""

    def setUp(self):
        """Build mock character predictions with known coordinates and tokens."""
        # Line 0: "HELLO WORLD"
        # Word 0: "HELLO" (x: 10, 25, 40, 55, 70; y: 20)
        # Word 1: "WORLD" (x: 110, 125, 140, 155, 170; y: 20)
        self.mock_line0 = [
            PredictionResult(predicted_char="H", confidence=0.98, class_index=17, probabilities=np.zeros(47), bbox=(10, 20, 12, 20), char_id=0, word_id=0, line_id=0),
            PredictionResult(predicted_char="E", confidence=0.95, class_index=14, probabilities=np.zeros(47), bbox=(25, 20, 12, 20), char_id=1, word_id=0, line_id=0),
            PredictionResult(predicted_char="L", confidence=0.92, class_index=21, probabilities=np.zeros(47), bbox=(40, 20, 12, 20), char_id=2, word_id=0, line_id=0),
            PredictionResult(predicted_char="L", confidence=0.94, class_index=21, probabilities=np.zeros(47), bbox=(55, 20, 12, 20), char_id=3, word_id=0, line_id=0),
            PredictionResult(predicted_char="O", confidence=0.99, class_index=24, probabilities=np.zeros(47), bbox=(70, 20, 12, 20), char_id=4, word_id=0, line_id=0),
            PredictionResult(predicted_char="W", confidence=0.91, class_index=32, probabilities=np.zeros(47), bbox=(110, 20, 12, 20), char_id=5, word_id=1, line_id=0),
            PredictionResult(predicted_char="O", confidence=0.97, class_index=24, probabilities=np.zeros(47), bbox=(125, 20, 12, 20), char_id=6, word_id=1, line_id=0),
            PredictionResult(predicted_char="R", confidence=0.93, class_index=27, probabilities=np.zeros(47), bbox=(140, 20, 12, 20), char_id=7, word_id=1, line_id=0),
            PredictionResult(predicted_char="L", confidence=0.96, class_index=21, probabilities=np.zeros(47), bbox=(155, 20, 12, 20), char_id=8, word_id=1, line_id=0),
            PredictionResult(predicted_char="D", confidence=0.90, class_index=13, probabilities=np.zeros(47), bbox=(170, 20, 12, 20), char_id=9, word_id=1, line_id=0),
        ]

        # Line 1 (standard line gap): "OCR 2026" (y: 50, gap = 50 - 40 = 10 px < 1.8 * 20)
        self.mock_line1 = [
            PredictionResult(predicted_char="O", confidence=0.98, class_index=24, probabilities=np.zeros(47), bbox=(10, 50, 12, 20), char_id=10, word_id=2, line_id=1),
            PredictionResult(predicted_char="C", confidence=0.96, class_index=12, probabilities=np.zeros(47), bbox=(25, 50, 12, 20), char_id=11, word_id=2, line_id=1),
            PredictionResult(predicted_char="R", confidence=0.95, class_index=27, probabilities=np.zeros(47), bbox=(40, 50, 12, 20), char_id=12, word_id=2, line_id=1),
            PredictionResult(predicted_char="2", confidence=0.99, class_index=2, probabilities=np.zeros(47), bbox=(70, 50, 12, 20), char_id=13, word_id=3, line_id=1),
            PredictionResult(predicted_char="0", confidence=0.99, class_index=0, probabilities=np.zeros(47), bbox=(85, 50, 12, 20), char_id=14, word_id=3, line_id=1),
            PredictionResult(predicted_char="2", confidence=0.98, class_index=2, probabilities=np.zeros(47), bbox=(100, 50, 12, 20), char_id=15, word_id=3, line_id=1),
            PredictionResult(predicted_char="6", confidence=0.97, class_index=6, probabilities=np.zeros(47), bbox=(115, 50, 12, 20), char_id=16, word_id=3, line_id=1),
        ]

        # Line 2 (large paragraph gap): "PARAGRAPH TWO" (y: 120, gap = 120 - 70 = 50 px > 1.8 * 20)
        self.mock_line2 = [
            PredictionResult(predicted_char="P", confidence=0.95, class_index=25, probabilities=np.zeros(47), bbox=(10, 120, 12, 20), char_id=17, word_id=4, line_id=2),
            PredictionResult(predicted_char="A", confidence=0.94, class_index=10, probabilities=np.zeros(47), bbox=(25, 120, 12, 20), char_id=18, word_id=4, line_id=2),
            PredictionResult(predicted_char="S", confidence=0.92, class_index=28, probabilities=np.zeros(47), bbox=(40, 120, 12, 20), char_id=19, word_id=4, line_id=2),
            PredictionResult(predicted_char="S", confidence=0.93, class_index=28, probabilities=np.zeros(47), bbox=(55, 120, 12, 20), char_id=20, word_id=4, line_id=2),
        ]

        self.all_mock_predictions = self.mock_line0 + self.mock_line1 + self.mock_line2

    def test_text_assembly_words_and_spaces(self):
        """Verify words are concatenated correctly and spaced with single space."""
        res = reconstruct_text(self.mock_line0)
        self.assertEqual(res.total_lines, 1)
        self.assertEqual(res.total_words, 2)
        self.assertEqual(res.total_characters, 10)
        self.assertEqual(res.lines[0].text, "HELLO WORLD")
        self.assertEqual(res.full_text, "HELLO WORLD")

    def test_line_and_paragraph_gap_formatting(self):
        """Verify single newline between standard lines and double newline for paragraph gaps."""
        res = reconstruct_text(self.all_mock_predictions, paragraph_gap_factor=1.8)
        self.assertEqual(res.total_lines, 3)
        self.assertEqual(res.total_words, 5)

        # Expected formatting:
        # Line 0: HELLO WORLD
        # Line 1: OCR 2026
        # Paragraph gap (gap=50px > 1.8*20=36px) -> \n\n
        # Line 2: PASS
        expected_text = "HELLO WORLD\nOCR 2026\n\nPASS"
        self.assertEqual(res.full_text, expected_text)

    def test_compute_enclosing_bbox(self):
        """Verify computation of minimal enclosing bounding box."""
        boxes = [(10, 20, 15, 20), (30, 22, 10, 18), (50, 15, 20, 30)]
        enclosing = compute_enclosing_bbox(boxes)
        # min_x = 10, max_x = 50 + 20 = 70 -> width = 60
        # min_y = 15, max_y = 15 + 30 = 45 -> height = 30
        self.assertEqual(enclosing, (10, 15, 60, 30))

        # None / empty input
        self.assertIsNone(compute_enclosing_bbox([]))

    def test_utf8_plaintext_file_export(self):
        """Verify UTF-8 plaintext export and exact byte restoration."""
        text_content = "Antigravity OCR 2026\nLine 2: Classical Computer Vision\nParagraph break:\n\nSuccess!"
        with tempfile.TemporaryDirectory() as tmp_dir:
            out_file = Path(tmp_dir) / "test_export.txt"
            saved_path = export_text_file(text_content, out_file)

            self.assertTrue(saved_path.exists())
            with open(saved_path, "r", encoding="utf-8") as f:
                read_back = f.read()

            self.assertEqual(read_back, text_content)

    def test_reconstruct_and_export_convenience_function(self):
        """Verify high-level reconstruct_and_export function."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            out_file = Path(tmp_dir) / "doc_recognized.txt"
            res = reconstruct_and_export(self.mock_line0, output_path=out_file)

            self.assertIsInstance(res, ReconstructionResult)
            self.assertEqual(res.output_file_path, out_file.resolve())
            self.assertTrue(out_file.exists())
            self.assertGreater(res.mean_confidence, 0.90)

    def test_inverse_homography_coordinate_transformation(self):
        """Verify transform_bbox_inverse maps bounding box corners accurately."""
        # Create a simple 2x scaling homography
        H = np.array([
            [2.0, 0.0, 10.0],
            [0.0, 2.0, 20.0],
            [0.0, 0.0, 1.0],
        ], dtype=np.float64)
        H_inv = np.linalg.inv(H)

        # In flattened space: box at (30, 40, 20, 20)
        # In original space: x = (30 - 10) / 2 = 10, y = (40 - 20) / 2 = 10, w = 10, h = 10
        orig_box = transform_bbox_inverse((30, 40, 20, 20), H_inv)
        self.assertEqual(orig_box, (10, 10, 10, 10))

    def test_render_annotated_document_output(self):
        """Verify annotated document visual rendering dimensions and channels."""
        canvas = np.full((300, 400, 3), 240, dtype=np.uint8)
        with tempfile.TemporaryDirectory() as tmp_dir:
            out_img_path = Path(tmp_dir) / "annotated.png"
            annotated = render_annotated_document(
                image=canvas,
                predictions=self.mock_line0,
                output_path=out_img_path,
                show_confidences=True,
                draw_words=True,
                draw_chars=True,
                draw_header_hud=True,
            )

            self.assertEqual(annotated.shape, (300, 400, 3))
            self.assertEqual(annotated.dtype, np.uint8)
            self.assertTrue(out_img_path.exists())

            # Verify that drawing actually occurred (canvas pixels modified)
            self.assertFalse(np.array_equal(canvas, annotated))

    def test_render_side_by_side_comparison(self):
        """Verify side-by-side comparison image dimensions."""
        img1 = np.full((200, 300, 3), 100, dtype=np.uint8)
        img2 = np.full((200, 300, 3), 200, dtype=np.uint8)
        comparison = render_side_by_side_comparison(img1, img2)

        self.assertEqual(comparison.shape, (200, 600, 3))

    def test_reconstruction_and_visualization_engine_classes(self):
        """Verify object-oriented APIs."""
        rec_engine = ReconstructionEngine(paragraph_gap_factor=1.8)
        rec_res = rec_engine.reconstruct(self.mock_line0)
        self.assertEqual(rec_res.full_text, "HELLO WORLD")

        vis_engine = VisualizationEngine()
        canvas = np.full((200, 300, 3), 255, dtype=np.uint8)
        annotated = vis_engine.annotate(canvas, self.mock_line0)
        self.assertEqual(annotated.shape, (200, 300, 3))

    def test_end_to_end_pipeline_execution(self):
        """Verify full end-to-end pipeline execution from sample image to output files."""
        sample_path = config.SAMPLES_DIR / "IMG_20261021_212517.jpg"
        if not sample_path.exists():
            sample_path = config.SAMPLES_DIR / "sample_camera_capture_synth.png"

        with tempfile.TemporaryDirectory() as tmp_dir:
            res = run_pipeline(
                input_path=sample_path,
                output_dir=tmp_dir,
                quiet=True,
            )

            self.assertIsInstance(res, PipelineResult)
            self.assertTrue(res.output_text_path.exists())
            self.assertTrue(res.output_annotated_path.exists())
            self.assertGreater(res.reconstruction.total_characters, 0)
            self.assertGreater(res.reconstruction.mean_confidence, 0.0)
            self.assertGreater(res.runtime_seconds, 0.0)


if __name__ == "__main__":
    unittest.main()
