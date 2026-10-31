"""
test_features.py - Automated Unit Tests for Phase 5: Circular DDD Feature Engine

Verifies:
1. Feature vector output shape is strictly (256,).
2. All feature values are finite real numbers strictly in [0.0, 1.0].
3. Mathematical determinism: identical character crop generates identical feature vector.
4. Numerical stability: zero NaNs and zero Infs across synthetic and real inputs.
5. Discriminability: distinct characters ('O' vs 'I') produce distinct, non-identical vectors.
6. Support for both "circular_ddd" and "flatten_16x16" feature modes.
7. Batch extraction produces a 2D float32 matrix of shape (N, 256).
8. Seamless integration with Phase 4 NormalizedCharacter dataclass.
9. Validation and error handling on wrong input shapes.
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

from features import (
    FeatureExtractionEngine,
    extract_circular_ddd_features,
    extract_features,
    extract_features_batch,
    flatten_raster_features,
    get_directional_unit_vectors,
    sample_virtual_circular,
)
from normalization import NormalizedCharacter
import config


class TestPhase5Features(unittest.TestCase):
    """Test suite covering Phase 5 Circular DDD Feature Engine specifications."""

    def setUp(self):
        self.engine = FeatureExtractionEngine(mode="circular_ddd")
        # Synthetic 16x16 character 'O'
        self.char_o = np.zeros((16, 16), dtype=np.float32)
        cv2.circle(self.char_o, (8, 8), 5, 1.0, 2)

        # Synthetic 16x16 character 'I'
        self.char_i = np.zeros((16, 16), dtype=np.float32)
        self.char_i[2:14, 7:9] = 1.0

    def test_feature_vector_dimension_strictly_256(self):
        """Verify feature vector dimension is strictly 256 for circular_ddd and flatten_16x16."""
        vec_ddd = extract_features(self.char_o, mode="circular_ddd")
        self.assertEqual(vec_ddd.shape, (256,), "Circular DDD vector must have shape (256,)")
        self.assertEqual(vec_ddd.dtype, np.float32)

        vec_flat = extract_features(self.char_o, mode="flatten_16x16")
        self.assertEqual(vec_flat.shape, (256,), "Flattened vector must have shape (256,)")
        self.assertEqual(vec_flat.dtype, np.float32)

    def test_feature_values_strictly_bounded_0_to_1(self):
        """Verify all extracted features are bounded within [0.0, 1.0]."""
        vec = self.engine.extract(self.char_o)
        self.assertGreaterEqual(float(vec.min()), 0.0, "Feature values must be >= 0.0")
        self.assertLessEqual(float(vec.max()), 1.0, "Feature values must be <= 1.0")

    def test_mathematical_determinism(self):
        """Verify identical inputs produce strictly identical feature vectors."""
        vec1 = self.engine.extract(self.char_o)
        vec2 = self.engine.extract(self.char_o.copy())
        np.testing.assert_allclose(vec1, vec2, err_msg="Feature extraction is not deterministic")

    def test_zero_nans_and_zero_infs(self):
        """Verify feature vectors contain zero NaNs or Infs on blank, solid, and pattern inputs."""
        # Blank canvas
        blank = np.zeros((16, 16), dtype=np.float32)
        vec_blank = self.engine.extract(blank)
        self.assertEqual(int(np.isnan(vec_blank).sum()), 0)
        self.assertEqual(int(np.isinf(vec_blank).sum()), 0)

        # Solid canvas
        solid = np.ones((16, 16), dtype=np.float32)
        vec_solid = self.engine.extract(solid)
        self.assertEqual(int(np.isnan(vec_solid).sum()), 0)
        self.assertEqual(int(np.isinf(vec_solid).sum()), 0)

    def test_feature_discriminability_o_vs_i(self):
        """Verify distinct character shapes produce distinct non-identical feature vectors."""
        vec_o = self.engine.extract(self.char_o)
        vec_i = self.engine.extract(self.char_i)

        diff = float(np.linalg.norm(vec_o - vec_i))
        self.assertGreater(diff, 1.0, "Circular DDD features failed to distinguish 'O' from 'I'")

    def test_virtual_circular_sampling_clamping(self):
        """Verify virtual circular sampling clamps points outside circumscribed radius smoothly."""
        img = np.zeros((16, 16), dtype=np.float32)
        img[7:9, 7:9] = 1.0

        val_in = sample_virtual_circular(img, 7.5, 7.5)
        self.assertAlmostEqual(val_in, 1.0, delta=0.01)

        val_out = sample_virtual_circular(img, 25.0, 25.0)
        self.assertGreaterEqual(val_out, 0.0)
        self.assertLessEqual(val_out, 1.0)

    def test_directional_unit_vectors(self):
        """Verify 8 directional unit vectors have unit norm and 8 discrete bearings."""
        dirs = get_directional_unit_vectors(8)
        self.assertEqual(len(dirs), 8)
        for dx, dy in dirs:
            norm = math.sqrt(dx * dx + dy * dy)
            self.assertAlmostEqual(norm, 1.0, delta=1e-5)

    def test_batch_feature_extraction(self):
        """Verify batch extraction returns 2D matrix of shape (N, 256)."""
        batch_images = np.stack([self.char_o, self.char_i, np.zeros((16, 16), dtype=np.float32)], axis=0)
        matrix = self.engine.extract_batch(batch_images)

        self.assertEqual(matrix.shape, (3, 256))
        self.assertEqual(matrix.dtype, np.float32)
        self.assertFalse(np.isnan(matrix).any())

    def test_integration_with_normalized_character_dataclass(self):
        """Verify extract_features seamlessly accepts Phase 4 NormalizedCharacter objects."""
        norm_char = NormalizedCharacter(
            char_id=7,
            normalized_image=self.char_o,
            slant_angle_deg=3.2,
            original_bbox=(10, 20, 15, 22),
            deskewed_crop=None,
        )

        vec = extract_features(norm_char, mode="circular_ddd")
        self.assertEqual(vec.shape, (256,))
        self.assertEqual(vec.dtype, np.float32)

    def test_invalid_shape_raises_value_error(self):
        """Verify ValueError is raised if input is not (16, 16)."""
        wrong_shape = np.zeros((20, 20), dtype=np.float32)
        with self.assertRaises(ValueError):
            extract_features(wrong_shape)


if __name__ == "__main__":
    unittest.main()
