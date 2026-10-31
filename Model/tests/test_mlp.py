"""
test_mlp.py - Automated Unit Tests for Phase 6: Shallow MLP Classifier

Verifies:
1. Model loading from models/emnist_mlp_model.joblib.
2. Scaler loading from models/feature_scaler.joblib.
3. Label mapping loading from models/label_mapping.json (47 alphanumeric classes).
4. Inference on synthetic 256-D vectors.
5. Class index maps to valid ASCII character.
6. Confidence score is bounded in [0.0, 1.0].
7. Pure NumPy forward pass matches scikit-learn Softmax probabilities.
8. Top-K predictions are sorted in descending confidence order.
9. Batch inference produces correct output dimensions and types.
10. OOP MLPInferenceEngine execution.
"""

import json
import sys
import unittest
from pathlib import Path
import numpy as np

# Ensure Model package is discoverable on python path
TEST_DIR = Path(__file__).resolve().parent
MODEL_DIR = TEST_DIR.parent
if str(MODEL_DIR) not in sys.path:
    sys.path.insert(0, str(MODEL_DIR))

from predict import (
    MLPInferenceEngine,
    PredictionResult,
    forward_pass_numpy,
    load_label_mapping,
    load_model,
    load_scaler,
    predict_batch,
    predict_character,
    predict_single_vector,
)
import config


class TestPhase6MLP(unittest.TestCase):
    """Test suite covering Phase 6 Shallow MLP Classifier specifications."""

    def setUp(self):
        self.model = load_model()
        self.scaler = load_scaler()
        self.label_mapping = load_label_mapping()
        self.engine = MLPInferenceEngine()

    def test_model_artifact_loaded_and_architecture(self):
        """Verify model weights are loaded and network topology matches 256 -> 100 -> 90 -> N."""
        self.assertIsNotNone(self.model)
        self.assertEqual(len(self.model.coefs_), 3, "MLP must have exactly 3 weight matrices (2 hidden layers)")
        # Check layer shapes
        w1, w2, w3 = self.model.coefs_
        self.assertEqual(w1.shape[0], 256, "Input layer must accept 256 features")
        self.assertEqual(w1.shape[1], 100, "Hidden layer 1 must have 100 neurons")
        self.assertEqual(w2.shape, (100, 90), "Hidden layer 2 must have 90 neurons")
        self.assertEqual(w3.shape[0], 90)
        self.assertIn(w3.shape[1], (10, 47), "Output layer must match class count (10 or 47)")

    def test_scaler_artifact_loaded_and_dimensions(self):
        """Verify feature scaler matches 256-dimensional feature space."""
        self.assertIsNotNone(self.scaler)
        self.assertEqual(len(self.scaler.mean_), 256)
        self.assertEqual(len(self.scaler.scale_), 256)

    def test_label_mapping_integrity(self):
        """Verify label mapping covers all classes and maps to valid non-empty ASCII characters."""
        self.assertIsNotNone(self.label_mapping)
        self.assertGreaterEqual(len(self.label_mapping), 10)
        for idx, char_str in self.label_mapping.items():
            self.assertIsInstance(idx, int)
            self.assertIsInstance(char_str, str)
            self.assertEqual(len(char_str), 1)

    def test_inference_synthetic_vector_output_ranges(self):
        """Verify inference on synthetic 256-D vector returns bounded confidence in [0.0, 1.0]."""
        dummy_feat = np.random.uniform(0.0, 1.0, 256).astype(np.float32)
        result = predict_character(dummy_feat)

        self.assertIsInstance(result, PredictionResult)
        self.assertIsInstance(result.predicted_char, str)
        self.assertGreaterEqual(result.confidence, 0.0)
        self.assertLessEqual(result.confidence, 1.0)
        self.assertIn(result.class_index, range(len(self.label_mapping)))

        # Probabilities sum to ~1.0
        prob_sum = float(np.sum(result.probabilities))
        self.assertAlmostEqual(prob_sum, 1.0, delta=1e-4)

    def test_pure_numpy_forward_pass_matches_sklearn(self):
        """Verify transparent pure NumPy forward pass produces identical Softmax distribution."""
        dummy_feat = np.random.uniform(0.0, 1.0, (5, 256)).astype(np.float32)
        x_scaled = self.scaler.transform(dummy_feat)

        # Sklearn forward pass
        probs_sklearn = self.model.predict_proba(x_scaled)

        # Pure NumPy forward pass
        probs_numpy = forward_pass_numpy(x_scaled, self.model.coefs_, self.model.intercepts_)

        np.testing.assert_allclose(
            probs_numpy,
            probs_sklearn,
            atol=1e-5,
            err_msg="Pure NumPy forward pass output diverged from scikit-learn",
        )

    def test_top_k_predictions_ranking(self):
        """Verify top-k predictions are ordered in descending probability order."""
        dummy_feat = np.random.uniform(0.0, 1.0, 256).astype(np.float32)
        result = predict_character(dummy_feat, top_k_count=5)

        self.assertEqual(len(result.top_k), 5)
        prev_prob = 1.0
        for char, prob in result.top_k:
            self.assertLessEqual(prob, prev_prob + 1e-6, "Top-K probabilities must be monotonically decreasing")
            prev_prob = prob

    def test_batch_prediction_output(self):
        """Verify batch prediction returns list of PredictionResult instances of correct length."""
        batch_feats = np.random.uniform(0.0, 1.0, (8, 256)).astype(np.float32)
        results = predict_batch(batch_feats)

        self.assertEqual(len(results), 8)
        for r in results:
            self.assertIsInstance(r, PredictionResult)
            self.assertIsInstance(r.predicted_char, str)
            self.assertGreaterEqual(r.confidence, 0.0)
            self.assertLessEqual(r.confidence, 1.0)

    def test_mlp_inference_engine_class(self):
        """Verify MLPInferenceEngine object-oriented API."""
        dummy_feat = np.random.uniform(0.0, 1.0, 256).astype(np.float32)
        res = self.engine.predict(dummy_feat)
        self.assertIsInstance(res, PredictionResult)

        batch_res = self.engine.predict_batch([dummy_feat, dummy_feat])
        self.assertEqual(len(batch_res), 2)


if __name__ == "__main__":
    unittest.main()
