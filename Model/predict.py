"""
predict.py - Phase 6: Shallow MLP Inference Engine, Confidence Scoring & Label Decoding

This module provides the runtime inference pipeline that ingests 256-dimensional
feature vectors (or normalized character crops), evaluates class probabilities via
the shallow Multi-Layer Perceptron (256 -> 100 -> 90 -> N Softmax), and decodes
predictions into ASCII characters with calibrated confidence scores.

Features:
1. Model & Scaler Artifact Management (joblib deserialization).
2. ASCII Label Decoding from models/label_mapping.json.
3. Prediction Confidence Scoring (Argmax probability score in [0.0, 1.0]).
4. Top-K Prediction Ranking (for diagnostic disambiguation).
5. Transparent Pure NumPy Forward Propagation (demonstrating zero external runtime dependencies).
6. High-throughput Batch Inference API.
"""

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
import joblib
import numpy as np

try:
    from . import config
    from .features import extract_features, extract_features_batch
    from .normalization import NormalizedCharacter, NormalizationResult
except ImportError:
    import config
    from features import extract_features, extract_features_batch
    from normalization import NormalizedCharacter, NormalizationResult


@dataclass
class PredictionResult:
    """Represents the classification result for an individual character."""
    predicted_char: str                      # ASCII decoded character (e.g. 'A', '5', 't')
    confidence: float                         # Maximum softmax probability score in [0.0, 1.0]
    class_index: int                          # Argmax integer index (0..N-1)
    probabilities: np.ndarray                 # Softmax probability distribution over all classes
    top_k: List[Tuple[str, float]] = field(default_factory=list)  # Top-k candidate tuples (char, prob)
    bbox: Optional[Tuple[int, int, int, int]] = None  # Global bounding box (x, y, w, h)
    char_id: int = 0
    word_id: int = 0
    line_id: int = 0

    def to_dict(self) -> Dict[str, Any]:
        """Convert prediction result to standard dictionary."""
        return {
            "predicted_char": self.predicted_char,
            "confidence": round(float(self.confidence), 4),
            "class_index": int(self.class_index),
            "top_k": [(c, round(float(p), 4)) for c, p in self.top_k],
            "bbox": list(self.bbox) if self.bbox is not None else None,
            "char_id": self.char_id,
            "word_id": self.word_id,
            "line_id": self.line_id,
        }


# -------------------------------------------------------------------------
# Artifact Loaders & Caching
# -------------------------------------------------------------------------

_CACHED_MODEL = None
_CACHED_SCALER = None
_CACHED_LABEL_MAP = None


def load_model(model_path: Optional[Union[str, Path]] = None) -> Any:
    """Loads serialized shallow MLPClassifier artifact."""
    global _CACHED_MODEL
    path = Path(model_path or config.MODEL_FILE)
    if not path.exists():
        raise FileNotFoundError(
            f"[Phase 6 Error] Model file not found at: '{path}'. "
            f"Run 'python train_mlp.py' first to train and serialize model artifacts."
        )
    _CACHED_MODEL = joblib.load(str(path))
    return _CACHED_MODEL


def load_scaler(scaler_path: Optional[Union[str, Path]] = None) -> Any:
    """Loads serialized StandardScaler artifact."""
    global _CACHED_SCALER
    path = Path(scaler_path or config.SCALER_FILE)
    if not path.exists():
        raise FileNotFoundError(
            f"[Phase 6 Error] Scaler file not found at: '{path}'. "
            f"Run 'python train_mlp.py' first."
        )
    _CACHED_SCALER = joblib.load(str(path))
    return _CACHED_SCALER


def load_label_mapping(label_map_path: Optional[Union[str, Path]] = None) -> Dict[int, str]:
    """Loads label mapping dictionary (int class index -> ASCII character)."""
    global _CACHED_LABEL_MAP
    path = Path(label_map_path or config.LABEL_MAP_FILE)
    if not path.exists():
        raise FileNotFoundError(
            f"[Phase 6 Error] Label mapping file not found at: '{path}'. "
            f"Run 'python train_mlp.py' first."
        )
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    _CACHED_LABEL_MAP = {int(k): str(v) for k, v in data.items()}
    return _CACHED_LABEL_MAP


# -------------------------------------------------------------------------
# Transparent Pure NumPy Forward Pass (Zero External Dependencies)
# -------------------------------------------------------------------------

def softmax_numpy(z: np.ndarray) -> np.ndarray:
    """Numerically stable softmax activation over 2D array or 1D vector."""
    if z.ndim == 1:
        shift_z = z - np.max(z)
        exps = np.exp(shift_z)
        return exps / np.sum(exps)
    else:
        shift_z = z - np.max(z, axis=1, keepdims=True)
        exps = np.exp(shift_z)
        return exps / np.sum(exps, axis=1, keepdims=True)


def relu_numpy(z: np.ndarray) -> np.ndarray:
    """Rectified Linear Unit activation: max(0, z)."""
    return np.maximum(0.0, z)


def forward_pass_numpy(
    x_scaled: np.ndarray,
    weights: List[np.ndarray],
    biases: List[np.ndarray],
) -> np.ndarray:
    """
    Computes forward propagation using pure NumPy matrix multiplications:
        a1 = ReLU(x * W1 + b1)
        a2 = ReLU(a1 * W2 + b2)
        logits = a2 * W3 + b3
        probs = Softmax(logits)

    Args:
        x_scaled: Scaled feature matrix of shape (N, 256) or vector (256,).
        weights: List of weight matrices [W1 (256x100), W2 (100x90), W3 (90xN)].
        biases: List of bias vectors [b1 (100,), b2 (90,), b3 (N,)].

    Returns:
        Probability distribution array matching input batch size.
    """
    is_1d = (x_scaled.ndim == 1)
    a = np.atleast_2d(x_scaled)

    # Hidden Layer 1: 256 -> 100
    z1 = np.dot(a, weights[0]) + biases[0]
    a1 = relu_numpy(z1)

    # Hidden Layer 2: 100 -> 90
    z2 = np.dot(a1, weights[1]) + biases[1]
    a2 = relu_numpy(z2)

    # Output Layer: 90 -> N
    logits = np.dot(a2, weights[2]) + biases[2]
    probs = softmax_numpy(logits)

    return probs[0] if is_1d else probs


# -------------------------------------------------------------------------
# Inference Pipeline
# -------------------------------------------------------------------------

def predict_single_vector(
    feature_vector: np.ndarray,
    model: Any,
    scaler: Any,
    label_mapping: Dict[int, str],
    top_k_count: int = 3,
    use_numpy_forward: bool = False,
) -> PredictionResult:
    """
    Performs inference on a single 256-D feature vector.

    Args:
        feature_vector: 1D float32 array of shape (256,).
        model: Trained MLPClassifier.
        scaler: Fitted StandardScaler.
        label_mapping: Dict mapping int class index -> ASCII character.
        top_k_count: Number of top candidate predictions to return.
        use_numpy_forward: Whether to use transparent pure NumPy forward pass.

    Returns:
        PredictionResult dataclass.
    """
    if feature_vector.ndim == 1:
        x_2d = feature_vector.reshape(1, -1)
    else:
        x_2d = feature_vector

    # 1. Standardize features using training scaler
    x_scaled = scaler.transform(x_2d)

    # 2. Forward Propagation -> Softmax class probability distribution
    if use_numpy_forward and hasattr(model, "coefs_"):
        probs = forward_pass_numpy(x_scaled[0], model.coefs_, model.intercepts_)
    else:
        probs = model.predict_proba(x_scaled)[0]

    # 3. Argmax prediction and confidence
    best_idx = int(np.argmax(probs))
    confidence = float(probs[best_idx])
    predicted_char = label_mapping.get(best_idx, "?")

    # 4. Top-K predictions
    top_indices = np.argsort(probs)[::-1][:top_k_count]
    top_k = [(label_mapping.get(int(idx), "?"), float(probs[idx])) for idx in top_indices]

    return PredictionResult(
        predicted_char=predicted_char,
        confidence=confidence,
        class_index=best_idx,
        probabilities=probs,
        top_k=top_k,
    )


def predict_character(
    char_input: Union[np.ndarray, Any],
    model: Optional[Any] = None,
    scaler: Optional[Any] = None,
    label_mapping: Optional[Dict[int, str]] = None,
    top_k_count: int = 3,
    feature_mode: str = config.FEATURE_MODE,
) -> PredictionResult:
    """
    High-level API: Evaluates an individual character crop or feature vector.

    Accepts:
    - 1D 256-D feature vector array.
    - NormalizedCharacter object from Phase 4.
    - 2D 16x16 normalized character array.

    Returns:
        PredictionResult dataclass.
    """
    clf = model or _CACHED_MODEL or load_model()
    scl = scaler or _CACHED_SCALER or load_scaler()
    lmap = label_mapping or _CACHED_LABEL_MAP or load_label_mapping()

    bbox = None
    char_id = 0
    word_id = 0
    line_id = 0

    if hasattr(char_input, "normalized_image"):
        vec = extract_features(char_input, mode=feature_mode)
        bbox = getattr(char_input, "original_bbox", None)
        char_id = getattr(char_input, "char_id", 0)
        word_id = getattr(char_input, "word_id", 0)
        line_id = getattr(char_input, "line_id", 0)
    elif isinstance(char_input, np.ndarray):
        if char_input.ndim == 2 and char_input.shape == config.NORMALIZED_CHAR_SIZE:
            vec = extract_features(char_input, mode=feature_mode)
        elif char_input.ndim == 1 and len(char_input) == config.FEATURE_VECTOR_DIM:
            vec = char_input
        else:
            raise ValueError(f"[Phase 6 Error] Invalid array shape for prediction: {char_input.shape}")
    else:
        raise TypeError(f"[Phase 6 Error] Unsupported input type for prediction: {type(char_input)}")

    res = predict_single_vector(vec, clf, scl, lmap, top_k_count=top_k_count)
    res.bbox = bbox
    res.char_id = char_id
    res.word_id = word_id
    res.line_id = line_id
    return res


def predict_batch(
    char_inputs: Union[List[Any], Any, np.ndarray],
    model: Optional[Any] = None,
    scaler: Optional[Any] = None,
    label_mapping: Optional[Dict[int, str]] = None,
    top_k_count: int = 3,
    feature_mode: str = config.FEATURE_MODE,
) -> List[PredictionResult]:
    """
    High-throughput batch inference across a list of characters or feature vectors.

    Args:
        char_inputs: List of NormalizedCharacter objects, NormalizationResult, or (N, 256) feature matrix.

    Returns:
        List of PredictionResult dataclass instances.
    """
    clf = model or _CACHED_MODEL or load_model()
    scl = scaler or _CACHED_SCALER or load_scaler()
    lmap = label_mapping or _CACHED_LABEL_MAP or load_label_mapping()

    # Extract items and feature matrix
    if hasattr(char_inputs, "characters"):
        items = char_inputs.characters
        X = extract_features_batch(items, mode=feature_mode)
    elif isinstance(char_inputs, np.ndarray) and char_inputs.ndim == 2 and char_inputs.shape[1] == config.FEATURE_VECTOR_DIM:
        items = None
        X = char_inputs
    elif isinstance(char_inputs, list):
        items = char_inputs
        if len(items) > 0 and isinstance(items[0], np.ndarray) and items[0].ndim == 1 and items[0].shape[0] == config.FEATURE_VECTOR_DIM:
            X = np.stack(items, axis=0).astype(np.float32)
        else:
            X = extract_features_batch(items, mode=feature_mode)
    else:
        items = [char_inputs]
        if isinstance(char_inputs, np.ndarray) and char_inputs.ndim == 1 and char_inputs.shape[0] == config.FEATURE_VECTOR_DIM:
            X = np.expand_dims(char_inputs, axis=0).astype(np.float32)
        else:
            X = extract_features_batch(items, mode=feature_mode)

    if X.shape[0] == 0:
        return []

    # Batch standardization & Softmax probability forward pass
    X_scaled = scl.transform(X)
    probs_matrix = clf.predict_proba(X_scaled)

    results: List[PredictionResult] = []
    for idx, probs in enumerate(probs_matrix):
        best_idx = int(np.argmax(probs))
        confidence = float(probs[best_idx])
        predicted_char = lmap.get(best_idx, "?")

        top_indices = np.argsort(probs)[::-1][:top_k_count]
        top_k = [(lmap.get(int(i), "?"), float(probs[i])) for i in top_indices]

        bbox = None
        char_id = idx
        word_id = 0
        line_id = 0
        if items is not None and idx < len(items):
            item = items[idx]
            bbox = getattr(item, "original_bbox", None)
            char_id = getattr(item, "char_id", idx)
            word_id = getattr(item, "word_id", 0)
            line_id = getattr(item, "line_id", 0)

        results.append(
            PredictionResult(
                predicted_char=predicted_char,
                confidence=confidence,
                class_index=best_idx,
                probabilities=probs,
                top_k=top_k,
                bbox=bbox,
                char_id=char_id,
                word_id=word_id,
                line_id=line_id,
            )
        )

    return results


class MLPInferenceEngine:
    """
    Object-oriented encapsulation of Phase 6 Inference for clean modular execution.
    """

    def __init__(
        self,
        model_path: Optional[Union[str, Path]] = None,
        scaler_path: Optional[Union[str, Path]] = None,
        label_map_path: Optional[Union[str, Path]] = None,
        feature_mode: str = config.FEATURE_MODE,
        top_k_count: int = 3,
    ):
        self.model = load_model(model_path)
        self.scaler = load_scaler(scaler_path)
        self.label_mapping = load_label_mapping(label_map_path)
        self.feature_mode = feature_mode
        self.top_k_count = top_k_count

    def predict(self, char_input: Union[np.ndarray, Any]) -> PredictionResult:
        """Inference on a single character or feature vector."""
        return predict_character(
            char_input=char_input,
            model=self.model,
            scaler=self.scaler,
            label_mapping=self.label_mapping,
            top_k_count=self.top_k_count,
            feature_mode=self.feature_mode,
        )

    def predict_batch(self, char_inputs: Union[List[Any], Any, np.ndarray]) -> List[PredictionResult]:
        """Batch inference on multiple characters or feature matrix."""
        return predict_batch(
            char_inputs=char_inputs,
            model=self.model,
            scaler=self.scaler,
            label_mapping=self.label_mapping,
            top_k_count=self.top_k_count,
            feature_mode=self.feature_mode,
        )
