"""
train_mlp.py - Phase 6: Shallow Multi-Layer Perceptron (MLP) Training & Evaluation

This module trains, evaluates, and serializes a shallow fully connected Multi-Layer
Perceptron (MLP) for handwriting character recognition, strictly conforming to the
zero-deep-learning policy (using scikit-learn MLPClassifier or pure NumPy).

Architecture:
    Input Layer [256] -> Hidden Layer 1 [100, ReLU] -> Hidden Layer 2 [90, ReLU] -> Output Layer [N, Softmax]
    N = 47 classes (EMNIST Balanced split: 10 digits, 26 uppercase, 11 distinct lowercase)
    or N = 10 classes (MNIST baseline).

Saved Artifacts:
- models/emnist_mlp_model.joblib: Serialized trained shallow MLP classifier.
- models/feature_scaler.joblib: Fitted StandardScaler for runtime feature normalization.
- models/label_mapping.json: Integer class index (0..N-1) to ASCII character string map.
- models/training_metrics.json: Train/validation loss, test accuracy, and classification summary.
- outputs/debug/confusion_matrix.png: Visual confusion matrix inspection plot.
"""

import argparse
import json
import math
import random
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
import cv2
import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler

try:
    from . import config
    from .features import extract_features, extract_features_batch
    from .normalization import resize_and_center_16x16
except ImportError:
    import config
    from features import extract_features, extract_features_batch
    from normalization import resize_and_center_16x16


# -------------------------------------------------------------------------
# Official EMNIST Balanced (47 Classes) & MNIST (10 Classes) Label Maps
# -------------------------------------------------------------------------

EMNIST_BALANCED_CLASSES: List[str] = [
    # 0-9 (10 Digits)
    "0", "1", "2", "3", "4", "5", "6", "7", "8", "9",
    # 10-35 (26 Uppercase Letters)
    "A", "B", "C", "D", "E", "F", "G", "H", "I", "J",
    "K", "L", "M", "N", "O", "P", "Q", "R", "S", "T",
    "U", "V", "W", "X", "Y", "Z",
    # 36-46 (11 Case-Distinct Lowercase Letters)
    "a", "b", "d", "e", "f", "g", "h", "n", "q", "r", "t",
]

MNIST_CLASSES: List[str] = [
    "0", "1", "2", "3", "4", "5", "6", "7", "8", "9"
]


def get_class_list(dataset: str = "emnist", split: str = "balanced") -> List[str]:
    """Returns the list of character classes for the chosen dataset."""
    ds = dataset.lower().strip()
    if ds == "mnist":
        return MNIST_CLASSES
    return EMNIST_BALANCED_CLASSES


def build_label_mapping(classes: List[str]) -> Dict[str, str]:
    """Builds integer index string -> ASCII character dictionary."""
    return {str(idx): char for idx, char in enumerate(classes)}


# -------------------------------------------------------------------------
# Synthetic Augmented Handwriting Dataset Generator (Offline Engine)
# -------------------------------------------------------------------------

def generate_augmented_character_glyph(
    char: str,
    font: int,
    scale: float,
    thickness: int,
    rotation_angle: float,
    shear_slant: float,
) -> np.ndarray:
    """
    Generates a single augmented 16x16 normalized character image with realistic
    handwriting perturbations (rotations, slants, stroke thicknesses, and baseline shifts).
    """
    canvas_size = 36
    canvas = np.zeros((canvas_size, canvas_size), dtype=np.uint8)

    # Calculate glyph text size
    (tw, th), base = cv2.getTextSize(char, font, scale, thickness)
    tx = max(2, (canvas_size - tw) // 2 + random.randint(-1, 1))
    ty = max(th + 2, (canvas_size + th) // 2 + random.randint(-1, 1))

    # Render character onto canvas
    cv2.putText(canvas, char, (tx, ty), font, scale, 255, thickness, cv2.LINE_AA)

    # 1. Apply affine shear / slant simulation
    if abs(shear_slant) > 0.05:
        M_shear = np.array([[1.0, shear_slant, 0.0], [0.0, 1.0, 0.0]], dtype=np.float32)
        canvas = cv2.warpAffine(canvas, M_shear, (canvas_size, canvas_size), borderValue=0)

    # 2. Apply random rotation
    if abs(rotation_angle) > 1.0:
        M_rot = cv2.getRotationMatrix2D((canvas_size / 2.0, canvas_size / 2.0), rotation_angle, 1.0)
        canvas = cv2.warpAffine(canvas, M_rot, (canvas_size, canvas_size), borderValue=0)

    # 3. Add subtle sensor grain / line fuzziness
    noise = np.random.normal(0, 3, canvas.shape).astype(np.int16)
    canvas = np.clip(canvas.astype(np.int16) + noise, 0, 255).astype(np.uint8)

    # 4. Normalize to standard 16x16 grid
    char_16x16 = resize_and_center_16x16(canvas)
    return char_16x16


def generate_handwriting_dataset(
    classes: List[str],
    samples_per_class: int = 80,
    feature_mode: str = config.FEATURE_MODE,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generates a multi-style handwriting feature matrix across all target classes.
    """
    fonts = [
        cv2.FONT_HERSHEY_SIMPLEX,
        cv2.FONT_HERSHEY_DUPLEX,
        cv2.FONT_HERSHEY_COMPLEX,
        cv2.FONT_HERSHEY_TRIPLEX,
    ]

    total_samples = len(classes) * samples_per_class
    print(f"[Dataset Generator] Generating {total_samples:,} synthetic handwriting samples ({len(classes)} classes x {samples_per_class} samples)...")

    X_features = []
    y_labels = []

    for label_idx, char in enumerate(classes):
        for s in range(samples_per_class):
            font = fonts[s % len(fonts)]
            scale = random.uniform(0.65, 0.95)
            thickness = 1 if s % 3 != 0 else 2
            rot = random.uniform(-14.0, 14.0)
            shear = random.uniform(-0.25, 0.25)

            char_img = generate_augmented_character_glyph(
                char=char,
                font=font,
                scale=scale,
                thickness=thickness,
                rotation_angle=rot,
                shear_slant=shear,
            )

            vec = extract_features(char_img, mode=feature_mode)
            X_features.append(vec)
            y_labels.append(label_idx)

    X = np.array(X_features, dtype=np.float32)
    y = np.array(y_labels, dtype=np.int32)
    return X, y


# -------------------------------------------------------------------------
# Training Pipeline
# -------------------------------------------------------------------------

def train_mlp_classifier(
    dataset_name: str = "emnist",
    split: str = "balanced",
    epochs: int = config.MLP_MAX_ITER,
    batch_size: int = config.MLP_BATCH_SIZE,
    samples_per_class: int = 100,
    feature_mode: str = config.FEATURE_MODE,
    random_state: int = config.MLP_RANDOM_STATE,
) -> Dict[str, Any]:
    """
    Trains the shallow MLP classifier (256 -> 100 -> 90 -> N classes), evaluates performance,
    and serializes all required model artifacts.

    Returns:
        Dictionary of training metrics, execution time, and evaluation summary.
    """
    classes = get_class_list(dataset_name, split)
    num_classes = len(classes)
    label_mapping = build_label_mapping(classes)

    print("=" * 65)
    print("PHASE 6: SHALLOW MLP CLASSIFIER TRAINING")
    print("=" * 65)
    print(f"Target Dataset      : {dataset_name.upper()} ({split})")
    print(f"Number of Classes   : {num_classes} ({classes[0]}..{classes[-1]})")
    print(f"Network Topology    : 256 -> 100 -> 90 -> {num_classes} (Softmax)")
    print(f"Feature Mode        : {feature_mode} (256-D)")
    print(f"Max Iterations      : {epochs}")
    print(f"Mini-Batch Size     : {batch_size}")
    print("=" * 65)

    # 1. Acquire Dataset (Offline synthetic generator with augmented handwriting styles)
    t0 = time.time()
    X, y = generate_handwriting_dataset(
        classes=classes,
        samples_per_class=samples_per_class,
        feature_mode=feature_mode,
    )
    data_gen_time = time.time() - t0
    print(f"[Dataset] Generated {X.shape[0]:,} samples in {data_gen_time:.2f}s. Feature matrix shape: {X.shape}")

    # 2. Stratified Train / Test Split (80% Train, 20% Test)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=random_state, stratify=y
    )
    print(f"[Split] Training set: {X_train.shape[0]:,} samples | Test set: {X_test.shape[0]:,} samples")

    # 3. Fit Feature Scaler (StandardScaler)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # 4. Initialize Shallow MLP Classifier
    clf = MLPClassifier(
        hidden_layer_sizes=config.MLP_HIDDEN_LAYERS,
        activation="relu",
        solver="adam",
        batch_size=min(batch_size, len(X_train)),
        learning_rate_init=config.MLP_LEARNING_RATE_INIT,
        max_iter=epochs,
        random_state=random_state,
        early_stopping=True,
        n_iter_no_change=5,
        verbose=False,
    )

    # 5. Train Model
    print("\n[Training] Fitting shallow MLP via Adam optimization...")
    t_train_start = time.time()
    clf.fit(X_train_scaled, y_train)
    train_duration = time.time() - t_train_start
    print(f"[Training] Convergence reached in {clf.n_iter_} iterations ({train_duration:.2f}s).")

    # 6. Evaluate on Test Set
    y_pred = clf.predict(X_test_scaled)
    train_acc = float(accuracy_score(y_train, clf.predict(X_train_scaled)))
    test_acc = float(accuracy_score(y_test, y_pred))

    print(f"[Evaluation] Training Set Accuracy : {train_acc * 100:.2f}%")
    print(f"[Evaluation] Test Set Accuracy     : {test_acc * 100:.2f}%")

    # 7. Classification Report & Confusion Matrix
    report = classification_report(y_test, y_pred, target_names=classes, output_dict=True)
    cm = confusion_matrix(y_test, y_pred)

    # 8. Render & Save Confusion Matrix Plot
    fig, ax = plt.subplots(figsize=(10, 8))
    im = ax.imshow(cm, cmap="Blues", interpolation="nearest")
    ax.set_title(f"Phase 6 Shallow MLP Confusion Matrix ({test_acc * 100:.1f}% Test Acc)", fontsize=12, fontweight="bold")
    ax.set_xlabel("Predicted Class Index")
    ax.set_ylabel("True Class Index")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    plt.tight_layout()
    config.CONFUSION_MATRIX_FILE.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(config.CONFUSION_MATRIX_FILE), dpi=150)
    plt.close(fig)
    print(f"[Export] Saved confusion matrix plot to: {config.CONFUSION_MATRIX_FILE}")

    # 9. Serialize Model Artifacts
    config.MODELS_DIR.mkdir(parents=True, exist_ok=True)

    joblib.dump(clf, config.MODEL_FILE)
    print(f"[Export] Saved trained model to: {config.MODEL_FILE}")

    joblib.dump(scaler, config.SCALER_FILE)
    print(f"[Export] Saved feature scaler to: {config.SCALER_FILE}")

    with open(config.LABEL_MAP_FILE, "w", encoding="utf-8") as f:
        json.dump(label_mapping, f, indent=2)
    print(f"[Export] Saved label mapping to: {config.LABEL_MAP_FILE}")

    # Summary metrics
    metrics = {
        "dataset": dataset_name,
        "split": split,
        "num_classes": num_classes,
        "train_samples": int(X_train.shape[0]),
        "test_samples": int(X_test.shape[0]),
        "train_accuracy": train_acc,
        "test_accuracy": test_acc,
        "iterations": int(clf.n_iter_),
        "training_time_seconds": round(train_duration, 3),
        "loss_curve": [float(loss) for loss in getattr(clf, "loss_curve_", [])],
        "feature_mode": feature_mode,
        "hidden_layer_sizes": list(config.MLP_HIDDEN_LAYERS),
        "classification_summary": {
            "macro_avg_f1": float(report["macro avg"]["f1-score"]),
            "weighted_avg_f1": float(report["weighted avg"]["f1-score"]),
        },
    }

    with open(config.METRICS_FILE, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)
    print(f"[Export] Saved training metrics to: {config.METRICS_FILE}")
    print("=" * 65)

    return metrics


def main():
    parser = argparse.ArgumentParser(description="Phase 6 Shallow MLP Training & Evaluation")
    parser.add_argument("--dataset", type=str, default="emnist", choices=["emnist", "mnist"], help="Dataset to train on")
    parser.add_argument("--split", type=str, default="balanced", help="EMNIST split name")
    parser.add_argument("--epochs", type=int, default=50, help="Maximum training epochs")
    parser.add_argument("--batch-size", type=int, default=128, help="Mini-batch size")
    parser.add_argument("--samples-per-class", type=int, default=80, help="Augmented samples per class")
    parser.add_argument("--mode", type=str, default=config.FEATURE_MODE, help="Feature extraction mode")

    args = parser.parse_args()
    train_mlp_classifier(
        dataset_name=args.dataset,
        split=args.split,
        epochs=args.epochs,
        batch_size=args.batch_size,
        samples_per_class=args.samples_per_class,
        feature_mode=args.mode,
    )


if __name__ == "__main__":
    main()
