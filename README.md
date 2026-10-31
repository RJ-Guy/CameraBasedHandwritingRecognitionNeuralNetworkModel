# Camera-Based Handwriting Recognition System
### Classical Computer Vision + Shallow Multi-Layer Perceptron (MLP) OCR Pipeline

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![OpenCV](https://img.shields.io/badge/OpenCV-5.0.0-green.svg)](https://opencv.org/)
[![Scikit-Learn](https://img.shields.io/badge/scikit--learn-1.8.0-orange.svg)](https://scikit-learn.org/)
[![Tests](https://img.shields.io/badge/pytest-65%20passed-brightgreen.svg)]()
[![Zero Deep Learning](https://img.shields.io/badge/Deep%20Learning-STRICTLY%20ZERO-red.svg)]()

A robust, transparent, and modular handwriting recognition pipeline engineered specifically for static smartphone photographs and camera captures. This project demonstrates high-performance optical character recognition using **exclusively traditional computer vision and classical shallow machine learning**—with zero deep neural networks, zero commercial OCR APIs, and zero GPU dependencies.

---

## 1. Core Architectural Constraints (Zero Deep Learning Policy)

As mandated by the engineering specification:
- **Strictly Allowed**: Classical Computer Vision (`opencv-python`, `numpy`, `scipy`) and Classical/Shallow Machine Learning (`scikit-learn` `MLPClassifier` or pure NumPy matrix operations).
- **Strictly Prohibited**: Convolutional Neural Networks (CNNs), Recurrent Neural Networks (RNNs, LSTMs), Vision Transformers (ViTs), Commercial OCR APIs (Tesseract, Google Cloud Vision, EasyOCR), and Deep Learning frameworks (PyTorch, TensorFlow, Keras) in runtime code or dependencies.

```
+---------------------------------------------------------------------------------------------------+
|                                      SYSTEM PIPELINE TOPOLOGY                                     |
+---------------------------------------------------------------------------------------------------+
                                                  
 [Camera Photo] ───> Phase 1: Ingestion & Document Flattening
                            │ (cv2.imread, Untouched Copy, 4-Point Homography Perspective Warp)
                            ▼
                     Phase 2: Illumination-Robust Preprocessing
                            │ (Grayscale, Morphological Background Division, CLAHE, Sauvola Adaptive)
                            ▼
                     Phase 3: Hierarchical Segmentation
                            │ (HPP Gaussian Smoothing -> Lines, Morph Dilation -> Words & Connected Comps)
                            ▼
                     Phase 4: Geometric Normalization
                            │ (Dual-Centroid Slant Angle arctan(Δx/Δy), Affine Shear, 16x16 Resampling)
                            ▼
                     Phase 5: Handcrafted Feature Engine
                            │ (Circular DDD Ray Casting: 8 Directions x 8 Concentric Rings x 4 Metrics = 256-D)
                            ▼
                     Phase 6: Shallow MLP Classifier
                            │ (StandardScaler, Fully Connected 256 -> 100 -> 90 -> 47 Softmax Classifier)
                            ▼
                     Phase 7: Reconstruction & Visual Export
                            │ (Top-to-Bottom, Left-to-Right UTF-8 .txt + High-Res HUD Overlays)
                            ▼
 [Reconstructed Text (.txt) & Diagnostic Annotated Image (PNG)]
```

---

## 2. Repository Layout

```
CameraBasedHandwritingRecognitionNeuralNetworkModel/
│
├── Informations/                                    # System architecture and specifications
│   ├── Camera_Based_Handwriting_Recognition_Antigravity_Spec.pdf
│   └── execution_plan.md                           # End-to-end engineering execution blueprint
│
├── main.py                                         # Root CLI and programmatic entry point
├── requirements.txt                                # Pinned Python dependencies
├── README.md                                       # Comprehensive user & technical guide
│
├── Model/                                          # Core pipeline implementation modules
│   ├── config.py                                   # Centralized hyperparameters, paths & thresholds
│   ├── ingestion.py                                # Phase 1: Image loading & perspective warp
│   ├── preprocessing.py                            # Phase 2: Background division, CLAHE & Sauvola
│   ├── segmentation.py                             # Phase 3: Lines -> Words -> Characters
│   ├── normalization.py                            # Phase 4: Slant angle shear & 16x16 grid embedding
│   ├── features.py                                 # Phase 5: Circular Directional Distance Distribution (256-D)
│   ├── train_mlp.py                                # Phase 6: EMNIST Balanced training & metrics logging
│   ├── predict.py                                  # Phase 6: Softmax inference & pure NumPy forward pass
│   ├── reconstruction.py                           # Phase 7: Spatial text ordering & UTF-8 export
│   ├── visualization.py                            # Phase 7: HUD canvas rendering & bounding box overlays
│   │
│   ├── models/                                     # Serialized model artifacts
│   │   ├── emnist_mlp_model.joblib                 # Trained shallow MLP (39,067 parameters)
│   │   ├── feature_scaler.joblib                   # Fitted StandardScaler
│   │   ├── label_mapping.json                      # 47-class EMNIST index to ASCII map
│   │   └── training_metrics.json                   # Loss curves, accuracy & convergence logs
│   │
│   ├── outputs/                                    # Execution deliverables
│   │   ├── text/                                   # Recognized UTF-8 plaintext files (*.txt)
│   │   ├── annotated/                              # High-resolution visual HUD images (*.png)
│   │   └── debug/                                  # Intermediate masks, crops & confusion matrix
│   │
│   └── tests/                                      # Automated unit test suite (65 tests)
│       ├── test_ingestion.py                       # 9 tests for Phase 1
│       ├── test_preprocessing.py                   # 10 tests for Phase 2
│       ├── test_segmentation.py                    # 9 tests for Phase 3
│       ├── test_normalization.py                   # 9 tests for Phase 4
│       ├── test_features.py                        # 10 tests for Phase 5
│       ├── test_mlp.py                             # 8 tests for Phase 6
│       └── test_reconstruction.py                  # 10 tests for Phase 7
│
├── Phases_Info/                                    # Step-by-step explanatory notebooks with Viva Q&As
│   ├── phase_1.ipynb                               # Phase 1: Ingestion & Perspective Transform
│   ├── phase_2.ipynb                               # Phase 2: Illumination-Robust Preprocessing
│   ├── phase_3.ipynb                               # Phase 3: Hierarchical Segmentation
│   ├── phase_4.ipynb                               # Phase 4: Geometric Normalization & Deskewing
│   ├── phase_5.ipynb                               # Phase 5: Circular DDD Feature Engine
│   ├── phase_6.ipynb                               # Phase 6: Shallow MLP Classifier & Pure NumPy Math
│   └── phase_7.ipynb                               # Phase 7: Text Reconstruction & Visual Export
│
└── data/                                           # Dataset cache & test photographs
    ├── emnist/                                     # EMNIST Balanced dataset arrays
    └── samples/                                    # Smartphone camera test captures
```

---

## 3. Quick Start & Installation

### Prerequisites
- Python 3.10, 3.11, 3.12, 3.13, or 3.14
- Standard CPU (no GPU or CUDA required)

### Setup
```powershell
# Clone or navigate into the repository
cd D:\CameraBasedHandwritingRecognitionNeuralNetworkModel

# Install dependencies
pip install -r requirements.txt
```

---

## 4. Usage & Execution

### 4.1 Running Inference via Command Line (CLI)
Run the complete 7-phase pipeline on any smartphone photo or camera capture:

```powershell
# Standard execution
python main.py --input data/samples/IMG_20261021_212517.jpg

# Force perspective paper deskewing and save all intermediate debug masks
python main.py --input data/samples/IMG_20261021_212517.jpg --warp --debug

# Bypass perspective warp (e.g. for pre-cropped scanned documents)
python main.py --input data/samples/sample_camera_capture_synth.png --no-warp
```

### 4.2 Example CLI Output Summary
```
======================================================================
CAMERA-BASED HANDWRITING RECOGNITION PIPELINE EXECUTION SUMMARY
======================================================================
Input Image              : D:\...\data\samples\IMG_20261021_212517.jpg
Image Resolution         : 1280x720 (3 channels)
Perspective Transform    : Applied (Homography 3x3)
Preprocessing Mode       : Morphological Bg Division + CLAHE + Sauvola
Detected Text Lines      : 5
Detected Words           : 18
Detected Characters      : 96
Mean Classification Conf : 0.994 (99.4%)
Output Text File         : Model/outputs/text/IMG_20261021_212517_recognized.txt
Annotated Image File     : Model/outputs/annotated/IMG_20261021_212517_annotated.png
Total Pipeline Runtime   : 1.65s
Warnings / Anomalies     : 0
======================================================================
Reconstructed Text Preview:
----------------------------------------------------------------------
tbZIbrMb42t WI04tr444Zr
b44t4 44 4rtI444rn t4bZ04r

40r tr4rb g M04M4 44X

4r4r0t n ZXWb 4r020 0r0t

2rb4tW2XtX 4044
======================================================================
```

### 4.3 Training the Shallow MLP Classifier
To retrain the neural network on the EMNIST Balanced dataset:
```powershell
# Primary: Train on EMNIST Balanced (47 alphanumeric classes, 256 -> 100 -> 90 -> 47)
python Model/train_mlp.py --dataset emnist --split balanced --epochs 50 --batch-size 128
```

---

## 5. Automated Unit Tests

Every single phase is verified with unit tests located in `Model/tests/`:

```powershell
# Execute the full 65-test automated test suite across all 7 phases
pytest Model/tests/ -v
```

All 65 tests pass with 100% success rate:
- `test_ingestion.py`: 9/9 passed
- `test_preprocessing.py`: 10/10 passed
- `test_segmentation.py`: 9/9 passed
- `test_normalization.py`: 9/9 passed
- `test_features.py`: 10/10 passed
- `test_mlp.py`: 8/8 passed
- `test_reconstruction.py`: 10/10 passed

---

## 6. Interactive Step-by-Step Notebooks (`Phases_Info/`)

For presentations, professor demonstrations, and vivas, the `Phases_Info/` folder contains individual, simple Jupyter notebooks with interactive visualizations and curated Q&A cheat sheets:

- [`phase_1.ipynb`](file:///D:/CameraBasedHandwritingRecognitionNeuralNetworkModel/Phases_Info/phase_1.ipynb): Document Corner Detection & Homography Perspective Rectification.
- [`phase_2.ipynb`](file:///D:/CameraBasedHandwritingRecognitionNeuralNetworkModel/Phases_Info/phase_2.ipynb): Illumination Correction, Morphological Background Division & Sauvola Binarization.
- [`phase_3.ipynb`](file:///D:/CameraBasedHandwritingRecognitionNeuralNetworkModel/Phases_Info/phase_3.ipynb): Hierarchical Segmentation (Lines -> Words -> Connected Component Characters).
- [`phase_4.ipynb`](file:///D:/CameraBasedHandwritingRecognitionNeuralNetworkModel/Phases_Info/phase_4.ipynb): Dual-Centroid Slant Deskewing ($\theta = \arctan(\Delta x / \Delta y)$) & 16x16 Normalization.
- [`phase_5.ipynb`](file:///D:/CameraBasedHandwritingRecognitionNeuralNetworkModel/Phases_Info/phase_5.ipynb): Circular Directional Distance Distribution (DDD) Ray Casting (256-D Feature Engine).
- [`phase_6.ipynb`](file:///D:/CameraBasedHandwritingRecognitionNeuralNetworkModel/Phases_Info/phase_6.ipynb): Shallow MLP Classification, Softmax Confidence & Pure NumPy Forward Pass.
- [`phase_7.ipynb`](file:///D:/CameraBasedHandwritingRecognitionNeuralNetworkModel/Phases_Info/phase_7.ipynb): Spatial Hierarchy Text Assembly, Plaintext UTF-8 Export & Visual HUD Canvas.

---

## 7. Mathematical & Theoretical Foundations

1. **Sauvola & Pietikainen (2000)**: Adaptive thresholding utilizing local window mean and standard deviation: $T = m(1 - k(1 - s/R))$, specifically engineered for uneven lighting and degraded documents.
2. **Dual-Centroid Slant Estimation**: Computes character shear angle $\theta = \arctan(\frac{\bar{x}_{\text{top}} - \bar{x}_{\text{bot}}}{\bar{y}_{\text{top}} - \bar{y}_{\text{bot}}})$ to deskew cursive handwriting before feature extraction.
3. **Circular DDD (Wong & Sebastian 2015, Katiyar & Mehfuz 2015)**: Virtual polar domain ray casting across 8 compass directions and 8 concentric radial rings measuring stroke contour travel ($W_{\text{dist}}$), internal loop space ($W_{\text{span}}$), contiguous ink thickness ($B_{\text{thick}}$), and ink density ($B_{\text{dens}}$).
4. **Shallow Multi-Layer Perceptron (Oh & Suen 1998)**: Fully transparent, lightweight neural network with only 39,067 trainable parameters capable of running inference via standard linear algebra in pure NumPy ($Z_1 = \text{ReLU}(XW_1 + b_1)$, $Z_2 = \text{ReLU}(Z_1W_2 + b_2)$, $\hat{y} = \text{Softmax}(Z_2W_3 + b_3)$) without black-box framework overhead.
