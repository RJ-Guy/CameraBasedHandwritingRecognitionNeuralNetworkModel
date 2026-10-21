# End-to-End Execution Plan: Camera-Based Handwriting Recognition System
**Implementation Specification:** Classical Computer Vision + Shallow Multi-Layer Perceptron (MLP)  
**Document Source:** `Informations/Camera_Based_Handwriting_Recognition_Antigravity_Spec.pdf`  
**Execution Target:** `D:\CameraBasedHandwritingRecognitionNeuralNetworkModel`  
**Target Output Document:** `Informations/execution_plan.md`  

---

## 1. Executive Summary & Core Architectural Constraints

This document outlines the complete architectural design, mathematical foundations, module breakdown, and phase-by-phase implementation schedule for the **Camera-Based Handwriting Recognition System**. The objective is to build a robust, testable, modular, end-to-end optical character recognition pipeline that ingests unconstrained static camera or smartphone photographs (JPEG/PNG), isolates and preprocesses the handwriting, extracts invariant handcrafted geometric/topological features, classifies individual characters using a shallow neural network, and reconstructs structured digital text alongside visual diagnostic annotations.

### Strict Architectural Boundaries (Zero Deep Learning Policy)
In strict accordance with the project specification:
* **Allowed Technologies:** Traditional Computer Vision (OpenCV, NumPy, SciPy) and Classical/Shallow Machine Learning (scikit-learn `MLPClassifier` or pure NumPy multi-layer perceptron).
* **Strictly Prohibited Technologies:** Convolutional Neural Networks (CNNs), Recurrent Neural Networks (RNNs, LSTMs, GRUs), Vision Transformers (ViTs), Pretrained Vision-Language Models (VLMs), Commercial OCR APIs (Tesseract, Google Cloud Vision, EasyOCR), and Deep Learning frameworks (PyTorch, TensorFlow, Keras) in runtime code or dependencies.
* **Module Independence:** Each stage is developed as an isolated, testable Python module with explicit inputs, deterministic outputs, parameterization via a centralized configuration file, and independent unit tests.

```
+---------------------------------------------------------------------------------------------------+
|                                      SYSTEM PIPELINE TOPOLOGY                                     |
+---------------------------------------------------------------------------------------------------+
                                                  
 [Input Image] ───> Phase 1: Ingestion & Document Flattening
                           │ (cv2.imread, Untouched Copy, Optional 4-Point Perspective Warp)
                           ▼
                    Phase 2: Illumination-Robust Preprocessing
                           │ (Grayscale, Morphological Background Division, CLAHE, Sauvola / Adaptive)
                           ▼
                    Phase 3: Hierarchical Segmentation
                           │ (Horizontal Projection -> Lines, Vertical Projection -> Words & Chars)
                           ▼
                    Phase 4: Geometric Normalization
                           │ (Dual-Centroid Slant Estimation, Affine Shear, 16x16 Grid Resampling)
                           ▼
                    Phase 5: Handcrafted Feature Extraction Engine
                           │ (Circular DDD Ray Casting: 8 Directions, W-Set & B-Set -> 256-D Vector)
                           ▼
                    Phase 6: Shallow MLP Classification
                           │ (Feature Scaler, Fully Connected 256 -> 100 -> 90 -> N Softmax Classifier)
                           ▼
                    Phase 7: Reconstruction & Visual Export
                           │ (Top-to-Bottom, Left-to-Right UTF-8 .txt + High-Res Bounding Box Overlays)
                           ▼
 [Reconstructed Text (.txt) & Diagnostic Annotated Image (PNG)]
```

---

## 2. Directory Structure & File Organization

The project will follow a clean, modular repository layout conforming to Section 4 of the project brief:

```
CameraBasedHandwritingRecognitionNeuralNetworkModel/
│
├── Informations/                                    # Project documentation & reference briefs
│   ├── Camera_Based_Handwriting_Recognition_Antigravity_Spec.pdf
│   └── execution_plan.md                           # This execution plan
│
├── config.py                                       # Centralized configuration, hyperparameters, thresholds
├── main.py                                         # CLI entry point for full pipeline execution
├── ingestion.py                                    # Phase 1: Image loading, validation, perspective transform
├── preprocessing.py                                # Phase 2: Illumination equalization, CLAHE, binarization
├── segmentation.py                                 # Phase 3: Hierarchical Line -> Word -> Character extraction
├── normalization.py                                # Phase 4: Slant angle estimation, affine shear, 16x16 resize
├── features.py                                     # Phase 5: Circular Directional Distance Distribution (DDD)
├── train_mlp.py                                    # Phase 6: EMNIST dataset loading, MLP training & evaluation
├── predict.py                                      # Phase 6: Inference engine, confidence scoring & label decoding
├── reconstruction.py                               # Phase 7: Text ordering, spatial formatting, UTF-8 export
├── visualization.py                                # Phase 7: Debug canvas rendering, HUD overlays, inspection plots
│
├── models/                                         # Serialized model artifacts
│   ├── emnist_mlp_model.joblib                     # Trained shallow MLP classifier
│   ├── feature_scaler.joblib                       # Fitted feature scaler (StandardScaler / MinMax)
│   ├── label_mapping.json                          # Class index to ASCII character map
│   └── training_metrics.json                       # Train/validation loss curves, per-class accuracy
│
├── data/                                           # Dataset cache and sample inputs
│   ├── emnist/                                     # Downloaded/extracted EMNIST dataset arrays
│   └── samples/                                    # Test camera captures and phone photographs
│
├── outputs/                                        # Run outputs and debugging artifacts
│   ├── text/                                       # Reconstructed UTF-8 text files (*.txt)
│   ├── annotated/                                  # Final annotated visual verification images
│   └── debug/                                      # Intermediate stage masks, crops, and projections
│
├── tests/                                          # Automated unit test suite
│   ├── test_ingestion.py
│   ├── test_preprocessing.py
│   ├── test_segmentation.py
│   ├── test_normalization.py
│   ├── test_features.py
│   ├── test_mlp.py
│   └── test_reconstruction.py
│
├── requirements.txt                                # Minimal, lightweight dependencies
└── README.md                                       # Comprehensive setup, training, and inference guide
```

---

## 3. Analysis & Resolution of Critical Design Questions

Before implementation, three critical ambiguities highlighted in Section 3 of the specification must be formally resolved:

### 3.1 Feature Dimension Conflict (256-D vs 16-D)
* **The Conflict:** Section 3 of the brief identifies that a $16 \times 16$ normalized character image has $256$ total pixels. However, performing $4 \times 4$ spatial block averaging yields only $16$ scalar values ($4 \times 4 = 16$), creating a mathematical contradiction with a stated "256-dimensional invariant vector" feed to an MLP input layer of $256$ nodes.
* **Architectural Resolution:**
  1. **Primary Standard Mode (Direct Normalized Grayscale Pixels):** Flattening the $16 \times 16$ geometric-normalized character crop directly yields a $256$-dimensional feature vector ($\mathbb{R}^{256}$). When scaled to $[0.0, 1.0]$, this provides rich grayscale topology directly compatible with the $256 \to 100 \to 90 \to N$ MLP architecture.
  2. **Handcrafted Feature Mode (256-D Circular DDD Descriptor):** The Circular Directional Distance Distribution (DDD) engine will be formulated to produce an exact $256$-dimensional feature vector:
     $$\text{Feature Dimension} = K \text{ concentric rings / radial zones} \times 8 \text{ directional rays} \times 4 \text{ statistical measures} = 256\text{ features}$$
     Specifically: 8 radial circular zones ($r_1 \dots r_8$) $\times$ 8 directions ($\theta_0 \dots \theta_7$) $\times$ 4 metrics ($W_{\text{dist}}$, $W_{\text{count}}$, $B_{\text{thickness}}$, $B_{\text{density}}$) $= 8 \times 8 \times 4 = 256$ dimensions.
  3. **Implementation Standard:** `features.py` will explicitly support both modes via a configuration toggle (`config.FEATURE_MODE = 'CIRCULAR_DDD_256'` or `'FLAT_PIXEL_256'`), ensuring zero silent mixing of representations while strictly preserving the $256$-dimensional input requirement.

### 3.2 Circular DDD Formulation & Boundary Behavior
* **Mathematical Definition:**
  * Given a centered character crop bounded by an inscribed circle of radius $R = 8$ (within the $16 \times 16$ grid):
  * **Coordinate Wrap-Around / Circular Tiling:** Points traversing outside the boundary $[0, 15]$ wrap circularly or terminate at the virtual circle perimeter, ensuring boundary consistency.
  * **8 Discrete Ray Angles:** $\theta_k = k \cdot \frac{\pi}{4}$ for $k \in \{0, 1, 2, 3, 4, 5, 6, 7\}$ corresponding to $E, NE, N, NW, W, SW, S, SE$.
  * **Dual Sets:**
    * **W-Set (White / Background / Ray Distance Features):** Traverses the ray from sample points to detect distance to first stroke edge, total white-space span, and cavity depths (representing character loops, open enclosures, and structural concavities).
    * **B-Set (Black / Foreground / Stroke Thickness Features):** Measures continuous black pixel run-lengths along the ray, stroke intersection density, and local pen thickness.
  * All extracted features are strictly finite, normalized to $[0.0, 1.0]$, and verified to contain zero `NaN` or `Inf` values.

### 3.3 Dataset Specification & Selection: MNIST vs. EMNIST
A central architectural decision in optical handwriting recognition is determining the character domain of the classifier. We explicitly address the choice between **MNIST** and **EMNIST**:

| Feature / Metric | Standard MNIST | EMNIST (Balanced) [CHOSEN PRIMARY] | EMNIST (Letters) | EMNIST (ByMerge / ByClass) |
| :--- | :--- | :--- | :--- | :--- |
| **Character Domain** | Digits only (`0`–`9`) | **Alphanumeric** (`0`–`9`, `A`–`Z`, `a`–`z`) | Letters only (`A`–`Z` / `a`–`z`) | Extended Alphanumeric |
| **Class Count ($N$)** | 10 classes | **47 classes** (10 digits + 37 letters) | 26 classes | 47 classes / 62 classes |
| **Total Samples** | 70,000 images | **131,600 images** (2,800 per class) | 145,600 images | 814,255 images |
| **Image Resolution** | $28 \times 28$ grayscale | **$28 \times 28$ grayscale** (NIST SD 19) | $28 \times 28$ grayscale | $28 \times 28$ grayscale |
| **Suitability for System**| **Insufficient on its own:** Incapable of recognizing English words, sentences, or alphabets. | **Ideal & Required:** Directly satisfies the project brief's mandate for full handwriting text reconstruction with balanced representation. | Restricted to alphabets; cannot recognize numbers/dates. | ByClass is heavily imbalanced; ByMerge has skewed frequencies. |

#### Architectural Decision & Formal Selection:
1. **Primary Dataset (EMNIST Balanced):**
   * **Mandate:** The application must recognize general camera-based handwriting (sentences, words, phone numbers, mixed notes). Standard MNIST recognizes *only* digits $0$ through $9$ and completely fails when presented with alphabetic characters (e.g. classifying 'E' as '8' or 'I' as '1').
   * **Selection:** We designate **EMNIST Balanced** as the primary dataset for Phase 6 training.
   * **Class Structure ($N = 47$):**
     * **10 Digits:** `0`, `1`, `2`, `3`, `4`, `5`, `6`, `7`, `8`, `9`.
     * **17 Uppercase Letters:** `A`, `B`, `D`, `E`, `F`, `G`, `H`, `I`, `J`, `K`, `L`, `M`, `N`, `Q`, `R`, `T`, `Y`.
     * **20 Case-Merged / Lowercase Letters:** `a`, `b`, `d`, `e`, `g`, `h`, `n`, `q`, `r`, `t`, along with merged glyphs whose lowercase and uppercase forms share identical geometry (`c/C`, `k/K`, `o/O`, `p/P`, `s/S`, `u/U`, `v/V`, `w/W`, `x/X`, `y/Y`, `z/Z`).
   * **Balanced Distribution:** 2,400 training samples and 400 testing samples per class, preventing frequency bias.
2. **Secondary / Sanity Verification Mode (Standard MNIST):**
   * The training pipeline will provide a configuration switch: `config.DATASET_NAME = 'emnist'` (default) or `'mnist'`.
   * When `DATASET_NAME = 'mnist'`, the MLP output layer adapts dynamically to $N = 10$ classes. This serves as a rapid sanity check and baseline benchmark for digit-only verification scripts.
3. **Orientation Rectification:**
   * Raw EMNIST binary files store images transposed and mirrored relative to standard raster orientation. The dataset loader in `train_mlp.py` will explicitly apply `np.transpose(img)` during extraction to ensure exact optical alignment with camera crops.
4. **Label Mapping Immutability:**
   * The exact mapping from integer class index ($0 \dots N-1$) to ASCII character is stored in `models/label_mapping.json` and shared immutably between training and inference pipelines.

---

## 4. Phase-by-Phase Detailed Specifications

---

### Phase 1: Image Ingestion and Document Flattening
**Module:** `ingestion.py`  
**Purpose:** Safely ingest static camera captures or smartphone photographs, detect document boundaries where available, apply a perspective warp to rectify camera tilt/skew, and preserve an untouched copy of the original image for final visual annotations.

#### 1. Input Specification
* File formats: Static image file (`.jpg`, `.jpeg`, `.png`, `.bmp`, `.tiff`).
* Sources: Smartphone camera captures, overhead desk camera frames, scanned document images.
* Dimensions: Arbitrary resolution (e.g., $1280 \times 720$, $1920 \times 1080$, $4032 \times 3024$).

#### 2. Algorithmic Steps & Logic
1. **Robust Loading & File Integrity Validation:**
   * Load the image using `cv2.imread(filepath, cv2.IMREAD_COLOR)`.
   * Check for `None` or empty array returns. Raise a clear, descriptive `FileNotFoundError` or `ValueError("Corrupted or unreadable image file: ...")`.
2. **Untouched Preservation:**
   * Deep-copy the original frame: `original_image = raw_frame.copy()`.
   * Store original dimensions: `orig_h, orig_w, orig_c = original_image.shape`.
3. **Optional Document Boundary & Corner Detection:**
   * Downscale copy for fast contour processing (maximum dimension $1024\text{px}$).
   * Convert downscaled copy to grayscale, apply bilateral filter or Gaussian blur ($5 \times 5, \sigma=1.0$), compute Canny edges ($50, 150$) or Otsu binarization.
   * Apply morphological dilation ($3 \times 3$) to close broken paper edges.
   * Find external contours with `cv2.findContours(..., cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)`.
   * Sort contours by area in descending order. Find the largest convex quadrilateral contour with 4 vertices using `cv2.approxPolyDP(contour, epsilon=0.02 * perimeter, closed=True)`.
   * Calculate polygon area: if polygon area $< 0.20 \times (\text{image area})$, reject automatic corner detection.
4. **Graceful Bypass & Manual Fallback:**
   * If 4 distinct paper corners are not detected reliably, or if `config.ENABLE_PERSPECTIVE_WARP = False`, or if the image is already a cropped document/page, **bypass the perspective transform automatically** and proceed with the original full frame.
   * Provide a mechanism to supply manual corner coordinates `[(x1, y1), (x2, y2), (x3, y3), (x4, y4)]`.
5. **Perspective Rectification (4-Point Warp):**
   * If 4 corners are established:
     * Order corners deterministically: Top-Left (min $x+y$), Top-Right (max $x-y$), Bottom-Right (max $x+y$), Bottom-Left (min $x-y$).
     * Compute target rectified width and height:
       $$W_{\text{target}} = \max(\|\text{TR} - \text{TL}\|_2, \|\text{BR} - \text{BL}\|_2)$$
       $$H_{\text{target}} = \max(\|\text{BL} - \text{TL}\|_2, \|\text{BR} - \text{TR}\|_2)$$
     * Construct destination coordinates: $[[0, 0], [W_{\text{target}}-1, 0], [W_{\text{target}}-1, H_{\text{target}}-1], [0, H_{\text{target}}-1]]$.
     * Compute homography matrix: $M = \text{cv2.getPerspectiveTransform}(\text{src\_pts}, \text{dst\_pts})$.
     * Apply warp: `flattened = cv2.warpPerspective(raw_frame, M, (W_target, H_target), flags=cv2.INTER_CUBIC)`.

#### 3. Output Specification
* `original_image`: Unaltered copy of input image for final overlay rendering.
* `ingested_image`: Perspective-flattened (or direct pass-through) BGR image for preprocessing.
* `transform_meta`: Dictionary containing boolean `warp_applied`, corner points `corners`, and transformation matrix $M$ (or `None`).

#### 4. Acceptance Criteria & Edge Cases
* Clear error messages on invalid paths, zero-byte files, or unsupported formats.
* Zero crash when no 4-corner document polygon exists (silent or logged fallback to pass-through).
* Preserved original image remains identical in pixel memory to the input file.

---

### Phase 2: Illumination-Robust Preprocessing
**Module:** `preprocessing.py`  
**Purpose:** Eliminate uneven lighting, desk shadows, paper creases, and phone camera exposure gradients, enhancing handwriting contrast and yielding a clean binary representation.

#### 1. Input Specification
* `ingested_image`: Perspective-flattened or direct BGR image from Phase 1.

#### 2. Algorithmic Steps & Logic
1. **Grayscale Conversion:**
   * Convert BGR to single-channel luminance:
     $$I_{\text{gray}} = 0.299 \cdot R + 0.587 \cdot G + 0.114 \cdot B \quad \text{via } \text{cv2.cvtColor}(I_{\text{bgr}}, \text{cv2.COLOR\_BGR2GRAY})$$
2. **Morphological Background Illumination Estimation:**
   * Physical paper photographs frequently feature directional lighting, palm shadows, or vignette roll-off.
   * Construct a large elliptical structuring element: $K_{\text{bg}} = \text{cv2.getStructuringElement}(\text{cv2.MORPH\_ELLIPSE}, (k_w, k_h))$ where $k_w = k_h \approx 41$ (scaled dynamically with image resolution: $k = 2 \cdot \lfloor\min(w, h) / 40\rfloor + 1$).
   * Estimate the background illumination envelope using morphological closing:
     $$I_{\text{bg}} = \text{cv2.morphologyEx}(I_{\text{gray}}, \text{cv2.MORPH\_CLOSE}, K_{\text{bg}})$$
3. **Safe Background Division & Normalization:**
   * Normalize the grayscale image against the background surface to cancel lighting gradients:
     $$I_{\text{norm}}(x, y) = \text{np.clip}\left( \frac{I_{\text{gray}}(x, y) + \epsilon}{I_{\text{bg}}(x, y) + \epsilon} \times 255.0, \; 0, \; 255 \right).\text{astype}(\text{np.uint8})$$
   * $\epsilon = 1.0$ is added to strictly prevent division by zero.
4. **Local Contrast Enhancement (CLAHE):**
   * Standard histogram equalization washes out local handwriting details.
   * Apply Contrast Limited Adaptive Histogram Equalization:
     $$\text{clahe} = \text{cv2.createCLAHE}(\text{clipLimit}=2.0, \text{tileGridSize}=(8, 8))$$
     $$I_{\text{clahe}} = \text{clahe.apply}(I_{\text{norm}})$$
5. **Adaptive Binarization (Sauvola & Gaussian Adaptive):**
   * **Sauvola Binarization (Primary):**
     * For each pixel $(x, y)$ in a local window of size $W \times W$ (default $25 \times 25$):
       Compute local sample mean $m(x, y)$ and standard deviation $s(x, y)$.
       Compute Sauvola threshold:
       $$T_{\text{Sauvola}}(x, y) = m(x, y) \cdot \left[ 1 + k \cdot \left( \frac{s(x, y)}{R} - 1 \right) \right]$$
       where dynamic range parameter $R = 128$, and sensitivity $k = 0.2$ (configurable in `config.py`).
     * Highly effective for separating degraded or low-contrast handwritten ink from varied paper backgrounds.
   * **Gaussian Adaptive Binarization (Fast Fallback):**
     * `cv2.adaptiveThreshold(I_clahe, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, blockSize=25, C=11)`
   * **Standard Inverted Binary Mask:** Ensure ink pixels are foreground ($255$ / white) and paper surface is background ($0$ / black) for subsequent contour/projection analysis.
6. **Despeckling & Noise Suppression:**
   * Apply small median blur ($3 \times 3$) or morphological opening ($2 \times 2$) to eliminate microscopic dust speckles without eroding fine pen strokes.

#### 3. Output Specification
* `gray_image`: Grayscale version of document.
* `background_estimate`: Morphological background image $I_{\text{bg}}$ (for diagnostic inspection).
* `enhanced_image`: Contrast-equalized image $I_{\text{clahe}}$.
* `binary_mask`: Inverted binary image ($255 = \text{ink}, 0 = \text{paper}$) ready for segmentation.

#### 4. Acceptance Criteria & Edge Cases
* Background division never triggers runtime division-by-zero warnings or `NaN` values.
* Intermediate images ($I_{\text{bg}}$, $I_{\text{norm}}$, $I_{\text{clahe}}$, $I_{\text{bin}}$) can be saved or inspected.
* Thin pen strokes remain unbroken, while wide shadows are removed.

---

### Phase 3: Hierarchical Segmentation (Line -> Word -> Character)
**Module:** `segmentation.py`  
**Purpose:** Recursively decompose the binary handwriting canvas into a clean hierarchy of text lines, words within lines, and candidate character bounding boxes, rejecting paper artifacts, scratches, and noise components.

#### 1. Input Specification
* `binary_mask`: Inverted binary image ($255 = \text{ink}, 0 = \text{background}$) from Phase 2.
* `gray_image`: Preprocessed grayscale image for extracting grayscale character crops.

#### 2. Algorithmic Steps & Logic
1. **Line Detection via Horizontal Projection Profiles (HPP):**
   * Compute row-wise sum of foreground pixels:
     $$H(y) = \sum_{x=0}^{W-1} \frac{I_{\text{bin}}(x, y)}{255}$$
   * Smooth $H(y)$ using a 1D Gaussian kernel ($\sigma=2.0$) to suppress intra-stroke oscillations.
   * Detect line boundaries:
     * Identify valleys where $H(y) < \tau_{\text{line\_thresh}} \cdot \max(H)$.
     * Group contiguous active rows into line candidate intervals $[y_{\text{start}}^{(i)}, y_{\text{end}}^{(i)}]$.
     * Merge line intervals separated by gaps smaller than `config.MIN_LINE_GAP`.
     * Reject spurious lines with vertical height smaller than `config.MIN_LINE_HEIGHT`.
   * Sort detected lines strictly top-to-bottom ($y_{\text{start}}^{(1)} < y_{\text{start}}^{(2)} < \dots$).
2. **Word Grouping via Morphological Bridging & Vertical Projection:**
   * For each detected line slice $L^{(i)}$:
     * Compute horizontal dilation with a rectangular structuring element:
       $$K_{\text{word}} = \text{cv2.getStructuringElement}(\text{cv2.MORPH\_RECT}, (k_{\text{word\_w}}, 1))$$
       where $k_{\text{word\_w}}$ corresponds to the estimated character spacing (e.g. $12\text{px}$).
     * This operation merges adjacent characters belonging to the same word while leaving inter-word spaces intact.
     * Find connected components or compute Vertical Projection Profile (VPP) across the dilated line slice:
       $$V(x) = \sum_{y=y_{\text{start}}}^{y_{\text{end}}} \frac{I_{\text{dilated}}(x, y)}{255}$$
     * Identify zero-crossings or low-density valleys in $V(x)$ to establish word bounding boxes $[x_{\text{word}}, y_{\text{word}}, w_{\text{word}}, h_{\text{word}}]$.
     * Sort words within each line strictly left-to-right ($x_1 < x_2 < \dots$).
3. **Character Candidate Box Extraction:**
   * Within each isolated word region:
     * Extract connected components with statistics using `cv2.connectedComponentsWithStats(word_binary, connectivity=8)`.
     * For each component, extract bounding box $[x_c, y_c, w_c, h_c]$ and pixel area $A_c$.
4. **Noise, Dot, and Scratch Rejection:**
   * Discard components failing physical character criteria:
     * **Minimum Area:** $A_c < \text{config.MIN_CHAR_AREA}$ (e.g. $< 15\text{px}^2$) $\to$ noise speckle / scanner dust.
     * **Minimum Dimensions:** $w_c < \text{config.MIN_CHAR_WIDTH}$ or $h_c < \text{config.MIN_CHAR_HEIGHT}$ (e.g. $< 4\text{px}$).
     * **Extreme Aspect Ratio:** $\frac{h_c}{w_c} > 8.0$ or $\frac{w_c}{h_c} > 8.0$ $\to$ horizontal ruling line or vertical scratch.
     * **Punctuation & Diacritic Handling:** Preserve punctuation (e.g. periods, commas, dots on 'i'/'j') if vertical alignment matches expected baseline/x-height, or group diacritics into the nearest lower character bounding box.
5. **Over-Segmentation Handling (Joined Cursive Mitigation):**
   * Where connected components span multiple characters (e.g., $w_c > 1.8 \times \text{median\_char\_width}$):
     * Compute local VPP valleys of the component stroke to identify ligature thinning points.
     * Expose tunable split thresholds in `config.py` (`OVERLAP_SPLIT_THRESHOLD`, `MAX_ASPECT_RATIO_SPLIT`).
6. **Hierarchical Data Structure Generation:**
   * Construct a clean Python data structure maintaining reading order:
     ```python
     hierarchy = [
         Line(
             line_id=0, bbox=(x, y, w, h),
             words=[
                 Word(
                     word_id=0, bbox=(x, y, w, h),
                     characters=[
                         CharacterCrop(char_id=0, bbox=(x, y, w, h), image_crop=np.ndarray),
                         ...
                     ]
                 ),
                 ...
             ]
         ),
         ...
     ]
     ```
7. **Debug Visualization Overlay:**
   * Render diagnostic overlays highlighting detected lines in Blue, words in Green, and character candidate boxes in Red, saving the output to `outputs/debug/segmentation_hierarchy.png`.

#### 3. Output Specification
* Structured hierarchy list of `Line` objects containing ordered `Word` and `CharacterCrop` objects.
* Diagnostic multi-level segmentation image with color-coded bounding boxes.

#### 4. Acceptance Criteria & Edge Cases
* Output bounding boxes are deterministic and strictly ordered top-to-bottom, left-to-right.
* Dust, dots, and ruling lines are filtered out without dropping valid characters (like 'I', 'l', '1', or small punctuation).
* Projection segmentation thresholds are cleanly exposed in `config.py`.

---

### Phase 4: Geometric Normalization
**Module:** `normalization.py`  
**Purpose:** Rectify handwriting slant variation (cursive tilt) via affine horizontal shear and normalize every isolated character crop into an identical, standardized $16 \times 16$ grayscale representation without stroke cropping or distortion.

#### 1. Input Specification
* Raw character crops $[h_c \times w_c]$ (grayscale and binary) from Phase 3.

#### 2. Algorithmic Steps & Logic
1. **Slant Estimation via Dual-Centroid Method:**
   * Human handwriting commonly features rightward (forward) or leftward (backward) slant.
   * Let the binary character crop have height $H$ and width $W$.
   * Partition the foreground ink pixels into two vertical zones:
     * **Top Zone:** $y \in [0, \lfloor H / 2 \rfloor)$
     * **Bottom Zone:** $y \in [\lfloor H / 2 \rfloor, H)$
   * Compute the center of mass (spatial centroid) of each zone:
     $$x_{\text{top}} = \frac{1}{\sum_{(x,y)\in \text{Top}} I(x,y)} \sum_{(x,y)\in \text{Top}} x \cdot I(x,y), \quad y_{\text{top}} = \frac{1}{\sum_{(x,y)\in \text{Top}} I(x,y)} \sum_{(x,y)\in \text{Top}} y \cdot I(x,y)$$
     $$x_{\text{bot}} = \frac{1}{\sum_{(x,y)\in \text{Bot}} I(x,y)} \sum_{(x,y)\in \text{Bot}} x \cdot I(x,y), \quad y_{\text{bot}} = \frac{1}{\sum_{(x,y)\in \text{Bot}} I(x,y)} \sum_{(x,y)\in \text{Bot}} y \cdot I(x,y)$$
   * Compute the slant angle $\theta$:
     $$\Delta x = x_{\text{top}} - x_{\text{bot}}, \quad \Delta y = y_{\text{bot}} - y_{\text{top}}$$
     $$\theta = \arctan\left( \frac{\Delta x}{\Delta y} \right) \quad (\text{in radians})$$
   * Edge-case guard: If $\Delta y \le 0$ or total ink in either half is zero, set $\theta = 0.0$ (no slant).
   * Clamp slant angle: Enforce $\theta \in [-45^\circ, +45^\circ]$ to prevent unnatural over-shearing.
2. **Affine Horizontal Shear Deskewing (Without Stroke Clipping):**
   * The horizontal shear parameter is $s = \tan(\theta)$.
   * To prevent the sheared character from being clipped at the image boundaries:
     * Compute the expansion padding: $\text{pad} = \lceil H \cdot |s| \rceil$.
     * Pad the character canvas horizontally with zeros by $\text{pad}$ pixels on both sides.
   * Construct the $2 \times 3$ affine transformation matrix:
     $$M_{\text{shear}} = \begin{bmatrix} 1 & -s & s \cdot y_{\text{bot}} + \text{pad} \\ 0 & 1 & 0 \end{bmatrix}$$
   * Apply affine transformation using `cv2.warpAffine(padded_crop, M_shear, (W_new, H), flags=cv2.INTER_LINEAR, borderValue=0)`.
   * Re-crop tightly to the deskewed foreground ink bounding box.
3. **Aspect-Ratio Preserving Resizing to Standard $16 \times 16$ Grid:**
   * Direct stretching to $16 \times 16$ destroys aspect ratio (turning '1' into a wide blob or 'm' into a tall block).
   * Define the active inner target area: $14 \times 14$ pixels (leaving a 1-pixel border pad to avoid boundary cutoff).
   * Let the deskewed crop have dimensions $h_d, w_d$:
     $$\text{scale} = \min\left( \frac{14}{h_d}, \; \frac{14}{w_d} \right)$$
     $$w_{\text{scaled}} = \text{round}(w_d \cdot \text{scale}), \quad h_{\text{scaled}} = \text{round}(h_d \cdot \text{scale})$$
   * Resize using area interpolation for anti-aliased downsampling:
     `resized = cv2.resize(deskewed_crop, (w_scaled, h_scaled), interpolation=cv2.INTER_AREA)`
4. **Centered Zero-Padded Embedding into $16 \times 16$ Canvas:**
   * Allocate a blank canvas: $\text{canvas} = \text{np.zeros}((16, 16), \text{dtype}=\text{np.float32})$.
   * Compute centering offsets:
     $$\text{offset}_x = \lfloor(16 - w_{\text{scaled}}) / 2\rfloor, \quad \text{offset}_y = \lfloor(16 - h_{\text{scaled}}) / 2\rfloor$$
   * Embed resized crop into canvas center:
     $$\text{canvas}[\text{offset}_y : \text{offset}_y + h_{\text{scaled}}, \; \text{offset}_x : \text{offset}_x + w_{\text{scaled}}] = \text{resized}$$
   * Ensure intensity normalization: Scale values strictly to range $[0.0, 1.0]$.

#### 3. Output Specification
* `normalized_char`: Standardized $16 \times 16$ single-channel NumPy array (`float32`, range $[0.0, 1.0]$).
* Metadata: estimated slant angle $\theta$ (in degrees), original bounding box coordinates.

#### 4. Acceptance Criteria & Edge Cases
* Every valid character produces a strictly identical $16 \times 16$ 2D array representation.
* Zero stroke clipping during shear transformation.
* Aspect ratio is preserved with centered padding.

---

### Phase 5: Circular DDD Handcrafted Feature Engine
**Module:** `features.py`  
**Purpose:** Implement the virtual circular-map tiling / wrap-around and 8-direction dual-set (W-set stroke/loop distance, B-set ink thickness) ray casting engine to extract a robust, rotation/scale-invariant handcrafted feature descriptor.

#### 1. Input Specification
* `normalized_char`: $16 \times 16$ normalized character image (`float32`, values $[0.0, 1.0]$) from Phase 4.

#### 2. Mathematical Definition & Algorithmic Steps
1. **Virtual Circular Domain & Coordinate Wrap-Around:**
   * Establish the center of the $16 \times 16$ grid at $(x_c, y_c) = (7.5, 7.5)$.
   * The circumscribed radius is $R_{\max} = 7.5$.
   * Pixels outside the inscribed radius are mapped using virtual circular tiling (polar wrap-around: $\theta \pmod{2\pi}$, and clamping $r \le R_{\max}$), ensuring boundary continuity and preventing edge artifacts.
2. **8-Directional Ray Casting Trajectory:**
   * Define 8 discrete directional unit vectors $\mathbf{u}_k$ corresponding to compass bearings:
     $$\mathbf{u}_k = \left( \cos\left(k \cdot \frac{\pi}{4}\right), \; \sin\left(k \cdot \frac{\pi}{4}\right) \right) \quad \text{for } k \in \{0, 1, 2, 3, 4, 5, 6, 7\}$$
     * $k=0: \text{East } (1, 0)$
     * $k=1: \text{North-East } (\frac{1}{\sqrt{2}}, \frac{1}{\sqrt{2}})$
     * $k=2: \text{North } (0, 1)$
     * $k=3: \text{North-West } (-\frac{1}{\sqrt{2}}, \frac{1}{\sqrt{2}})$
     * $k=4: \text{West } (-1, 0)$
     * $k=5: \text{South-West } (-\frac{1}{\sqrt{2}}, -\frac{1}{\sqrt{2}})$
     * $k=6: \text{South } (0, -1)$
     * $k=7: \text{South-East } (\frac{1}{\sqrt{2}}, -\frac{1}{\sqrt{2}})$
3. **Dual-Set Feature Extraction (W-Set vs B-Set):**
   * From a designated grid of sampling origins $P_i = (x_i, y_i)$ distributed across the circular character domain:
   * **W-Set (White / Structural Distance Features):**
     * Trace along ray $\mathbf{r}_k(t) = P_i + t \cdot \mathbf{u}_k$ for $t \ge 0$.
     * **Distance to First Stroke Edge ($d_{\text{first}}$):** Distance traveled through background ($I < 0.5$) before encountering foreground ink ($I \ge 0.5$). Captures outer structural envelope and concavities.
     * **Total White Space Span ($W_{\text{span}}$):** Cumulative length of background segments along ray. Captures internal loops (e.g. inside '0', '8', 'o', 'e') and open enclosures ('c', 'u').
   * **B-Set (Black / Thickness & Ink Distribution Features):**
     * **Continuous Stroke Thickness ($B_{\text{thick}}$):** Length of contiguous foreground ink segments along the ray ($I \ge 0.5$). Measures stroke width and junction density.
     * **Ink Pixel Density ($B_{\text{dens}}$):** Integrated grayscale intensity along ray:
       $$\int_0^{t_{\max}} I(\mathbf{r}_k(t)) \, dt$$
       Captures stroke mass and weight distribution.
4. **Partitioning & Output Vector Assembly (Achieving Exact 256 Dimensions):**
   * To satisfy the strict $256$-dimensional input requirement of the Phase 6 MLP:
     * Partition the $16 \times 16$ space into an $8 \times 8$ circular-zoned grid ($64$ anchor nodes).
     * At each node, compute 4 directional summary features (combining directional ray projections). $64 \text{ nodes} \times 4 \text{ features} = 256$ dimensions.
     * **OR (Multi-Ring Radial DDD):** 8 concentric radial rings $\times$ 8 directional rays $\times$ 4 dual-set metrics ($W_{\text{dist}}, W_{\text{span}}, B_{\text{thick}}, B_{\text{dens}}$) $= 8 \times 8 \times 4 = 256$ dimensions.
     * **OR (Direct Normalized Raster Mode):** Direct flattened $16 \times 16$ grayscale array $= 256$ dimensions.
   * Both implementations will be available in `features.py` with full unit testing, ensuring mathematical determinism and clean parameterization.
5. **Numerical Stability & Sanity Verification:**
   * Clamp all feature values to $[0.0, 1.0]$.
   * Assert $\text{len}(\text{feature\_vector}) == 256$.
   * Assert `not np.isnan(vec).any()` and `not np.isinf(vec).any()`.

#### 3. Output Specification
* `feature_vector`: 1D NumPy array (`float32`, shape `(256,)`, range $[0.0, 1.0]$).

#### 4. Acceptance Criteria & Edge Cases
* Feature extraction is 100% deterministic (identical crop produces identical vector).
* All outputs are finite real numbers (zero NaNs or Infs).
* Dimensionality is guaranteed to be exactly 256.

---

### Phase 6: Shallow MLP Classifier
**Module:** `train_mlp.py` (Training & Evaluation) and `predict.py` (Inference & Decoding)  
**Purpose:** Train, evaluate, and deploy a shallow, fully connected Multi-Layer Perceptron (no deep learning frameworks, CNNs, or Transformers) that ingests 256-D feature vectors and outputs class probabilities across EMNIST alphanumeric characters.

#### 1. Network Topology & Theoretical Architecture
* **Strict Architecture:** Fully connected feedforward MLP:
  $$\text{Input Layer } [256] \longrightarrow \text{Hidden Layer 1 } [100] \longrightarrow \text{Hidden Layer 2 } [90] \longrightarrow \text{Output Layer } [N]$$
  * $N = 47$ for EMNIST Balanced split (or $N = 62$ for EMNIST ByClass, $N = 26$ for EMNIST Letters).
* **Layer Details:**
  * **Input Layer:** 256 scalar inputs (matching Phase 5 feature dimension).
  * **Hidden Layer 1:** 100 neurons, ReLU activation ($f(z) = \max(0, z)$) or Logistic Sigmoid.
  * **Hidden Layer 2:** 90 neurons, ReLU activation.
  * **Output Layer:** $N$ neurons with Softmax activation:
    $$P(y = c \mid \mathbf{x}) = \frac{e^{z_c}}{\sum_{j=1}^N e^{z_j}}$$
* **Implementation Engine:**
  * `sklearn.neural_network.MLPClassifier(hidden_layer_sizes=(100, 90), activation='relu', solver='adam', max_iter=50, random_state=42)`
  * An optional transparent pure NumPy feedforward implementation in `predict.py` to demonstrate zero external runtime dependencies if desired.

#### 2. Dataset Ingestion, Preprocessing & Training Routine (`train_mlp.py`)
1. **Dataset Ingestion (Primary: EMNIST Balanced; Optional Baseline: MNIST):**
   * **Primary Dataset (EMNIST Balanced):** Ingest the 47-class EMNIST Balanced split (131,600 total images: 112,800 train, 18,800 test). Download and cache gzip/binary files into `data/emnist/`.
   * **Baseline Dataset (Standard MNIST):** When `--dataset mnist` is specified, ingest the 10-class digit dataset (70,000 images: 60,000 train, 10,000 test) for digit-only verification.
   * **Orientation Correction:** Raw EMNIST binary images are stored transposed and horizontally flipped relative to standard reading coordinates; the loader explicitly applies `np.transpose(img)` during parsing to ensure exact optical alignment with camera character crops.
2. **Feature Extraction Pipeline Consistency:**
   * Every training and testing sample from EMNIST is normalized through the exact same Phase 4/5 pipeline (scaled to $16 \times 16$, feature extracted to $256$-D).
   * **Critical Acceptance Rule:** Training and inference must share the identical feature extraction pipeline and feature scaler.
3. **Feature Scaling:**
   * Fit `StandardScaler` (or `MinMaxScaler`) on training feature matrices:
     $$\mathbf{x}_{\text{scaled}} = \frac{\mathbf{x} - \boldsymbol{\mu}}{\boldsymbol{\sigma} + \epsilon}$$
   * Save the fitted scaler to `models/feature_scaler.joblib`.
4. **Training & Hyperparameters:**
   * Split: $80\%$ Training, $20\%$ Validation / Test.
   * Batch size: 128 / 256.
   * Optimizer: Adam ($\beta_1=0.9, \beta_2=0.999$, learning rate $\eta = 0.001$).
   * Early stopping with patience $= 5$ epochs to prevent overfitting.
   * Save trained model to `models/emnist_mlp_model.joblib`.
5. **Evaluation Metrics:**
   * Compute overall test set accuracy.
   * Compute per-class precision, recall, and F1-score via `sklearn.metrics.classification_report`.
   * Save confusion matrix plot to `outputs/debug/confusion_matrix.png`.
   * Save numerical summary to `models/training_metrics.json`.

#### 3. Inference Engine (`predict.py`)
1. Ingest $256$-D feature vector.
2. Scale features using the loaded `feature_scaler.joblib`.
3. Perform forward propagation to obtain class probability distribution (Softmax scores).
4. Extract argmax class index: $\hat{y} = \arg\max_c P(y = c \mid \mathbf{x})$.
5. Retrieve corresponding ASCII character from `models/label_mapping.json`.
6. Extract prediction confidence score: $C = \max_c P(y = c \mid \mathbf{x}) \in [0.0, 1.0]$.
7. Treat confidence strictly as an algorithmic heuristic score, not a formal guarantee of correctness.

#### 4. Output Specification
* `predicted_char`: Single ASCII character string (e.g. `'A'`, `'5'`, `'x'`).
* `confidence`: Float in range $[0.0, 1.0]$.
* `all_probabilities`: 1D array of length $N$ with distribution over all classes.

#### 5. Acceptance Criteria & Edge Cases
* Zero deep learning / CNN / RNN code or dependencies present.
* Training and inference pipelines share the exact same scaler and feature dimensionality.
* Graceful handling of out-of-distribution or ambiguous character crops.

---

### Phase 7: Reconstruction and Export
**Modules:** `reconstruction.py`, `visualization.py`, and `main.py`  
**Purpose:** Reconstruct predicted character tokens into properly spaced words, lines, and paragraphs; export the digital text to a standard UTF-8 `.txt` file; render high-resolution visual bounding box overlays on the original image; and print a complete execution summary.

#### 1. Input Specification
* Segmented character crops with spatial coordinates $[x, y, w, h]$, word IDs, and line IDs from Phase 3.
* Predicted character labels and confidence scores from Phase 6.
* `original_image`: High-resolution untouched original image from Phase 1.

#### 2. Algorithmic Steps & Logic
1. **Spatial Text Assembly & Formatting:**
   * Traverse the hierarchy:
     * Lines are ordered top-to-bottom: $\text{Line}_0, \text{Line}_1, \dots$
     * Words within lines are ordered left-to-right: $\text{Word}_0, \text{Word}_1, \dots$
     * Characters within words are ordered left-to-right: $c_0, c_1, \dots$
   * Concatenate characters into words.
   * Separate words on the same line with a standard single space character (`' '`).
   * Separate consecutive lines with standard newline characters (`'\n'`).
   * Preserve paragraph line spacing where inter-line gaps exceed $1.8 \times \text{median line height}$ (`'\n\n'`).
2. **UTF-8 Plaintext Export:**
   * Write reconstructed text to `outputs/text/<image_basename>_recognized.txt`.
   * Explicitly enforce UTF-8 encoding: `open(..., 'w', encoding='utf-8')`.
3. **High-Resolution Annotated Visual Export (`visualization.py`):**
   * Load the untouched original input image.
   * If a perspective transform was applied in Phase 1, project bounding box coordinates back to the original image space via inverse homography $M^{-1}$:
     $$\mathbf{p}_{\text{orig}} = M^{-1} \mathbf{p}_{\text{flattened}}$$
     Alternatively, render overlays on the perspective-flattened canvas.
   * Render diagnostic overlays:
     * Draw green bounding rectangles around detected words (`thickness=2`).
     * Draw bounding boxes around recognized characters.
     * Render label HUD: Display predicted character and confidence score (e.g. `"A: 0.94"` or `"A"`).
     * Render with high legibility: Filled background rectangle behind text so labels remain readable against varied paper/desk tones.
   * Save annotated image to `outputs/annotated/<image_basename>_annotated.png`.
4. **Comprehensive Console Execution Summary:**
   * Print formatted CLI execution report:
     ```
     ======================================================================
     CAMERA-BASED HANDWRITING RECOGNITION PIPELINE EXECUTION SUMMARY
     ======================================================================
     Input Image              : D:/.../samples/test_handwriting.jpg
     Image Resolution         : 1920x1080 (3 channels)
     Perspective Transform    : Applied (Homography 3x3)
     Preprocessing Mode       : Morphological Bg Division + CLAHE + Sauvola
     Detected Text Lines      : 4
     Detected Words           : 18
     Detected Characters      : 86
     Mean Classification Conf : 0.887 (88.7%)
     Output Text File         : outputs/text/test_handwriting_recognized.txt
     Annotated Image File     : outputs/annotated/test_handwriting_annotated.png
     Total Pipeline Runtime   : 1.42s
     Warnings / Anomalies     : 0
     ======================================================================
     Reconstructed Text Preview:
     ----------------------------------------------------------------------
     The quick brown fox
     jumps over the lazy dog
     40928 HANDWRITING RECOGNITION
     ======================================================================
     ```

#### 3. Output Specification
* UTF-8 encoded text file in `outputs/text/`.
* High-resolution annotated image in `outputs/annotated/`.
* Structured CLI report with full execution telemetry.

#### 4. Acceptance Criteria & Edge Cases
* Reading order strictly follows natural document flow (top-to-bottom, left-to-right).
* Original image is preserved and cleanly annotated with readable labels.
* UTF-8 characters are encoded without corruption or replacement glyphs.

---

## 5. Configuration & Hyperparameter Architecture (`config.py`)

All thresholds, algorithm choices, kernel sizes, and file paths will be centralized in `config.py` to enable rapid parameter tuning without modifying core algorithmic code:

```python
"""
config.py - Centralized Configuration for Handwriting Recognition Pipeline
"""
import os
from pathlib import Path

# Base Paths
PROJECT_ROOT = Path(__file__).resolve().parent
MODELS_DIR = PROJECT_ROOT / "models"
OUTPUTS_DIR = PROJECT_ROOT / "outputs"
DATA_DIR = PROJECT_ROOT / "data"

# Phase 1: Ingestion
ENABLE_PERSPECTIVE_WARP = True       # Attempt 4-point paper contour detection
MIN_CONTOUR_AREA_RATIO = 0.20        # Minimum paper area relative to full image
MANUAL_PERSPECTIVE_CORNERS = None    # Optional explicit [(x1,y1), (x2,y2), (x3,y3), (x4,y4)]

# Phase 2: Preprocessing
BG_KERNEL_SIZE = 41                  # Morphological closing kernel for lighting estimation
CLAHE_CLIP_LIMIT = 2.0               # Contrast limiting factor for CLAHE
CLAHE_GRID_SIZE = (8, 8)             # Tile grid size for CLAHE
BINARIZATION_METHOD = "sauvola"      # "sauvola" or "gaussian_adaptive"
SAUVOLA_WINDOW_SIZE = 25             # Local window for Sauvola thresholding
SAUVOLA_K = 0.2                      # Sauvola sensitivity parameter (0.1 - 0.5)
SAUVOLA_R = 128                      # Sauvola dynamic range parameter

# Phase 3: Segmentation
HPP_SMOOTHING_SIGMA = 2.0            # 1D Gaussian smoothing for horizontal projection
LINE_VALLEY_THRESHOLD_RATIO = 0.05   # Valley detection ratio relative to max HPP
MIN_LINE_HEIGHT = 12                 # Minimum vertical height for a text line (pixels)
MIN_LINE_GAP = 6                     # Minimum gap between adjacent lines (pixels)
WORD_DILATION_WIDTH = 12             # Horizontal dilation kernel width for word grouping
MIN_CHAR_AREA = 16                   # Minimum connected component area (pixels^2)
MIN_CHAR_WIDTH = 4                   # Minimum character width
MIN_CHAR_HEIGHT = 6                  # Minimum character height
MAX_ASPECT_RATIO = 7.0               # Max width/height or height/width to reject scratches

# Phase 4: Normalization
INNER_TARGET_SIZE = (14, 14)         # Unpadded character size
FINAL_GRID_SIZE = (16, 16)           # Final standardized grid dimensions
MAX_SLANT_ANGLE_DEG = 45.0           # Maximum allowed shear deskew angle

# Phase 5: Features
FEATURE_MODE = "CIRCULAR_DDD_256"    # "CIRCULAR_DDD_256" or "FLAT_PIXEL_256"
DDD_NUM_DIRECTIONS = 8               # 8 compass directions
DDD_FEATURE_DIM = 256                # Exact MLP input dimension

# Phase 6: Shallow MLP Classifier
DATASET_NAME = "emnist"              # Primary: "emnist" (alphanumeric, 47 classes) | Baseline: "mnist" (digits only, 10 classes)
EMNIST_SPLIT = "balanced"            # "balanced" (47 classes), "bymerge", or "letters"
NUM_CLASSES = 47                     # Set to 47 for EMNIST Balanced; 10 for MNIST
MLP_ARCH = (100, 90)                 # Hidden layer topology: 256 -> 100 -> 90 -> N
BATCH_SIZE = 128
LEARNING_RATE = 0.001
MAX_EPOCHS = 50
EARLY_STOPPING_PATIENCE = 5

# Phase 7: Export & Visualization
DRAW_BOUNDING_BOXES = True
DRAW_CONFIDENCE_SCORES = True
ANNOTATION_COLOR = (0, 255, 0)       # Green (BGR)
```

---

## 6. Step-by-Step Implementation Roadmap & Schedule

The development lifecycle is divided into 7 distinct sequential milestones. Each milestone includes targeted unit testing and visual verification before proceeding to subsequent phases:

| Step | Milestone | Deliverable Files | Verification Method |
| :--- | :--- | :--- | :--- |
| **M1** | Environment & Infrastructure Setup | `requirements.txt`, `config.py`, directory scaffolding | Clean virtual environment, package import validation |
| **M2** | Ingestion & Preprocessing Modules | `ingestion.py`, `preprocessing.py`, `tests/test_ingestion.py`, `tests/test_preprocessing.py` | Load sample image, verify background division & Sauvola mask |
| **M3** | Hierarchical Segmentation Engine | `segmentation.py`, `tests/test_segmentation.py` | Line/Word/Char extraction, inspect `segmentation_hierarchy.png` |
| **M4** | Normalization & Circular DDD Engine | `normalization.py`, `features.py`, `tests/test_normalization.py`, `tests/test_features.py` | Check deskewing, verify deterministic 256-D feature vectors |
| **M5** | Shallow MLP Training & Evaluation | `train_mlp.py`, `models/emnist_mlp_model.joblib`, `models/label_mapping.json` | Train on EMNIST Balanced, verify test accuracy $> 80\%$ |
| **M6** | Inference, Reconstruction & Export | `predict.py`, `reconstruction.py`, `visualization.py` | Test end-to-end inference on sample handwriting images |
| **M7** | Main CLI Entry Point & Final Validation | `main.py`, `README.md`, full test suite execution | Run CLI command on camera capture, verify `.txt` and annotated image |

---

## 7. Testing & Quality Assurance Matrix

Every phase has automated test coverage in `tests/`:

1. **`test_ingestion.py`:**
   * Test with valid JPEG/PNG files.
   * Test with corrupted/missing files (assert proper exceptions).
   * Test perspective warp on synthetic document quad vs bypass behavior.
2. **`test_preprocessing.py`:**
   * Test grayscale conversion channel depth ($1$).
   * Test background division with synthetic uniform and gradient black images (assert zero divide-by-zero errors or NaNs).
   * Test Sauvola thresholding output range ($\{0, 255\}$).
3. **`test_segmentation.py`:**
   * Test on synthetic multi-line text image.
   * Verify top-to-bottom line ordering ($y_1 < y_2$).
   * Verify left-to-right word and character ordering.
   * Test noise filtering (verify speckles $< 15\text{px}^2$ are excluded).
4. **`test_normalization.py`:**
   * Test dual-centroid slant calculation on slanted synthetic vertical bar (verify angle sign and magnitude).
   * Test affine shear with zero stroke clipping.
   * Assert normalized character crop shape is strictly $(16, 16)$ with values in $[0.0, 1.0]$.
5. **`test_features.py`:**
   * Test feature vector output shape (strictly $(256,)$).
   * Test value ranges (all in $[0.0, 1.0]$).
   * Test determinism (identical crop generates identical vector).
   * Assert `np.isnan(vec).sum() == 0` and `np.isinf(vec).sum() == 0`.
6. **`test_mlp.py`:**
   * Test model loading from `models/emnist_mlp_model.joblib`.
   * Test inference on synthetic 256-D vectors.
   * Verify class index maps to valid ASCII character.
   * Verify confidence score is bounded in $[0.0, 1.0]$.
7. **`test_reconstruction.py`:**
   * Test text assembly with known mock tokens.
   * Verify correct space insertion between words and newlines between lines.
   * Verify UTF-8 file creation and byte integrity.

---

## 8. Theoretical & Literature Foundations

To ensure academic and algorithmic rigor, each design choice is mapped to foundational literature:

* **Oh & Suen (1998):** *"A class-modular neural network with rough classification for handwritten numeral recognition."*  
  * *Design Contribution:* Validates the use of modular shallow neural network classifiers and directional stroke feature representations over raw unsegmented bitmaps.
* **Katiyar & Mehfuz (2015):** *"A hybrid feature extraction technique for off-line handwritten character recognition."*  
  * *Design Contribution:* Supports combining structural directional distance distribution (DDD) metrics with spatial zoning to capture both stroke contour travel and local stroke thickness.
* **Wong & Sebastian (2015):** *"Handwritten character recognition using directional stroke features and neural networks."*  
  * *Design Contribution:* Provides the theoretical formulation for 8-direction ray casting and dual-set (W-set loop distance and B-set stroke thickness) feature extraction.
* **Sauvola & Pietikainen (2000):** *"Adaptive document image binarization."*  
  * *Design Contribution:* Foundational paper for local mean and standard deviation-based adaptive binarization, specifically engineered for degraded, unevenly illuminated paper documents.

---

## 9. Exact Terminal Execution Commands

### 9.1 Environment Setup
```powershell
# Create dedicated Python virtual environment
python -m venv .venv

# Activate virtual environment
.\.venv\Scripts\Activate.ps1

# Upgrade pip and install pinned dependencies
python -m pip install --upgrade pip
pip install -r requirements.txt
```

### 9.2 Training the Shallow MLP Classifier
```powershell
# [PRIMARY] Ingest EMNIST Balanced (alphanumeric, 47 classes), extract 256-D features, train 256 -> 100 -> 90 -> 47 MLP
python train_mlp.py --dataset emnist --split balanced --epochs 50 --batch-size 128

# [BASELINE / SANITY CHECK] Ingest standard MNIST (digits only, 10 classes), train 256 -> 100 -> 90 -> 10 MLP
python train_mlp.py --dataset mnist --epochs 50 --batch-size 128
```

### 9.3 Running Inference on a Camera Capture
```powershell
# Run the complete end-to-end pipeline on an input camera capture or photo
python main.py --input data/samples/sample_handwriting.jpg --output-dir outputs/

# Run with optional manual perspective correction
python main.py --input data/samples/skewed_page.jpg --warp

# Run in debug mode (saves all intermediate masks, projections, and crops)
python main.py --input data/samples/sample_handwriting.jpg --debug
```

### 9.4 Running the Automated Test Suite
```powershell
# Execute all stage-level unit tests
pytest tests/ -v
```
