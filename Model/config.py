"""
config.py - Centralized Configuration for Handwriting Recognition Pipeline

Centralizes all hyperparameters, directory paths, and thresholds across
the 7-phase classical computer vision and shallow MLP OCR pipeline.
"""
from pathlib import Path

# -------------------------------------------------------------------------
# Base Paths & Directory Layout
# -------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent
WORKSPACE_ROOT = PROJECT_ROOT.parent if PROJECT_ROOT.name == "Model" else PROJECT_ROOT
DATA_DIR = WORKSPACE_ROOT / "data"
EMNIST_DIR = DATA_DIR / "emnist"
SAMPLES_DIR = DATA_DIR / "samples"

MODELS_DIR = PROJECT_ROOT / "models" if (PROJECT_ROOT / "models").exists() else WORKSPACE_ROOT / "models"
OUTPUTS_DIR = PROJECT_ROOT / "outputs" if (PROJECT_ROOT / "outputs").exists() else WORKSPACE_ROOT / "outputs"
TEXT_OUTPUT_DIR = OUTPUTS_DIR / "text"
ANNOTATED_OUTPUT_DIR = OUTPUTS_DIR / "annotated"
DEBUG_OUTPUT_DIR = OUTPUTS_DIR / "debug"

# Ensure all runtime directory scaffolds exist
for _dir in [DATA_DIR, EMNIST_DIR, SAMPLES_DIR, MODELS_DIR, OUTPUTS_DIR, TEXT_OUTPUT_DIR, ANNOTATED_OUTPUT_DIR, DEBUG_OUTPUT_DIR]:
    _dir.mkdir(parents=True, exist_ok=True)

# -------------------------------------------------------------------------
# Phase 1: Ingestion and Document Flattening
# -------------------------------------------------------------------------
ENABLE_PERSPECTIVE_WARP = True       # Attempt 4-point paper contour detection
MIN_CONTOUR_AREA_RATIO = 0.20        # Minimum paper area relative to full image
MANUAL_PERSPECTIVE_CORNERS = None    # Optional explicit [(x1,y1), (x2,y2), (x3,y3), (x4,y4)]
MAX_INGESTION_PROC_DIM = 1024        # Downscale max dimension for contour detection

# -------------------------------------------------------------------------
# Phase 2: Illumination-Robust Preprocessing
# -------------------------------------------------------------------------
BG_KERNEL_SIZE = 41                  # Morphological closing kernel for lighting estimation
CLAHE_CLIP_LIMIT = 2.0               # Contrast limiting factor for CLAHE
CLAHE_GRID_SIZE = (8, 8)             # Tile grid size for CLAHE
BINARIZATION_METHOD = "sauvola"      # "sauvola" or "gaussian_adaptive"
SAUVOLA_WINDOW_SIZE = 25             # Local window for Sauvola thresholding
SAUVOLA_K = 0.2                      # Sauvola sensitivity parameter (0.1 - 0.5)
SAUVOLA_R = 128                      # Sauvola dynamic range parameter
DESPECKLE_METHOD = "median"          # "median", "morph_open", or None
DESPECKLE_KERNEL_SIZE = 3            # Window size for despeckling noise suppression

# -------------------------------------------------------------------------
# Phase 3: Hierarchical Segmentation (Line -> Word -> Character)
# -------------------------------------------------------------------------
HPP_SMOOTHING_SIGMA = 2.0            # 1D Gaussian smoothing for horizontal projection
LINE_VALLEY_THRESHOLD_RATIO = 0.05   # Valley detection ratio relative to max HPP
MIN_LINE_HEIGHT = 12                 # Minimum vertical height for a text line (pixels)
MIN_LINE_GAP = 6                     # Gap below which consecutive lines are merged (pixels)
WORD_DILATION_KERNEL_W = 15          # Width of horizontal dilation kernel for word bridging
WORD_GAP_MULTIPLIER = 1.2            # Inter-word space threshold multiplier vs median char gap
MIN_CHAR_AREA = 15                   # Minimum pixel area for valid character CC (noise filter)
MIN_CHAR_WIDTH = 3                   # Minimum character bounding box width (pixels)
MIN_CHAR_HEIGHT = 5                  # Minimum character bounding box height (pixels)
MAX_CHAR_ASPECT_RATIO = 8.0          # Filter horizontal ruling lines or vertical scratches
MAX_ASPECT_RATIO_SPLIT = 1.8         # Split wide connected components (joined characters)
OVERLAP_SPLIT_THRESHOLD = 1.8        # Ratio against median width to trigger ligature split

# -------------------------------------------------------------------------
# Phase 4: Geometric Normalization
# -------------------------------------------------------------------------
NORMALIZED_CHAR_SIZE = (16, 16)      # Standard input grid size (height, width)
INNER_TARGET_SIZE = (14, 14)          # Active character size within canvas
SLANT_CORRECTION_ENABLED = True      # Apply dual-centroid shear correction
MAX_SLANT_ANGLE_DEG = 45.0           # Maximum plausible handwriting slant angle
MIN_INK_PIXELS_FOR_SLANT = 5         # Minimum pixels in top/bottom zones for slant calculation

# -------------------------------------------------------------------------
# Phase 5: Circular DDD Handcrafted Feature Engine
# -------------------------------------------------------------------------
FEATURE_MODE = "circular_ddd"        # "circular_ddd" (256-D) or "flatten_16x16" (256-D)
DDD_NUM_DIRECTIONS = 8               # Ray-casting directional count (8 compass directions)
DDD_NUM_RINGS = 8                    # Concentric radial sampling rings (8 * 8 * 4 = 256)
DDD_RAY_STEP_SIZE = 0.5              # Ray-casting integration step size (pixels)
DDD_MAX_RAY_STEPS = 15               # Maximum sampling steps along each ray trajectory
FEATURE_VECTOR_DIM = 256             # Strict input dimension for shallow MLP

# -------------------------------------------------------------------------
# Phase 6: Shallow MLP Classifier
# -------------------------------------------------------------------------
MLP_HIDDEN_LAYERS = (100, 90)        # Fully connected hidden layer topology
MLP_MAX_ITER = 50                    # Training epochs
MLP_BATCH_SIZE = 128                 # Mini-batch size
MLP_LEARNING_RATE_INIT = 0.001       # Initial Adam learning rate
MLP_RANDOM_STATE = 42                # Reproducibility seed
EMNIST_SPLIT = "balanced"            # 47 classes (balanced digits + uppercase + lowercase)
MODEL_FILE = MODELS_DIR / "emnist_mlp_model.joblib"
SCALER_FILE = MODELS_DIR / "feature_scaler.joblib"
LABEL_MAP_FILE = MODELS_DIR / "label_mapping.json"
METRICS_FILE = MODELS_DIR / "training_metrics.json"
CONFUSION_MATRIX_FILE = DEBUG_OUTPUT_DIR / "confusion_matrix.png"

# -------------------------------------------------------------------------
# Phase 7: Reconstruction & Visual Export
# -------------------------------------------------------------------------
OUTPUT_TEXT_ENCODING = "utf-8"
DRAW_BOUNDING_BOXES = True
DRAW_CONFIDENCE_SCORES = True
ANNOTATION_COLOR = (0, 255, 0)       # Green (BGR)
