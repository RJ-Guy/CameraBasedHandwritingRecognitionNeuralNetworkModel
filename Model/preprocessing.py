"""
preprocessing.py - Phase 2: Illumination-Robust Preprocessing

This module implements classical computer vision techniques to eliminate
uneven lighting, directional shadows, paper creases, and phone camera exposure
gradients. It produces a contrast-enhanced grayscale representation and a clean,
inverted binary mask (255 = ink, 0 = paper background) for downstream segmentation.

Pipeline Stages:
1. Grayscale Conversion (cv2.COLOR_BGR2GRAY).
2. Morphological Background Illumination Estimation (Morphological Closing with Ellipse Kernel).
3. Safe Background Division & Normalization (Cancels lighting gradients without divide-by-zero).
4. Local Contrast Enhancement (Contrast Limited Adaptive Histogram Equalization - CLAHE).
5. Adaptive Binarization (Sauvola thresholding as primary, Gaussian Adaptive as fallback).
6. Despeckling & Noise Suppression (Median blur to remove microscopic dust without eroding strokes).
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional, Tuple, Union
import cv2
import numpy as np

try:
    from . import config
except ImportError:
    import config


@dataclass
class PreprocessingResult:
    """Container holding intermediate and final outputs of Phase 2 preprocessing."""
    gray_image: np.ndarray           # Grayscale representation
    background_estimate: np.ndarray  # Estimated background illumination envelope
    normalized_image: np.ndarray     # Background-divided normalized image
    enhanced_image: np.ndarray       # CLAHE contrast-enhanced image
    binary_mask: np.ndarray          # Inverted binary mask (255 = ink, 0 = paper)

    def to_dict(self) -> Dict[str, np.ndarray]:
        """Convert result fields to a standard dictionary."""
        return {
            "gray_image": self.gray_image,
            "background_estimate": self.background_estimate,
            "normalized_image": self.normalized_image,
            "enhanced_image": self.enhanced_image,
            "binary_mask": self.binary_mask,
        }


def convert_to_grayscale(image: np.ndarray) -> np.ndarray:
    """
    Converts an input image to single-channel 8-bit grayscale.

    Args:
        image: BGR color image (H, W, 3) or grayscale image (H, W).

    Returns:
        2D uint8 numpy array of shape (H, W).
    """
    if image is None or image.size == 0:
        raise ValueError("[Phase 2 Error] Input image is empty or None.")

    if len(image.shape) == 2:
        return image.copy().astype(np.uint8)

    if len(image.shape) == 3:
        if image.shape[2] == 1:
            return image[:, :, 0].copy().astype(np.uint8)
        elif image.shape[2] == 3:
            return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        elif image.shape[2] == 4:
            return cv2.cvtColor(image, cv2.COLOR_BGRA2GRAY)

    raise ValueError(f"[Phase 2 Error] Unsupported image shape for grayscale conversion: {image.shape}")


def compute_bg_kernel_size(
    height: int,
    width: int,
    base_size: int = config.BG_KERNEL_SIZE,
    dynamic: bool = True,
) -> int:
    """
    Computes an appropriate odd kernel size for morphological background estimation.

    Scaled dynamically with image resolution:
        k = 2 * floor(min(w, h) / 40) + 1
    ensuring k is an odd integer and strictly fits within image dimensions.

    Args:
        height: Image height in pixels.
        width: Image width in pixels.
        base_size: Default base kernel size from configuration.
        dynamic: Whether to dynamically scale kernel with resolution.

    Returns:
        Odd integer kernel dimension >= 3.
    """
    min_dim = min(height, width)
    if min_dim < 3:
        raise ValueError(f"[Phase 2 Error] Image dimensions ({width}x{height}) too small for background estimation.")

    if dynamic:
        dyn_k = 2 * (min_dim // 40) + 1
        k = max(base_size, dyn_k)
    else:
        k = base_size

    # Ensure odd
    if k % 2 == 0:
        k += 1

    # Ensure strictly smaller than minimum dimension
    if k >= min_dim:
        k = (min_dim // 2) * 2 - 1
        k = max(3, k)

    return k


def estimate_background(
    gray: np.ndarray,
    kernel_size: Optional[int] = None,
    dynamic: bool = True,
) -> np.ndarray:
    """
    Estimates the low-frequency background illumination surface using morphological closing.

    A large elliptical structuring element bridges text strokes and captures the underlying
    paper surface lighting, shadows, and vignette roll-off.

    Args:
        gray: 2D uint8 grayscale image.
        kernel_size: Explicit kernel dimension (if None, computed dynamically).
        dynamic: Whether to dynamically scale kernel size with resolution.

    Returns:
        2D uint8 estimated background illumination surface.
    """
    h, w = gray.shape[:2]
    if kernel_size is None:
        k_size = compute_bg_kernel_size(h, w, base_size=config.BG_KERNEL_SIZE, dynamic=dynamic)
    else:
        k_size = kernel_size
        if k_size % 2 == 0:
            k_size += 1
        k_size = min(k_size, max(3, (min(h, w) // 2) * 2 - 1))

    # Construct elliptical structuring element
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k_size, k_size))

    # Morphological closing: Dilation followed by Erosion
    background = cv2.morphologyEx(gray, cv2.MORPH_CLOSE, kernel)
    return background


def normalize_illumination(
    gray: np.ndarray,
    background: np.ndarray,
    epsilon: float = 1.0,
) -> np.ndarray:
    """
    Cancels uneven lighting by dividing the grayscale image by the estimated background surface.

    Mathematical formulation:
        I_norm(x, y) = clip( ((I_gray + epsilon) / (I_bg + epsilon)) * 255.0, 0, 255 )

    The epsilon parameter strictly prevents division by zero and numerical instability.

    Args:
        gray: 2D uint8 grayscale image.
        background: 2D uint8 background illumination estimate.
        epsilon: Small positive constant preventing divide-by-zero.

    Returns:
        2D uint8 lighting-normalized image.
    """
    if gray.shape != background.shape:
        raise ValueError(
            f"[Phase 2 Error] Shape mismatch: gray {gray.shape} vs background {background.shape}"
        )

    gray_f = gray.astype(np.float32)
    bg_f = background.astype(np.float32)

    # Safe element-wise division and scaling
    normalized = np.clip(((gray_f + epsilon) / (bg_f + epsilon)) * 255.0, 0.0, 255.0)

    # Check for any unexpected NaN or Inf
    if not np.isfinite(normalized).all():
        normalized = np.nan_to_num(normalized, nan=255.0, posinf=255.0, neginf=0.0)

    return normalized.astype(np.uint8)


def enhance_contrast_clahe(
    image: np.ndarray,
    clip_limit: float = config.CLAHE_CLIP_LIMIT,
    grid_size: Tuple[int, int] = config.CLAHE_GRID_SIZE,
) -> np.ndarray:
    """
    Applies Contrast Limited Adaptive Histogram Equalization (CLAHE) to locally boost ink contrast.

    Unlike global histogram equalization, CLAHE operates on localized contextual tiles,
    preventing over-amplification of noise while sharpening fine handwriting strokes.

    Args:
        image: 2D uint8 grayscale or illumination-normalized image.
        clip_limit: Contrast clipping limit (default 2.0).
        grid_size: Tile grid dimensions (rows, cols) (default (8, 8)).

    Returns:
        2D uint8 contrast-enhanced image.
    """
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=grid_size)
    return clahe.apply(image)


def binarize_sauvola(
    image: np.ndarray,
    window_size: int = config.SAUVOLA_WINDOW_SIZE,
    k: float = config.SAUVOLA_K,
    r: float = config.SAUVOLA_R,
) -> np.ndarray:
    """
    Applies Sauvola local adaptive binarization (Sauvola & Pietikainen, 2000).

    Computes local threshold for each pixel based on local mean m(x, y) and standard deviation s(x, y):
        T(x, y) = m(x, y) * [ 1 + k * (s(x, y) / R - 1) ]

    Ink pixels (dark on light background) satisfy I(x, y) <= T(x, y).
    Returns an inverted binary mask where:
        255 = Ink (Foreground)
        0   = Paper (Background)

    Vectorized implementation using OpenCV boxFilter for high performance and zero Python loops.

    Args:
        image: 2D uint8 contrast-enhanced grayscale image.
        window_size: Odd integer local neighborhood window size (default 25).
        k: Sensitivity parameter controlling stroke thickness (0.1 - 0.5, default 0.2).
        r: Dynamic range parameter (default 128 for 8-bit images).

    Returns:
        2D uint8 inverted binary mask with values strictly in {0, 255}.
    """
    if window_size % 2 == 0:
        window_size += 1

    h, w = image.shape[:2]
    # Bound window size to image dimensions
    window_size = min(window_size, max(3, (min(h, w) // 2) * 2 - 1))

    img_f = image.astype(np.float32)

    # Compute local mean m(x, y)
    mean = cv2.boxFilter(img_f, cv2.CV_32F, (window_size, window_size), borderType=cv2.BORDER_REFLECT)

    # Compute local mean of squares m2(x, y)
    sqmean = cv2.boxFilter(img_f ** 2, cv2.CV_32F, (window_size, window_size), borderType=cv2.BORDER_REFLECT)

    # Local variance: Var(X) = E[X^2] - (E[X])^2
    variance = np.maximum(sqmean - mean ** 2, 0.0)
    std = np.sqrt(variance)

    # Sauvola local threshold formula
    threshold = mean * (1.0 + k * (std / float(r) - 1.0))

    # Inverted binarization: ink is dark in grayscale (<= threshold), output as 255
    binary_mask = np.where(img_f <= threshold, 255, 0).astype(np.uint8)
    return binary_mask


def binarize_gaussian_adaptive(
    image: np.ndarray,
    block_size: int = 25,
    c: int = 11,
) -> np.ndarray:
    """
    Fast fallback binarization using Gaussian adaptive thresholding.

    Returns an inverted binary mask where:
        255 = Ink (Foreground)
        0   = Paper (Background)

    Args:
        image: 2D uint8 contrast-enhanced grayscale image.
        block_size: Odd neighborhood window size.
        c: Constant subtracted from weighted mean.

    Returns:
        2D uint8 inverted binary mask with values in {0, 255}.
    """
    if block_size % 2 == 0:
        block_size += 1
    h, w = image.shape[:2]
    block_size = min(block_size, max(3, (min(h, w) // 2) * 2 - 1))

    return cv2.adaptiveThreshold(
        image,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV,
        blockSize=block_size,
        C=c,
    )


def despeckle_mask(
    binary_mask: np.ndarray,
    method: Optional[str] = config.DESPECKLE_METHOD,
    kernel_size: int = config.DESPECKLE_KERNEL_SIZE,
) -> np.ndarray:
    """
    Suppresses microscopic dust speckles and sensor noise without eroding thin pen strokes.

    Args:
        binary_mask: 2D uint8 inverted binary mask (255 = ink).
        method: "median" (preserves thin strokes while removing salt/pepper),
                "morph_open" (morphological opening), or None/empty.
        kernel_size: Odd integer kernel dimension (default 3).

    Returns:
        2D uint8 cleaned binary mask.
    """
    if method is None or str(method).lower() in ("none", "disabled", "false"):
        return binary_mask.copy()

    method_lower = str(method).lower()

    if method_lower == "median":
        if kernel_size % 2 == 0:
            kernel_size += 1
        kernel_size = max(3, kernel_size)
        return cv2.medianBlur(binary_mask, kernel_size)

    elif method_lower in ("morph_open", "opening"):
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_size, kernel_size))
        return cv2.morphologyEx(binary_mask, cv2.MORPH_OPEN, kernel)

    else:
        # Fallback to copy if method is unrecognized
        return binary_mask.copy()


def preprocess_image(
    ingested_image: np.ndarray,
    method: str = config.BINARIZATION_METHOD,
    bg_kernel_size: Optional[int] = None,
    dynamic_bg: bool = True,
    clahe_clip_limit: float = config.CLAHE_CLIP_LIMIT,
    clahe_grid_size: Tuple[int, int] = config.CLAHE_GRID_SIZE,
    sauvola_window_size: int = config.SAUVOLA_WINDOW_SIZE,
    sauvola_k: float = config.SAUVOLA_K,
    sauvola_r: float = config.SAUVOLA_R,
    despeckle_method: Optional[str] = config.DESPECKLE_METHOD,
    despeckle_kernel_size: int = config.DESPECKLE_KERNEL_SIZE,
) -> PreprocessingResult:
    """
    End-to-end Phase 2 illumination-robust preprocessing pipeline.

    Processes the input BGR/grayscale image through:
    1. Grayscale Conversion
    2. Morphological Background Illumination Surface Estimation
    3. Safe Background Division & Normalization
    4. Contrast-Limited Adaptive Histogram Equalization (CLAHE)
    5. Adaptive Binarization (Sauvola or Gaussian Adaptive)
    6. Despeckling Noise Suppression

    Args:
        ingested_image: Input document image (BGR or grayscale) from Phase 1.
        method: Binarization method ("sauvola" or "gaussian_adaptive").
        bg_kernel_size: Optional fixed kernel size for background estimation.
        dynamic_bg: Whether to dynamically scale background kernel with resolution.
        clahe_clip_limit: Contrast clip limit for CLAHE.
        clahe_grid_size: Tile grid dimensions for CLAHE.
        sauvola_window_size: Local window size for Sauvola thresholding.
        sauvola_k: Sensitivity parameter for Sauvola thresholding.
        sauvola_r: Dynamic range parameter for Sauvola thresholding.
        despeckle_method: Noise suppression method ("median", "morph_open", or None).
        despeckle_kernel_size: Despeckling kernel size.

    Returns:
        PreprocessingResult dataclass containing:
        - gray_image: Grayscale version of document.
        - background_estimate: Morphological background image I_bg.
        - normalized_image: Background-divided normalized image I_norm.
        - enhanced_image: Contrast-equalized image I_clahe.
        - binary_mask: Inverted binary image (255 = ink, 0 = paper) ready for segmentation.
    """
    # 1. Grayscale conversion
    gray = convert_to_grayscale(ingested_image)

    # 2. Morphological background illumination estimation
    bg_estimate = estimate_background(gray, kernel_size=bg_kernel_size, dynamic=dynamic_bg)

    # 3. Safe background division & normalization
    normalized = normalize_illumination(gray, bg_estimate, epsilon=1.0)

    # 4. Local contrast enhancement via CLAHE
    enhanced = enhance_contrast_clahe(normalized, clip_limit=clahe_clip_limit, grid_size=clahe_grid_size)

    # 5. Adaptive binarization
    method_lower = str(method).lower()
    if method_lower == "sauvola":
        raw_binary = binarize_sauvola(
            enhanced,
            window_size=sauvola_window_size,
            k=sauvola_k,
            r=sauvola_r,
        )
    elif method_lower in ("gaussian_adaptive", "adaptive", "gaussian"):
        raw_binary = binarize_gaussian_adaptive(enhanced, block_size=sauvola_window_size)
    else:
        raise ValueError(
            f"[Phase 2 Error] Unknown binarization method: '{method}'. "
            f"Expected 'sauvola' or 'gaussian_adaptive'."
        )

    # 6. Despeckling noise suppression
    clean_binary = despeckle_mask(
        raw_binary,
        method=despeckle_method,
        kernel_size=despeckle_kernel_size,
    )

    return PreprocessingResult(
        gray_image=gray,
        background_estimate=bg_estimate,
        normalized_image=normalized,
        enhanced_image=enhanced,
        binary_mask=clean_binary,
    )


class PreprocessingEngine:
    """
    Object-oriented encapsulation of Phase 2 Preprocessing for modular pipeline execution.
    """

    def __init__(
        self,
        method: str = config.BINARIZATION_METHOD,
        bg_kernel_size: Optional[int] = None,
        dynamic_bg: bool = True,
        clahe_clip_limit: float = config.CLAHE_CLIP_LIMIT,
        clahe_grid_size: Tuple[int, int] = config.CLAHE_GRID_SIZE,
        sauvola_window_size: int = config.SAUVOLA_WINDOW_SIZE,
        sauvola_k: float = config.SAUVOLA_K,
        sauvola_r: float = config.SAUVOLA_R,
        despeckle_method: Optional[str] = config.DESPECKLE_METHOD,
        despeckle_kernel_size: int = config.DESPECKLE_KERNEL_SIZE,
    ):
        self.method = method
        self.bg_kernel_size = bg_kernel_size
        self.dynamic_bg = dynamic_bg
        self.clahe_clip_limit = clahe_clip_limit
        self.clahe_grid_size = clahe_grid_size
        self.sauvola_window_size = sauvola_window_size
        self.sauvola_k = sauvola_k
        self.sauvola_r = sauvola_r
        self.despeckle_method = despeckle_method
        self.despeckle_kernel_size = despeckle_kernel_size

    def process(self, image: np.ndarray) -> PreprocessingResult:
        """Processes an image through the Phase 2 pipeline."""
        return preprocess_image(
            ingested_image=image,
            method=self.method,
            bg_kernel_size=self.bg_kernel_size,
            dynamic_bg=self.dynamic_bg,
            clahe_clip_limit=self.clahe_clip_limit,
            clahe_grid_size=self.clahe_grid_size,
            sauvola_window_size=self.sauvola_window_size,
            sauvola_k=self.sauvola_k,
            sauvola_r=self.sauvola_r,
            despeckle_method=self.despeckle_method,
            despeckle_kernel_size=self.despeckle_kernel_size,
        )
