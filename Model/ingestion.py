"""
ingestion.py - Phase 1: Image Ingestion and Document Flattening

This module handles:
1. Robust image ingestion from static camera captures or phone photos.
2. Untouched preservation of the raw input image for visual audit and HUD overlays.
3. Document quadrilateral boundary detection via edge detection and polygon approximation.
4. Deterministic 4-point perspective warp rectification (deskewing and flattening).
5. Automatic graceful bypass when page boundaries are not detected or warp is disabled.
6. Manual corner coordinate fallback support.
"""

from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
import cv2
import numpy as np

try:
    from . import config
except ImportError:
    import config


def load_image(file_path: Union[str, Path]) -> Tuple[np.ndarray, np.ndarray]:
    """
    Robustly loads an image from disk using OpenCV, reporting missing or
    corrupted files with clear, actionable exceptions.

    Args:
        file_path: Path to the image file (JPG, PNG, BMP, TIFF, etc.).

    Returns:
        Tuple of (original_image_copy, raw_frame), where original_image_copy
        is an untouched deep copy preserving pixel data.

    Raises:
        FileNotFoundError: If the file does not exist on disk.
        ValueError: If OpenCV fails to decode the image file.
    """
    path_obj = Path(file_path)
    if not path_obj.exists():
        raise FileNotFoundError(f"[Phase 1 Error] Image file not found: '{file_path}'")
    
    # Read image preserving BGR color channels
    raw_frame = cv2.imread(str(path_obj), cv2.IMREAD_COLOR)
    if raw_frame is None or raw_frame.size == 0:
        raise ValueError(
            f"[Phase 1 Error] Failed to decode image. File may be empty, "
            f"corrupted, or in an unsupported format: '{file_path}'"
        )
    
    # Deep copy for pristine overlay preservation
    original_image = raw_frame.copy()
    return original_image, raw_frame


def order_quad_points(pts: np.ndarray) -> np.ndarray:
    """
    Deterministically orders 4 quadrilateral points in standard sequence:
    [Top-Left, Top-Right, Bottom-Right, Bottom-Left].

    Mathematical formulation:
    - Top-Left minimizes (x + y)
    - Bottom-Right maximizes (x + y)
    - Top-Right maximizes (x - y)
    - Bottom-Left minimizes (x - y)

    Args:
        pts: Array of shape (4, 2) containing unordered (x, y) coordinates.

    Returns:
        Array of shape (4, 2) with points ordered [TL, TR, BR, BL] as float32.
    """
    pts = np.asarray(pts, dtype=np.float32).reshape((4, 2))
    rect = np.zeros((4, 2), dtype=np.float32)

    # Sum of coordinates: s = x + y
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]  # Top-Left
    rect[2] = pts[np.argmax(s)]  # Bottom-Right

    # Difference of coordinates: d = x - y
    d = pts[:, 0] - pts[:, 1]
    rect[1] = pts[np.argmax(d)]  # Top-Right
    rect[3] = pts[np.argmin(d)]  # Bottom-Left

    return rect


def detect_document_corners(
    image: np.ndarray,
    min_area_ratio: float = config.MIN_CONTOUR_AREA_RATIO,
    max_proc_dim: int = config.MAX_INGESTION_PROC_DIM,
) -> Optional[np.ndarray]:
    """
    Detects paper document quadrilateral corners in an overhead or desk photograph.

    Pipeline:
    1. Downscales image if resolution exceeds max_proc_dim for speed and noise reduction.
    2. Converts to grayscale and applies Gaussian smoothing (5x5, sigma=1.0).
    3. Computes Canny edges and applies morphological dilation to close broken boundaries.
    4. Finds external contours, approximates polygons via Douglas-Peucker (epsilon = 0.02 * peri).
    5. Filters for convex 4-point quadrilaterals exceeding the minimum area ratio.
    6. Rescales detected corner coordinates back to the original full image resolution.

    Args:
        image: BGR image array.
        min_area_ratio: Minimum polygon area relative to total image area.
        max_proc_dim: Maximum processing dimension for contour finding.

    Returns:
        Ordered array of shape (4, 2) with corners [TL, TR, BR, BL] on original image scale,
        or None if no valid document quadrilateral is detected.
    """
    h, w = image.shape[:2]
    max_dim = max(h, w)

    # Compute downscaling ratio for efficient contour analysis
    scale = 1.0
    if max_dim > max_proc_dim:
        scale = max_proc_dim / float(max_dim)
        proc_w = int(w * scale)
        proc_h = int(h * scale)
        proc_image = cv2.resize(image, (proc_w, proc_h), interpolation=cv2.INTER_AREA)
    else:
        proc_image = image

    # 1. Grayscale conversion
    gray = cv2.cvtColor(proc_image, cv2.COLOR_BGR2GRAY)

    # 2. Gaussian smoothing to reduce sensor grain & paper texture
    blurred = cv2.GaussianBlur(gray, (5, 5), 1.0)

    # 3. Canny edge detection
    edges = cv2.Canny(blurred, 50, 150)

    # 4. Morphological dilation to bridge broken perimeter edges
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    dilated_edges = cv2.dilate(edges, kernel, iterations=1)

    # 5. Extract external contours
    contours, _ = cv2.findContours(dilated_edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None

    # Sort contours by area descending
    sorted_contours = sorted(contours, key=cv2.contourArea, reverse=True)
    proc_total_area = proc_image.shape[0] * proc_image.shape[1]

    for contour in sorted_contours:
        peri = cv2.arcLength(contour, closed=True)
        approx = cv2.approxPolyDP(contour, epsilon=0.02 * peri, closed=True)

        # Look for convex 4-corner polygon
        if len(approx) == 4 and cv2.isContourConvex(approx):
            contour_area = cv2.contourArea(approx)
            if contour_area >= (min_area_ratio * proc_total_area):
                # Rescale detected points back to original image coordinates
                pts = approx.reshape(4, 2).astype(np.float32)
                if scale != 1.0:
                    pts = pts / scale
                return order_quad_points(pts)

    return None


def warp_perspective(
    image: np.ndarray,
    corners: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Applies 4-point homography perspective warp to rectify a tilted document quadrilateral.

    Calculates target dimensions:
    - Width: max(distance(TL, TR), distance(BL, BR))
    - Height: max(distance(TL, BL), distance(TR, BR))

    Args:
        image: Original input BGR image.
        corners: Ordered (4, 2) corner array [TL, TR, BR, BL].

    Returns:
        Tuple of (rectified_image, homography_matrix_M).
    """
    rect = order_quad_points(corners)
    tl, tr, br, bl = rect

    # Calculate target rectified width
    width_top = np.linalg.norm(tr - tl)
    width_bottom = np.linalg.norm(br - bl)
    target_w = max(int(round(width_top)), int(round(width_bottom)))

    # Calculate target rectified height
    height_left = np.linalg.norm(bl - tl)
    height_right = np.linalg.norm(br - tr)
    target_h = max(int(round(height_left)), int(round(height_right)))

    # Guard against degenerately small regions
    target_w = max(target_w, 32)
    target_h = max(target_h, 32)

    # Destination rectangular coordinates
    dst = np.array(
        [
            [0, 0],
            [target_w - 1, 0],
            [target_w - 1, target_h - 1],
            [0, target_h - 1],
        ],
        dtype=np.float32,
    )

    # Compute 3x3 homography matrix and warp
    M = cv2.getPerspectiveTransform(rect, dst)
    flattened = cv2.warpPerspective(
        image,
        M,
        (target_w, target_h),
        flags=cv2.INTER_CUBIC,
        borderMode=cv2.BORDER_REPLICATE,
    )

    return flattened, M


def ingest_image(
    image_input: Union[str, Path, np.ndarray],
    enable_warp: Optional[bool] = None,
    min_area_ratio: Optional[float] = None,
    manual_corners: Optional[List[Tuple[float, float]]] = None,
) -> Tuple[np.ndarray, np.ndarray, Dict]:
    """
    End-to-end Phase 1 optical ingestion pipeline.

    Args:
        image_input: Path to file or pre-loaded BGR numpy array.
        enable_warp: Whether to attempt document corner detection and perspective warp.
                     Defaults to config.ENABLE_PERSPECTIVE_WARP.
        min_area_ratio: Minimum document area ratio. Defaults to config.MIN_CONTOUR_AREA_RATIO.
        manual_corners: Optional explicit 4 corners [(x1,y1), (x2,y2), (x3,y3), (x4,y4)].

    Returns:
        Tuple of:
        - original_image: Untouched BGR copy for annotations.
        - ingested_image: Rectified or pass-through BGR image ready for Phase 2.
        - transform_meta: Dictionary with keys:
            'warp_applied': bool
            'corners': np.ndarray or None
            'M': np.ndarray or None (Homography matrix)
            'original_shape': Tuple[int, int, int]
            'ingested_shape': Tuple[int, int, int]
    """
    if enable_warp is None:
        enable_warp = config.ENABLE_PERSPECTIVE_WARP
    if min_area_ratio is None:
        min_area_ratio = config.MIN_CONTOUR_AREA_RATIO
    if manual_corners is None:
        manual_corners = config.MANUAL_PERSPECTIVE_CORNERS

    # 1. Load or duplicate image
    if isinstance(image_input, (str, Path)):
        original_image, raw_frame = load_image(image_input)
    elif isinstance(image_input, np.ndarray):
        if image_input.size == 0:
            raise ValueError("[Phase 1 Error] Input numpy image array is empty.")
        original_image = image_input.copy()
        raw_frame = image_input.copy()
    else:
        raise TypeError(f"[Phase 1 Error] Unsupported image input type: {type(image_input)}")

    transform_meta: Dict = {
        "warp_applied": False,
        "corners": None,
        "M": None,
        "original_shape": original_image.shape,
        "ingested_shape": None,
    }

    # 2. Check for manual corner override
    if manual_corners is not None and len(manual_corners) == 4:
        ordered_corners = order_quad_points(np.array(manual_corners, dtype=np.float32))
        flattened, M = warp_perspective(raw_frame, ordered_corners)
        transform_meta["warp_applied"] = True
        transform_meta["corners"] = ordered_corners
        transform_meta["M"] = M
        ingested_image = flattened

    # 3. Automatic boundary detection if warp is enabled
    elif enable_warp:
        detected_corners = detect_document_corners(raw_frame, min_area_ratio=min_area_ratio)
        if detected_corners is not None:
            flattened, M = warp_perspective(raw_frame, detected_corners)
            transform_meta["warp_applied"] = True
            transform_meta["corners"] = detected_corners
            transform_meta["M"] = M
            ingested_image = flattened
        else:
            # Graceful bypass: document boundary not clearly isolated
            ingested_image = raw_frame
    else:
        # Perspective transform disabled: pass full frame
        ingested_image = raw_frame

    transform_meta["ingested_shape"] = ingested_image.shape
    return original_image, ingested_image, transform_meta


class DocumentIngestionEngine:
    """
    Object-oriented encapsulation of Phase 1 Ingestion for modular pipeline integration.
    """

    def __init__(
        self,
        enable_warp: bool = config.ENABLE_PERSPECTIVE_WARP,
        min_area_ratio: float = config.MIN_CONTOUR_AREA_RATIO,
        manual_corners: Optional[List[Tuple[float, float]]] = None,
    ):
        self.enable_warp = enable_warp
        self.min_area_ratio = min_area_ratio
        self.manual_corners = manual_corners

    def ingest(
        self, image_input: Union[str, Path, np.ndarray]
    ) -> Tuple[np.ndarray, np.ndarray, Dict]:
        """Ingests an image through the Phase 1 pipeline."""
        return ingest_image(
            image_input=image_input,
            enable_warp=self.enable_warp,
            min_area_ratio=self.min_area_ratio,
            manual_corners=self.manual_corners,
        )

    @staticmethod
    def generate_synthetic_document_sample(
        width: int = 1280, height: int = 720
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Generates a synthetic tilted document photograph against a textured desk
        background for offline unit testing, CI validation, and demonstrations.

        Returns:
            Tuple of (synthetic_frame, true_corners_array).
        """
        # Create dark textured wooden desk canvas
        frame = np.full((height, width, 3), (45, 55, 65), dtype=np.uint8)
        noise = np.random.normal(0, 3, (height, width, 3)).astype(np.int16)
        frame = np.clip(frame.astype(np.int16) + noise, 0, 255).astype(np.uint8)

        # Define paper sheet quadrilateral corners (tilted perspective)
        sheet_corners = np.array(
            [
                [int(width * 0.22), int(height * 0.16)],  # Top-Left
                [int(width * 0.78), int(height * 0.22)],  # Top-Right
                [int(width * 0.84), int(height * 0.85)],  # Bottom-Right
                [int(width * 0.16), int(height * 0.78)],  # Bottom-Left
            ],
            dtype=np.int32,
        )

        # Draw bright paper sheet polygon
        cv2.fillPoly(frame, [sheet_corners], (242, 244, 246))

        # Add mock handwritten ink strokes on sheet
        cv2.putText(
            frame,
            "Classical CV Handwriting OCR",
            (int(width * 0.26), int(height * 0.35)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.9,
            (25, 25, 30),
            2,
            cv2.LINE_AA,
        )
        cv2.putText(
            frame,
            "Phase 1 Optical Ingestion Test",
            (int(width * 0.25), int(height * 0.45)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (30, 30, 35),
            2,
            cv2.LINE_AA,
        )

        return frame, sheet_corners.astype(np.float32)
