"""
features.py - Phase 5: Circular DDD Handcrafted Feature Engine

This module extracts a rotation/scale-invariant, 256-dimensional handcrafted feature
descriptor from standardized 16x16 normalized character images.

Key Foundations & Algorithms (Wong & Sebastian 2015, Katiyar & Mehfuz 2015):
1. Virtual Circular Domain & Boundary Wrap-Around:
   The 16x16 character canvas is inscribed into a continuous circular polar domain
   centered at (7.5, 7.5) with radius R_max = 7.5. Out-of-bounds coordinates wrap around
   and clamp continuously to eliminate rectangular edge clipping artifacts.
2. 8-Directional Ray Casting Trajectory:
   Rays are cast along 8 compass bearings: East, NE, North, NW, West, SW, South, SE.
3. Dual-Set Feature Metrics (W-Set vs B-Set):
   Along each ray trajectory:
   - W-Set (White Distance & Loop Metrics):
     * W_dist: Distance traveled from origin before hitting first ink stroke (concavity metric).
     * W_span: Cumulative white space along the ray (internal loop / hole metric for '0', '8', 'O', 'e').
   - B-Set (Black Thickness & Density Metrics):
     * B_thick: Maximum contiguous stroke thickness (stroke width and junction metric).
     * B_dens: Integrated ink pixel density along the ray (mass distribution).
4. Multi-Ring Radial Topology:
   8 concentric radial rings x 8 directional rays x 4 dual-set metrics = 256 dimensions.
5. Alternative Direct Raster Mode:
   Direct flattened 16x16 grayscale array (256 dimensions) supported as baseline.
"""

import math
from dataclasses import dataclass
from typing import Any, List, Optional, Tuple, Union
import numpy as np

try:
    from . import config
except ImportError:
    import config

try:
    from .normalization import NormalizedCharacter, NormalizationResult
except ImportError:
    try:
        from normalization import NormalizedCharacter, NormalizationResult
    except ImportError:
        NormalizedCharacter = None
        NormalizationResult = None


# -------------------------------------------------------------------------
# Step 1: Virtual Circular Domain Sampling with Bilinear Interpolation
# -------------------------------------------------------------------------

def sample_virtual_circular(
    image: np.ndarray,
    x: float,
    y: float,
    xc: float = 7.5,
    yc: float = 7.5,
    r_max: float = 7.5,
) -> float:
    """
    Samples a continuous sub-pixel intensity from the normalized character image
    using virtual circular polar mapping and bilinear interpolation.

    Coordinates outside the circumscribed radius R_max are smoothly mapped
    along the polar trajectory r <= R_max to prevent rectangular boundary artifacts.

    Args:
        image: 2D float32 array of shape (16, 16) with values in [0.0, 1.0].
        x: Continuous horizontal query coordinate.
        y: Continuous vertical query coordinate.
        xc: Center x-coordinate (default 7.5).
        yc: Center y-coordinate (default 7.5).
        r_max: Maximum inscribed circular radius (default 7.5).

    Returns:
        Interpolated float32 intensity bounded in [0.0, 1.0].
    """
    dx = x - xc
    dy = y - yc
    r = math.sqrt(dx * dx + dy * dy)
    theta = math.atan2(dy, dx)

    # Polar clamping to circular boundary
    if r > r_max:
        r = r_max

    mx = xc + r * math.cos(theta)
    my = yc + r * math.sin(theta)

    # Bound integer lattice coordinates
    h, w = image.shape[:2]
    x0 = int(math.floor(mx))
    y0 = int(math.floor(my))
    x0 = max(0, min(w - 1, x0))
    y0 = max(0, min(h - 1, y0))

    x1 = min(w - 1, x0 + 1)
    y1 = min(h - 1, y0 + 1)

    wx = float(mx - x0)
    wy = float(my - y0)

    # Bilinear four-corner interpolation
    v00 = float(image[y0, x0])
    v01 = float(image[y0, x1])
    v10 = float(image[y1, x0])
    v11 = float(image[y1, x1])

    val = (
        (1.0 - wx) * (1.0 - wy) * v00
        + wx * (1.0 - wy) * v01
        + (1.0 - wx) * wy * v10
        + wx * wy * v11
    )

    return float(np.clip(val, 0.0, 1.0))


# -------------------------------------------------------------------------
# Step 2: Directional Ray Unit Vectors (8 Compass Bearings)
# -------------------------------------------------------------------------

def get_directional_unit_vectors(num_directions: int = config.DDD_NUM_DIRECTIONS) -> List[Tuple[float, float]]:
    """
    Computes unit vectors along discrete radial angular bearings.

    For num_directions = 8:
    - 0: East (1, 0)
    - 1: NE (1/sqrt(2), 1/sqrt(2))
    - 2: North (0, 1)
    - 3: NW (-1/sqrt(2), 1/sqrt(2))
    - 4: West (-1, 0)
    - 5: SW (-1/sqrt(2), -1/sqrt(2))
    - 6: South (0, -1)
    - 7: SE (1/sqrt(2), -1/sqrt(2))
    """
    vectors = []
    for k in range(num_directions):
        angle = k * (2.0 * math.pi / float(num_directions))
        vectors.append((math.cos(angle), math.sin(angle)))
    return vectors


# -------------------------------------------------------------------------
# Step 3: Circular Directional Distance Distribution (DDD) Ray Casting
# -------------------------------------------------------------------------

def extract_circular_ddd_features(
    image: np.ndarray,
    num_rings: int = config.DDD_NUM_RINGS,
    num_directions: int = config.DDD_NUM_DIRECTIONS,
    step_size: float = config.DDD_RAY_STEP_SIZE,
    max_steps: int = config.DDD_MAX_RAY_STEPS,
    ink_threshold: float = 0.5,
) -> np.ndarray:
    """
    Extracts the 256-dimensional Multi-Ring Radial Circular DDD feature descriptor.

    Architecture:
    - 8 concentric radial rings (r = 0.5 * r_step, 1.5 * r_step, ..., 7.5 * r_step)
    - 8 directional ray trajectories per anchor point
    - 4 dual-set metrics per ray:
      1. W_dist : Distance to first foreground stroke edge (concavity / outer contour)
      2. W_span : Cumulative white space traversed along ray (internal loop / hole metric)
      3. B_thick: Maximum contiguous stroke thickness encountered (junction / line width)
      4. B_dens : Integrated ink pixel density along trajectory (mass distribution)

    Total Dimensionality:
        8 rings x 8 directions x 4 dual-set metrics = 256 features.

    Args:
        image: 2D float32 normalized character image of shape (16, 16) with values in [0.0, 1.0].
        num_rings: Number of concentric radial sampling rings (default 8).
        num_directions: Directional compass count (default 8).
        step_size: Sub-pixel integration step along ray trajectory (default 0.5 px).
        max_steps: Maximum ray tracing steps (default 15 steps -> 7.5 px).
        ink_threshold: Foreground ink intensity threshold (default 0.5).

    Returns:
        1D float32 NumPy array of shape (256,) bounded strictly in [0.0, 1.0].
    """
    if image is None or image.size == 0:
        return np.zeros(config.FEATURE_VECTOR_DIM, dtype=np.float32)

    h, w = image.shape[:2]
    xc = float(w - 1) / 2.0  # 7.5
    yc = float(h - 1) / 2.0  # 7.5
    r_max = min(xc, yc)       # 7.5

    dirs = get_directional_unit_vectors(num_directions)
    radii = [(m + 0.5) * (r_max / float(num_rings)) for m in range(num_rings)]

    features: List[float] = []
    max_ray_distance = max_steps * step_size

    for r in radii:
        for dx, dy in dirs:
            # Ray origin anchor point along the ring
            ox = xc + r * dx
            oy = yc + r * dy

            w_dist = max_ray_distance
            w_span = 0.0
            b_thick = 0.0
            cur_thick = 0.0
            dens_sum = 0.0
            hit_first_ink = False

            for step in range(max_steps):
                t = step * step_size
                px = ox + t * dx
                py = oy + t * dy

                # Continuous sub-pixel sample
                val = sample_virtual_circular(image, px, py, xc, yc, r_max)
                dens_sum += val

                if val >= ink_threshold:
                    if not hit_first_ink:
                        w_dist = t
                        hit_first_ink = True
                    cur_thick += step_size
                    if cur_thick > b_thick:
                        b_thick = cur_thick
                else:
                    cur_thick = 0.0
                    w_span += step_size

            # 4 dual-set metrics normalized strictly to [0.0, 1.0]
            f_w_dist = min(1.0, max(0.0, float(w_dist / max_ray_distance)))
            f_w_span = min(1.0, max(0.0, float(w_span / max_ray_distance)))
            f_b_thick = min(1.0, max(0.0, float(b_thick / max_ray_distance)))
            f_b_dens = min(1.0, max(0.0, float(dens_sum / float(max_steps))))

            features.extend([f_w_dist, f_w_span, f_b_thick, f_b_dens])

    feature_vec = np.array(features, dtype=np.float32)

    # Sanity checks
    if len(feature_vec) != config.FEATURE_VECTOR_DIM:
        raise ValueError(
            f"[Phase 5 Error] Feature vector dimension mismatch: got {len(feature_vec)}, "
            f"expected {config.FEATURE_VECTOR_DIM}"
        )

    # Guarantee zero NaNs or Infs
    if not np.isfinite(feature_vec).all():
        feature_vec = np.nan_to_num(feature_vec, nan=0.0, posinf=1.0, neginf=0.0)

    return np.clip(feature_vec, 0.0, 1.0)


# -------------------------------------------------------------------------
# Step 4: Baseline Direct Raster Flattening Mode (16x16 -> 256)
# -------------------------------------------------------------------------

def flatten_raster_features(image: np.ndarray) -> np.ndarray:
    """
    Direct flattened 16x16 normalized raster representation.

    Args:
        image: 2D float32 normalized image of shape (16, 16).

    Returns:
        1D float32 array of shape (256,) in range [0.0, 1.0].
    """
    if image is None or image.size == 0:
        return np.zeros(config.FEATURE_VECTOR_DIM, dtype=np.float32)

    flat = image.astype(np.float32).ravel()
    if len(flat) != config.FEATURE_VECTOR_DIM:
        flat = np.resize(flat, config.FEATURE_VECTOR_DIM)

    return np.clip(flat, 0.0, 1.0)


# -------------------------------------------------------------------------
# Step 5: Unified Feature Extraction API & Batch Processing
# -------------------------------------------------------------------------

def extract_features(
    char_input: Union[np.ndarray, Any],
    mode: str = config.FEATURE_MODE,
) -> np.ndarray:
    """
    Extracts a 256-dimensional feature vector from a single normalized character.

    Accepts either:
    - 2D NumPy array of shape (16, 16) float32 in [0.0, 1.0].
    - NormalizedCharacter dataclass instance from Phase 4.

    Args:
        char_input: Normalized image array or NormalizedCharacter object.
        mode: Feature mode ("circular_ddd" or "flatten_16x16").

    Returns:
        1D float32 array of shape (256,) with values in [0.0, 1.0].
    """
    if hasattr(char_input, "normalized_image"):
        img = char_input.normalized_image
    elif isinstance(char_input, np.ndarray):
        img = char_input
    else:
        raise TypeError(f"[Phase 5 Error] Unsupported input type for feature extraction: {type(char_input)}")

    if img.shape != config.NORMALIZED_CHAR_SIZE:
        raise ValueError(
            f"[Phase 5 Error] Input image must be shape {config.NORMALIZED_CHAR_SIZE}, got {img.shape}"
        )

    mode_clean = str(mode).lower().strip()
    if mode_clean in ("circular_ddd", "ddd", "circular"):
        return extract_circular_ddd_features(img)
    elif mode_clean in ("flatten_16x16", "flatten", "raster", "direct"):
        return flatten_raster_features(img)
    else:
        raise ValueError(
            f"[Phase 5 Error] Unknown feature extraction mode: '{mode}'. "
            f"Expected 'circular_ddd' or 'flatten_16x16'."
        )


def extract_features_batch(
    char_inputs: Union[List[Any], Any, np.ndarray],
    mode: str = config.FEATURE_MODE,
) -> np.ndarray:
    """
    Extracts 256-dimensional feature vectors for a collection or batch of characters.

    Args:
        char_inputs: List of NormalizedCharacter objects, NormalizationResult, or (N, 16, 16) array.
        mode: Feature extraction mode.

    Returns:
        2D float32 NumPy array of shape (N, 256) where each row is a 256-D feature vector.
    """
    if hasattr(char_inputs, "characters"):
        items = char_inputs.characters
    elif isinstance(char_inputs, np.ndarray) and char_inputs.ndim == 3:
        items = [char_inputs[i] for i in range(char_inputs.shape[0])]
    elif isinstance(char_inputs, list):
        items = char_inputs
    else:
        items = [char_inputs]

    if not items:
        return np.empty((0, config.FEATURE_VECTOR_DIM), dtype=np.float32)

    feature_rows = []
    for item in items:
        vec = extract_features(item, mode=mode)
        feature_rows.append(vec)

    return np.stack(feature_rows, axis=0).astype(np.float32)


class FeatureExtractionEngine:
    """
    Object-oriented encapsulation of Phase 5 Feature Extraction for modular integration.
    """

    def __init__(self, mode: str = config.FEATURE_MODE):
        self.mode = mode

    def extract(self, char_input: Union[np.ndarray, Any]) -> np.ndarray:
        """Extracts a 256-D feature vector from a single character."""
        return extract_features(char_input, mode=self.mode)

    def extract_batch(self, char_inputs: Union[List[Any], Any, np.ndarray]) -> np.ndarray:
        """Extracts a batch matrix of shape (N, 256) from character crops."""
        return extract_features_batch(char_inputs, mode=self.mode)
