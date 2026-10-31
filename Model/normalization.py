"""
normalization.py - Phase 4: Geometric Normalization

This module normalizes extracted character crops to eliminate variations in handwriting
slant (cursive tilt) and character scale. Every character is converted into a standardized,
invariant 16x16 float32 representation ([0.0, 1.0]) for downstream feature extraction.

Key Stages:
1. Dual-Centroid Slant Estimation:
   Partitions the character into upper and lower vertical zones, calculates the spatial
   center of mass for each, and derives the slant angle theta = arctan(dx / dy).
2. Affine Horizontal Shear Deskewing:
   Applies an affine transformation matrix without stroke clipping via dynamic border padding.
3. Tight Non-Zero Bounding Box Re-Cropping:
   Trims empty zero-padded margins around the deskewed character.
4. Aspect-Ratio Preserving Resizing:
   Fits the deskewed character into a 14x14 active bounding box using anti-aliased area resampling.
5. Centered Zero-Padded Embedding into 16x16 Canvas:
   Embeds the resized character symmetrically into the center of a 16x16 grid with float32 values [0.0, 1.0].
"""

import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
import cv2
import numpy as np

try:
    from . import config
except ImportError:
    import config

try:
    from .segmentation import CharacterCrop
except ImportError:
    try:
        from segmentation import CharacterCrop
    except ImportError:
        CharacterCrop = None


@dataclass
class NormalizedCharacter:
    """Represents a geometrically normalized 16x16 character."""
    char_id: int
    normalized_image: np.ndarray             # 16x16 float32 array in [0.0, 1.0]
    slant_angle_deg: float                    # Estimated slant angle in degrees
    original_bbox: Tuple[int, int, int, int]  # (x, y, w, h) in global document coords
    deskewed_crop: Optional[np.ndarray] = None  # Crop after slant shear, before 16x16 resize
    word_id: int = 0
    line_id: int = 0

    @property
    def shape(self) -> Tuple[int, int]:
        return self.normalized_image.shape


@dataclass
class NormalizationResult:
    """Container holding a collection of normalized characters."""
    characters: List[NormalizedCharacter] = field(default_factory=list)
    mean_slant_deg: float = 0.0

    @property
    def total_characters(self) -> int:
        return len(self.characters)

    def get_feature_input_batch(self) -> np.ndarray:
        """Returns batch array of shape (N, 16, 16) float32."""
        if not self.characters:
            return np.empty((0, 16, 16), dtype=np.float32)
        return np.stack([c.normalized_image for c in self.characters], axis=0)


# -------------------------------------------------------------------------
# Step 1: Slant Estimation via Dual-Centroid Method
# -------------------------------------------------------------------------

def estimate_slant_angle(
    char_binary: np.ndarray,
    min_pixels: int = config.MIN_INK_PIXELS_FOR_SLANT,
    max_angle_deg: float = config.MAX_SLANT_ANGLE_DEG,
) -> float:
    """
    Estimates the handwriting slant angle (cursive tilt) using the dual-centroid method.

    Partitions the binary character crop into Top and Bottom vertical zones:
    - Top Zone: y in [0, floor(H / 2))
    - Bottom Zone: y in [floor(H / 2), H)

    Calculates the spatial center of mass for each zone:
        (x_top, y_top) and (x_bot, y_bot)
        dx = x_top - x_bot
        dy = y_bot - y_top  (note: y increases downwards, so y_bot > y_top)
        theta = arctan(dx / dy)

    A forward (rightward) slant yields theta > 0.
    A backward (leftward) slant yields theta < 0.

    Args:
        char_binary: 2D uint8 inverted binary mask (non-zero = ink).
        min_pixels: Minimum required ink pixels in each zone for reliable estimation.
        max_angle_deg: Maximum plausible slant clamp (default 45.0 degrees).

    Returns:
        Slant angle in degrees (clamped to [-max_angle_deg, max_angle_deg]).
    """
    if char_binary is None or char_binary.size == 0:
        return 0.0

    h, w = char_binary.shape[:2]
    if h < 4 or w < 2:
        return 0.0

    mid_y = h // 2
    top_mask = (char_binary[:mid_y, :] > 0)
    bot_mask = (char_binary[mid_y:, :] > 0)

    top_count = int(np.sum(top_mask))
    bot_count = int(np.sum(bot_mask))

    if top_count < min_pixels or bot_count < min_pixels:
        return 0.0

    # Grid coordinates
    y_grid, x_grid = np.indices((h, w))

    # Center of mass of top zone
    top_x = float(np.sum(x_grid[:mid_y, :][top_mask])) / top_count
    top_y = float(np.sum(y_grid[:mid_y, :][top_mask])) / top_count

    # Center of mass of bottom zone
    bot_x = float(np.sum(x_grid[mid_y:, :][bot_mask])) / bot_count
    bot_y = float(np.sum(y_grid[mid_y:, :][bot_mask])) / bot_count

    dx = top_x - bot_x
    dy = bot_y - top_y

    if dy <= 0.0:
        return 0.0

    angle_rad = math.atan2(dx, dy)
    angle_deg = math.degrees(angle_rad)

    # Clamp to configured maximum slant angle
    return float(np.clip(angle_deg, -max_angle_deg, max_angle_deg))


# -------------------------------------------------------------------------
# Step 2: Affine Horizontal Shear Deskewing (Without Stroke Clipping)
# -------------------------------------------------------------------------

def deskew_character(
    char_image: np.ndarray,
    slant_angle_deg: float,
    border_value: int = 0,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Applies an affine horizontal shear transformation to remove slant without clipping strokes.

    Padding is dynamically added to the left and right borders before shearing:
        pad = ceil(H * |tan(theta)|)
    ensuring all sheared pixels fit within the transformed image boundaries.

    Affine transformation matrix:
        M = [[1,  s, -s * y_bot],
             [0,  1,          0]]
        where s = tan(theta_rad), and y_bot is the bottom coordinate.

    Args:
        char_image: 2D uint8 binary or grayscale crop.
        slant_angle_deg: Slant angle in degrees.
        border_value: Pixel value used for padding (0 for background).

    Returns:
        Tuple of (tight_deskewed_crop, full_sheared_canvas).
    """
    if char_image is None or char_image.size == 0:
        return np.zeros((1, 1), dtype=char_image.dtype if char_image is not None else np.uint8), char_image

    if abs(slant_angle_deg) < 1.0:
        # Negligible slant; return tightly cropped image
        return crop_to_ink(char_image), char_image.copy()

    h, w = char_image.shape[:2]
    angle_rad = math.radians(slant_angle_deg)
    s = math.tan(angle_rad)

    # Calculate required horizontal padding to prevent edge clipping
    pad = int(math.ceil(h * abs(s))) + 2
    padded = cv2.copyMakeBorder(
        char_image,
        top=0,
        bottom=0,
        left=pad,
        right=pad,
        borderType=cv2.BORDER_CONSTANT,
        value=border_value,
    )

    new_w = padded.shape[1]
    y_ref = float(h - 1)

    # Shearing matrix aligning top to bottom reference
    M_shear = np.array(
        [
            [1.0, s, -s * y_ref],
            [0.0, 1.0, 0.0],
        ],
        dtype=np.float32,
    )

    # Apply affine shear
    sheared = cv2.warpAffine(
        padded,
        M_shear,
        (new_w, h),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=border_value,
    )

    # Crop tightly to non-zero ink pixels
    tight_crop = crop_to_ink(sheared)
    return tight_crop, sheared


def crop_to_ink(image: np.ndarray, threshold: int = 10) -> np.ndarray:
    """Crops 2D image tightly to active non-zero foreground ink pixels."""
    if image is None or image.size == 0:
        return np.zeros((1, 1), dtype=np.uint8)

    active_y, active_x = np.where(image > threshold)
    if len(active_x) == 0:
        return image.copy()

    min_x, max_x = int(active_x.min()), int(active_x.max())
    min_y, max_y = int(active_y.min()), int(active_y.max())

    return image[min_y:max_y + 1, min_x:max_x + 1].copy()


# -------------------------------------------------------------------------
# Step 3 & 4: Aspect-Ratio Preserving Resize & Centered Embedding into 16x16
# -------------------------------------------------------------------------

def resize_and_center_16x16(
    char_crop: np.ndarray,
    target_size: Tuple[int, int] = config.NORMALIZED_CHAR_SIZE,
    inner_size: Tuple[int, int] = config.INNER_TARGET_SIZE,
) -> np.ndarray:
    """
    Resizes a character crop into a standardized 16x16 canvas while strictly
    preserving aspect ratio and centering the stroke with zero border cutoff.

    Args:
        char_crop: 2D uint8 or float32 character image.
        target_size: Canvas dimension (height, width), default (16, 16).
        inner_size: Inner active area (height, width), default (14, 14).

    Returns:
        2D float32 array of shape (16, 16) with values normalized to [0.0, 1.0].
    """
    canvas = np.zeros(target_size, dtype=np.float32)
    if char_crop is None or char_crop.size == 0:
        return canvas

    h, w = char_crop.shape[:2]
    if h == 0 or w == 0:
        return canvas

    inner_h, inner_w = inner_size
    target_h, target_w = target_size

    # Compute scaling factor preserving aspect ratio
    scale = min(float(inner_h) / float(h), float(inner_w) / float(w))
    new_w = max(1, int(round(w * scale)))
    new_h = max(1, int(round(h * scale)))

    # Choose anti-aliasing interpolation: INTER_AREA for downscale, INTER_LINEAR for upscale
    interp = cv2.INTER_AREA if scale < 1.0 else cv2.INTER_LINEAR
    resized = cv2.resize(char_crop.astype(np.float32), (new_w, new_h), interpolation=interp)

    # Symmetrical centering offsets
    offset_x = (target_w - new_w) // 2
    offset_y = (target_h - new_h) // 2

    canvas[offset_y:offset_y + new_h, offset_x:offset_x + new_w] = resized

    # Normalize intensity range strictly to [0.0, 1.0]
    if canvas.max() > 1.0:
        canvas = canvas / 255.0

    return np.clip(canvas, 0.0, 1.0).astype(np.float32)


# -------------------------------------------------------------------------
# Step 5: End-to-End Character Normalization Pipeline
# -------------------------------------------------------------------------

def normalize_character(
    char_input: Union[np.ndarray, Any],
    char_id: int = 0,
    enable_slant_correction: bool = config.SLANT_CORRECTION_ENABLED,
    target_size: Tuple[int, int] = config.NORMALIZED_CHAR_SIZE,
    inner_size: Tuple[int, int] = config.INNER_TARGET_SIZE,
    max_slant_angle_deg: float = config.MAX_SLANT_ANGLE_DEG,
) -> NormalizedCharacter:
    """
    End-to-end normalization of a single character crop.

    Accepts either a 2D numpy array or a CharacterCrop object from Phase 3.

    Args:
        char_input: 2D binary numpy array or CharacterCrop dataclass instance.
        char_id: Character index identifier.
        enable_slant_correction: Whether to apply dual-centroid slant deskewing.
        target_size: Standard output canvas dimensions (16, 16).
        inner_size: Inner active area dimensions (14, 14).
        max_slant_angle_deg: Maximum allowed slant correction angle.

    Returns:
        NormalizedCharacter object with 16x16 float32 normalized image.
    """
    original_bbox = (0, 0, 0, 0)
    word_id = 0
    line_id = 0

    # Unpack CharacterCrop object if provided
    if hasattr(char_input, "binary_crop"):
        raw_crop = char_input.binary_crop
        original_bbox = getattr(char_input, "bbox", (0, 0, raw_crop.shape[1], raw_crop.shape[0]))
        char_id = getattr(char_input, "char_id", char_id)
        word_id = getattr(char_input, "word_id", 0)
        line_id = getattr(char_input, "line_id", 0)
    elif isinstance(char_input, np.ndarray):
        raw_crop = char_input
        original_bbox = (0, 0, raw_crop.shape[1] if raw_crop.ndim > 1 else 0, raw_crop.shape[0] if raw_crop.ndim > 0 else 0)
    else:
        raise TypeError(f"[Phase 4 Error] Unsupported character input type: {type(char_input)}")

    # 1. Dual-centroid slant estimation
    slant_deg = 0.0
    if enable_slant_correction:
        slant_deg = estimate_slant_angle(raw_crop, max_angle_deg=max_slant_angle_deg)

    # 2. Affine horizontal shear deskewing
    if enable_slant_correction and abs(slant_deg) >= 1.0:
        deskewed_crop, _ = deskew_character(raw_crop, slant_angle_deg=slant_deg)
    else:
        deskewed_crop = crop_to_ink(raw_crop)

    # 3 & 4. Aspect-ratio preserving resizing & centered 16x16 embedding
    normalized_16x16 = resize_and_center_16x16(
        deskewed_crop,
        target_size=target_size,
        inner_size=inner_size,
    )

    return NormalizedCharacter(
        char_id=char_id,
        normalized_image=normalized_16x16,
        slant_angle_deg=slant_deg,
        original_bbox=original_bbox,
        deskewed_crop=deskewed_crop,
        word_id=word_id,
        line_id=line_id,
    )


def normalize_all_characters(
    characters: Union[List[Any], Any],
    enable_slant_correction: bool = config.SLANT_CORRECTION_ENABLED,
    target_size: Tuple[int, int] = config.NORMALIZED_CHAR_SIZE,
    inner_size: Tuple[int, int] = config.INNER_TARGET_SIZE,
) -> NormalizationResult:
    """
    Normalizes a batch of character crops or a SegmentationResult hierarchy.

    Args:
        characters: List of CharacterCrop / 2D arrays, or a SegmentationResult instance.
        enable_slant_correction: Whether to apply dual-centroid slant deskewing.
        target_size: Output canvas dimensions (16, 16).
        inner_size: Inner active area (14, 14).

    Returns:
        NormalizationResult container with all normalized characters and summary statistics.
    """
    # If a SegmentationResult was passed, extract all characters
    if hasattr(characters, "get_all_characters"):
        char_list = characters.get_all_characters()
    elif isinstance(characters, list):
        char_list = characters
    else:
        char_list = [characters]

    normalized_list: List[NormalizedCharacter] = []
    total_slant = 0.0

    for idx, c in enumerate(char_list):
        norm_char = normalize_character(
            char_input=c,
            char_id=getattr(c, "char_id", idx),
            enable_slant_correction=enable_slant_correction,
            target_size=target_size,
            inner_size=inner_size,
        )
        normalized_list.append(norm_char)
        total_slant += norm_char.slant_angle_deg

    mean_slant = float(total_slant / max(1, len(normalized_list)))

    return NormalizationResult(
        characters=normalized_list,
        mean_slant_deg=mean_slant,
    )


class NormalizationEngine:
    """
    Object-oriented encapsulation of Phase 4 Normalization for modular pipeline integration.
    """

    def __init__(
        self,
        enable_slant_correction: bool = config.SLANT_CORRECTION_ENABLED,
        target_size: Tuple[int, int] = config.NORMALIZED_CHAR_SIZE,
        inner_size: Tuple[int, int] = config.INNER_TARGET_SIZE,
        max_slant_angle_deg: float = config.MAX_SLANT_ANGLE_DEG,
    ):
        self.enable_slant_correction = enable_slant_correction
        self.target_size = target_size
        self.inner_size = inner_size
        self.max_slant_angle_deg = max_slant_angle_deg

    def normalize(self, char_input: Union[np.ndarray, Any], char_id: int = 0) -> NormalizedCharacter:
        """Normalizes a single character crop."""
        return normalize_character(
            char_input=char_input,
            char_id=char_id,
            enable_slant_correction=self.enable_slant_correction,
            target_size=self.target_size,
            inner_size=self.inner_size,
            max_slant_angle_deg=self.max_slant_angle_deg,
        )

    def normalize_batch(self, characters: Union[List[Any], Any]) -> NormalizationResult:
        """Normalizes a batch or hierarchy of character crops."""
        return normalize_all_characters(
            characters=characters,
            enable_slant_correction=self.enable_slant_correction,
            target_size=self.target_size,
            inner_size=self.inner_size,
        )
