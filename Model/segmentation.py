"""
segmentation.py - Phase 3: Hierarchical Segmentation (Line -> Word -> Character)

This module recursively decomposes the clean inverted binary handwriting image
(255 = ink, 0 = paper background) into a deterministic hierarchy:
    Lines -> Words -> Character candidate boxes

Key Capabilities:
1. Horizontal Projection Profile (HPP) Line Detection:
   Row-wise pixel density projection with 1D Gaussian smoothing and valley grouping.
2. Morphological Word Bridging & Vertical Projection (VPP):
   Horizontal dilation bridges intra-word letter gaps while preserving inter-word spacing.
3. Connected Component Character Extraction:
   Isolates candidate character bounding boxes within each word.
4. Noise, Speckle, and Ruling Line Filtering:
   Rejects microscopic dust (< 15 px^2) and extreme aspect-ratio artifacts (scratches/lines).
5. Diacritic & Dot Association:
   Associates dots on 'i' and 'j' with their corresponding stems.
6. Joined Cursive Ligature Splitting:
   Identifies thinning points in abnormally wide connected components.
7. Color-coded Diagnostic Visual Overlay:
   Blue lines, Green words, and Red characters saved for verification.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
import cv2
import numpy as np

try:
    from scipy.ndimage import gaussian_filter1d
except ImportError:
    gaussian_filter1d = None

try:
    from . import config
except ImportError:
    import config


# -------------------------------------------------------------------------
# Hierarchical Data Structures
# -------------------------------------------------------------------------

@dataclass
class CharacterCrop:
    """Represents an isolated individual character candidate."""
    char_id: int
    bbox: Tuple[int, int, int, int]      # (x, y, w, h) in global image coordinates
    binary_crop: np.ndarray              # 2D uint8 inverted binary mask (255 = ink)
    gray_crop: Optional[np.ndarray] = None  # 2D uint8 grayscale crop
    word_id: int = 0
    line_id: int = 0

    @property
    def x(self) -> int:
        return self.bbox[0]

    @property
    def y(self) -> int:
        return self.bbox[1]

    @property
    def width(self) -> int:
        return self.bbox[2]

    @property
    def height(self) -> int:
        return self.bbox[3]

    @property
    def aspect_ratio(self) -> float:
        return float(self.width) / max(1, self.height)


@dataclass
class Word:
    """Represents a coherent word composed of ordered characters."""
    word_id: int
    bbox: Tuple[int, int, int, int]      # (x, y, w, h) in global image coordinates
    characters: List[CharacterCrop] = field(default_factory=list)
    line_id: int = 0

    @property
    def x(self) -> int:
        return self.bbox[0]

    @property
    def y(self) -> int:
        return self.bbox[1]

    @property
    def width(self) -> int:
        return self.bbox[2]

    @property
    def height(self) -> int:
        return self.bbox[3]


@dataclass
class Line:
    """Represents a single text line composed of ordered words."""
    line_id: int
    bbox: Tuple[int, int, int, int]      # (x, y, w, h) in global image coordinates
    words: List[Word] = field(default_factory=list)

    @property
    def x(self) -> int:
        return self.bbox[0]

    @property
    def y(self) -> int:
        return self.bbox[1]

    @property
    def width(self) -> int:
        return self.bbox[2]

    @property
    def height(self) -> int:
        return self.bbox[3]


@dataclass
class SegmentationResult:
    """Container holding the complete document segmentation hierarchy."""
    lines: List[Line] = field(default_factory=list)
    image_shape: Tuple[int, int] = (0, 0)

    @property
    def total_lines(self) -> int:
        return len(self.lines)

    @property
    def total_words(self) -> int:
        return sum(len(line.words) for line in self.lines)

    @property
    def total_characters(self) -> int:
        return sum(len(word.characters) for line in self.lines for word in line.words)

    def get_all_characters(self) -> List[CharacterCrop]:
        """Returns flat list of all characters in deterministic reading order."""
        chars = []
        for line in self.lines:
            for word in line.words:
                chars.extend(word.characters)
        return chars

    def render_debug_overlay(self, base_image: Optional[np.ndarray] = None) -> np.ndarray:
        """Renders diagnostic overlay with Blue lines, Green words, Red chars."""
        return render_segmentation_overlay(base_image, self)


# -------------------------------------------------------------------------
# Step 1: Line Detection via Horizontal Projection Profiles (HPP)
# -------------------------------------------------------------------------

def smooth_1d(signal: np.ndarray, sigma: float = config.HPP_SMOOTHING_SIGMA) -> np.ndarray:
    """Smoothes a 1D projection array using Gaussian filtering."""
    if len(signal) < 3:
        return signal.copy().astype(np.float32)

    if gaussian_filter1d is not None:
        return gaussian_filter1d(signal.astype(np.float32), sigma=sigma)

    # Fallback to OpenCV GaussianBlur on (N, 1) column vector
    ksize = int(2 * round(3 * sigma) + 1)
    ksize = max(3, min(ksize, len(signal) if len(signal) % 2 == 1 else len(signal) - 1))
    col = signal.reshape(-1, 1).astype(np.float32)
    smoothed = cv2.GaussianBlur(col, (1, ksize), sigmaX=0, sigmaY=sigma)
    return smoothed.ravel()


def compute_horizontal_projection(binary_mask: np.ndarray) -> np.ndarray:
    """Computes the row-wise sum of foreground ink pixels (255 = 1)."""
    return np.sum(binary_mask == 255, axis=1).astype(np.float32)


def detect_lines(
    binary_mask: np.ndarray,
    smoothing_sigma: float = config.HPP_SMOOTHING_SIGMA,
    valley_threshold_ratio: float = config.LINE_VALLEY_THRESHOLD_RATIO,
    min_line_height: int = config.MIN_LINE_HEIGHT,
    min_line_gap: int = config.MIN_LINE_GAP,
) -> List[Tuple[int, int, int, int]]:
    """
    Detects individual text lines using smoothed Horizontal Projection Profiles.

    Args:
        binary_mask: Inverted binary image (255 = ink, 0 = paper).
        smoothing_sigma: Gaussian kernel sigma for profile smoothing.
        valley_threshold_ratio: Relative threshold against max HPP to identify text presence.
        min_line_height: Minimum pixel height to accept as a valid line.
        min_line_gap: Consecutive line intervals separated by fewer pixels are merged.

    Returns:
        List of bounding boxes [(x, y, w, h)] sorted strictly top-to-bottom (y1 < y2).
    """
    h, w = binary_mask.shape[:2]
    hpp = compute_horizontal_projection(binary_mask)
    if hpp.max() == 0:
        return []

    hpp_smooth = smooth_1d(hpp, sigma=smoothing_sigma)
    threshold = valley_threshold_ratio * float(hpp_smooth.max())

    # Find contiguous active row segments
    active_mask = hpp_smooth > threshold
    intervals: List[Tuple[int, int]] = []
    in_interval = False
    start_y = 0

    for y, is_active in enumerate(active_mask):
        if is_active and not in_interval:
            in_interval = True
            start_y = y
        elif not is_active and in_interval:
            in_interval = False
            intervals.append((start_y, y))

    if in_interval:
        intervals.append((start_y, len(active_mask)))

    if not intervals:
        return []

    # Merge intervals separated by gaps smaller than min_line_gap
    merged_intervals: List[Tuple[int, int]] = []
    current_start, current_end = intervals[0]

    for next_start, next_end in intervals[1:]:
        gap = next_start - current_end
        if gap < min_line_gap:
            current_end = next_end
        else:
            merged_intervals.append((current_start, current_end))
            current_start, current_end = next_start, next_end
    merged_intervals.append((current_start, current_end))

    # Construct line bounding boxes, trimming horizontal margins
    line_boxes: List[Tuple[int, int, int, int]] = []
    for sy, ey in merged_intervals:
        line_h = ey - sy
        if line_h < min_line_height:
            continue

        # Find horizontal ink extent within line slice
        slice_mask = binary_mask[sy:ey, :]
        col_sums = np.sum(slice_mask == 255, axis=0)
        active_cols = np.where(col_sums > 0)[0]
        if len(active_cols) == 0:
            continue

        min_x = int(active_cols[0])
        max_x = int(active_cols[-1])
        line_w = max_x - min_x + 1

        # Add small vertical padding if feasible
        pad_y = max(0, sy - 1)
        pad_ey = min(h, ey + 1)
        line_boxes.append((min_x, pad_y, line_w, pad_ey - pad_y))

    # Sort strictly top-to-bottom
    line_boxes.sort(key=lambda box: box[1])
    return line_boxes


# -------------------------------------------------------------------------
# Step 2: Word Detection via Morphological Bridging & Vertical Projection
# -------------------------------------------------------------------------

def detect_words(
    line_binary: np.ndarray,
    line_offset: Tuple[int, int],
    word_dilation_w: int = config.WORD_DILATION_KERNEL_W,
    min_word_area: int = 25,
) -> List[Tuple[int, int, int, int]]:
    """
    Detects word regions within a line slice using adaptive bimodal gap clustering
    between character components, with dilation fallback.

    Args:
        line_binary: 2D binary mask of the line.
        line_offset: (x_offset, y_offset) to map local line coords to global image space.
        word_dilation_w: Width of rectangular horizontal dilation kernel.
        min_word_area: Minimum pixel area to keep as a valid word.

    Returns:
        List of global bounding boxes [(x, y, w, h)] sorted strictly left-to-right.
    """
    if line_binary.size == 0 or line_binary.max() == 0:
        return []

    off_x, off_y = line_offset
    line_h, line_w = line_binary.shape[:2]

    # Find connected components within the text line
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(line_binary, connectivity=8)
    valid_comps = []
    adaptive_min_area = max(15, int(line_h * 0.08 * line_h * 0.08))
    adaptive_min_h = max(6, int(line_h * 0.10))

    for i in range(1, num_labels):
        x, y, w, h, area = stats[i]
        if area >= adaptive_min_area and w >= 3 and h >= adaptive_min_h:
            valid_comps.append([x, y, w, h])

    if not valid_comps:
        return []

    valid_comps.sort(key=lambda b: b[0])

    # Cluster components that horizontally overlap or are adjacent stroke fragments
    merged = []
    for b in valid_comps:
        if not merged:
            merged.append(list(b))
        else:
            prev = merged[-1]
            if b[0] <= prev[0] + prev[2] + 2:
                new_x = min(prev[0], b[0])
                new_y = min(prev[1], b[1])
                new_r = max(prev[0] + prev[2], b[0] + b[2])
                new_b = max(prev[1] + prev[3], b[1] + b[3])
                merged[-1] = [new_x, new_y, new_r - new_x, new_b - new_y]
            else:
                merged.append(list(b))

    if len(merged) <= 1:
        ay, ax = np.where(line_binary == 255)
        if len(ax) == 0:
            return []
        rx, ry = int(ax.min()), int(ay.min())
        rw = int(ax.max() - ax.min() + 1)
        rh = int(ay.max() - ay.min() + 1)
        return [(off_x + rx, off_y + ry, rw, rh)]

    # Compute horizontal gaps between consecutive character clusters
    gaps = [max(0, merged[i + 1][0] - (merged[i][0] + merged[i][2])) for i in range(len(merged) - 1)]
    gaps_arr = np.array(sorted(gaps), dtype=np.float32)

    if len(np.unique(gaps_arr)) < 2:
        threshold = float(gaps_arr[0] - 1) if gaps_arr[0] >= max(6, int(line_h * 0.20)) else float(line_h * 0.35)
    else:
        # 1D Otsu thresholding to separate intra-word gaps from inter-word gaps
        best_thresh = gaps_arr[-1]
        best_var = -1.0
        for t in np.unique(gaps_arr)[:-1]:
            w0 = np.mean(gaps_arr <= t)
            w1 = np.mean(gaps_arr > t)
            if w0 == 0 or w1 == 0:
                continue
            m0 = np.mean(gaps_arr[gaps_arr <= t])
            m1 = np.mean(gaps_arr[gaps_arr > t])
            v = w0 * w1 * ((m0 - m1) ** 2)
            if v > best_var:
                best_var = v
                best_thresh = t

        m0 = np.mean(gaps_arr[gaps_arr <= best_thresh])
        m1 = np.mean(gaps_arr[gaps_arr > best_thresh])

        if m1 >= 1.6 * max(1.0, m0) and (m1 - m0) >= max(5.0, line_h * 0.10):
            t_low = gaps_arr[gaps_arr <= best_thresh][-1]
            t_high = gaps_arr[gaps_arr > best_thresh][0]
            threshold = float((t_low + t_high) / 2.0)
        else:
            threshold = float(line_h * 0.35)

    # Group character clusters into words
    word_groups = [[merged[0]]]
    for i in range(len(merged) - 1):
        g = merged[i + 1][0] - (merged[i][0] + merged[i][2])
        if g > threshold:
            word_groups.append([merged[i + 1]])
        else:
            word_groups[-1].append(merged[i + 1])

    word_boxes: List[Tuple[int, int, int, int]] = []
    for grp in word_groups:
        gx1 = min(b[0] for b in grp)
        gy1 = min(b[1] for b in grp)
        gx2 = max(b[0] + b[2] for b in grp)
        gy2 = max(b[1] + b[3] for b in grp)
        crop = line_binary[gy1:gy2, gx1:gx2]
        ay, ax = np.where(crop == 255)
        if len(ax) == 0:
            continue
        rx = gx1 + int(ax.min())
        ry = gy1 + int(ay.min())
        rw = int(ax.max() - ax.min() + 1)
        rh = int(ay.max() - ay.min() + 1)
        word_boxes.append((off_x + rx, off_y + ry, rw, rh))

    word_boxes.sort(key=lambda box: box[0])
    return word_boxes


# -------------------------------------------------------------------------
# Step 3: Character Extraction, Ligature Splitting & Diacritic Merging
# -------------------------------------------------------------------------

def merge_diacritics(
    boxes: List[Tuple[int, int, int, int]],
    max_vertical_gap: Optional[int] = 15,
    word_h: Optional[int] = None,
) -> List[Tuple[int, int, int, int]]:
    """
    Merges dots on 'i' and 'j', accents, and intersecting strokes of the same character.

    Args:
        boxes: List of candidate character bounding boxes (x, y, w, h).
        max_vertical_gap: Maximum vertical distance between diacritic and main stem.
        word_h: Optional word height for resolution-adaptive merging.

    Returns:
        Consolidated list of bounding boxes with diacritics and sub-strokes unified.
    """
    if len(boxes) <= 1:
        return boxes

    if word_h is None:
        word_h = max(b[3] for b in boxes)

    effective_vert_gap = max(max_vertical_gap or 15, int(word_h * 0.40))

    changed = True
    current = list(boxes)
    while changed:
        changed = False
        new_boxes = []
        used = set()
        for i in range(len(current)):
            if i in used:
                continue
            x1, y1, w1, h1 = current[i]
            merged_box = [x1, y1, w1, h1]

            for j in range(i + 1, len(current)):
                if j in used:
                    continue
                x2, y2, w2, h2 = current[j]

                overlap_x = min(merged_box[0] + merged_box[2], x2 + w2) - max(merged_box[0], x2)
                overlap_y = min(merged_box[1] + merged_box[3], y2 + h2) - max(merged_box[1], y2)
                vert_gap = max(0, max(merged_box[1], y2) - min(merged_box[1] + merged_box[3], y2 + h2))
                min_w = min(merged_box[2], w2)

                combined_w = max(merged_box[0] + merged_box[2], x2 + w2) - min(merged_box[0], x2)
                combined_h = max(merged_box[1] + merged_box[3], y2 + h2) - min(merged_box[1], y2)

                should_merge = False
                # 1. 2D intersection or significant horizontal overlap
                if overlap_x >= max(4, int(min_w * 0.15)):
                    if overlap_y > 0 or vert_gap <= effective_vert_gap:
                        should_merge = True

                # 2. Multi-stroke letter (adjacent vertical strokes like M or H sharing baseline)
                min_h = min(merged_box[3], h2)
                if overlap_x >= 1 and (overlap_y / max(1, min_h)) >= 0.70:
                    should_merge = True

                # 3. Small diacritic dot directly above a taller stem
                c1_dot = (merged_box[3] <= int(word_h * 0.35) and merged_box[2] <= int(word_h * 0.40))
                c2_dot = (h2 <= int(word_h * 0.35) and w2 <= int(word_h * 0.40))
                if (c1_dot or c2_dot) and vert_gap <= effective_vert_gap:
                    dot_cx = (merged_box[0] + merged_box[2] / 2.0) if c1_dot else (x2 + w2 / 2.0)
                    stem_x = x2 if c1_dot else merged_box[0]
                    stem_w = w2 if c1_dot else merged_box[2]
                    if (stem_x - 10) <= dot_cx <= (stem_x + stem_w + 10):
                        should_merge = True

                # Bound combined width to reasonable character aspect ratio
                if should_merge and combined_w <= int(max(word_h, 30) * 1.25):
                    new_x = min(merged_box[0], x2)
                    new_y = min(merged_box[1], y2)
                    merged_box = [new_x, new_y, combined_w, combined_h]
                    used.add(j)
                    changed = True

            used.add(i)
            new_boxes.append(tuple(merged_box))
        current = new_boxes

    current.sort(key=lambda b: b[0])
    return current


def split_wide_ligature(
    char_binary: np.ndarray,
    aspect_ratio_threshold: float = config.MAX_ASPECT_RATIO_SPLIT,
) -> List[Tuple[int, int, int, int]]:
    """
    Detects and splits joined cursive characters (ligatures) at thinned stroke valleys,
    preserving multi-hump individual characters (e.g. 'm', 'w', 'M', 'W').

    Args:
        char_binary: 2D binary crop of the connected component.
        aspect_ratio_threshold: Width/Height ratio above which splitting is attempted.

    Returns:
        List of relative sub-bounding boxes [(x, y, w, h)] within the component.
    """
    h, w = char_binary.shape[:2]
    if h == 0 or w == 0:
        return []

    aspect_ratio = float(w) / float(max(1, h))
    if aspect_ratio <= aspect_ratio_threshold or w < 28:
        return [(0, 0, w, h)]

    # Compute column projection (VPP)
    vpp = np.sum(char_binary == 255, axis=0)

    # Search in middle 50% region for a stroke thinning valley
    mid_start = int(0.25 * w)
    mid_end = int(0.75 * w)
    if mid_end <= mid_start:
        return [(0, 0, w, h)]

    mid_vpp = vpp[mid_start:mid_end]
    min_val = np.min(mid_vpp)
    max_val = np.max(vpp)

    # If significant thinning exists (valley <= 0.45 * max stroke density)
    if max_val > 0 and (min_val / float(max_val)) <= 0.45:
        split_x = mid_start + int(np.argmin(mid_vpp))
        # Ensure sub-boxes have reasonable widths
        if split_x >= 6 and (w - split_x) >= 6:
            return [(0, 0, split_x, h), (split_x, 0, w - split_x, h)]

    return [(0, 0, w, h)]


def extract_characters(
    word_binary: np.ndarray,
    word_offset: Tuple[int, int],
    full_binary_mask: np.ndarray,
    full_gray_image: Optional[np.ndarray] = None,
    word_id: int = 0,
    line_id: int = 0,
    min_area: int = config.MIN_CHAR_AREA,
    min_w: int = config.MIN_CHAR_WIDTH,
    min_h: int = config.MIN_CHAR_HEIGHT,
    max_aspect_ratio: float = config.MAX_CHAR_ASPECT_RATIO,
    enable_ligature_split: bool = True,
) -> List[CharacterCrop]:
    """
    Extracts individual character crops from a word region, applying noise filtering,
    diacritic merging, and ligature splitting.

    Args:
        word_binary: Inverted binary image of the word.
        word_offset: (x, y) global coordinate offset.
        full_binary_mask: Global binary document mask.
        full_gray_image: Optional global grayscale document image.
        word_id: Word identifier.
        line_id: Line identifier.
        min_area: Minimum connected component area.
        min_w: Minimum character box width.
        min_h: Minimum character box height.
        max_aspect_ratio: Scratch/ruling line aspect ratio cutoff.
        enable_ligature_split: Whether to split joined cursive components.

    Returns:
        List of CharacterCrop objects sorted strictly left-to-right.
    """
    if word_binary.size == 0 or word_binary.max() == 0:
        return []

    off_x, off_y = word_offset
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(word_binary, connectivity=8)

    word_h, word_w = word_binary.shape[:2]
    adaptive_min_area = max(min_area, int(word_h * 0.03 * word_h * 0.03))
    adaptive_min_h = max(min_h, int(word_h * 0.10))

    raw_candidate_boxes: List[Tuple[int, int, int, int]] = []
    for label in range(1, num_labels):
        x, y, w, h, area = stats[label]

        # 1. Reject tiny dust speckles
        if area < adaptive_min_area:
            continue

        # 2. Reject dimensions below minimum character threshold
        if w < min_w or h < adaptive_min_h:
            continue

        # 3. Reject extreme aspect ratios (scratches or horizontal ruling lines)
        ar = float(w) / float(max(1, h))
        if ar > max_aspect_ratio or (1.0 / max(0.01, ar)) > max_aspect_ratio:
            continue

        raw_candidate_boxes.append((x, y, w, h))

    if not raw_candidate_boxes:
        return []

    # 4. Merge diacritics (dots on 'i', accents, stroke fragments)
    boxes = merge_diacritics(raw_candidate_boxes, word_h=word_h)

    # 5. Ligature splitting and final crop generation
    final_crops: List[CharacterCrop] = []
    char_counter = 0

    img_h, img_w = full_binary_mask.shape[:2]

    for bx, by, bw, bh in boxes:
        comp_crop = word_binary[by:by+bh, bx:bx+bw]

        sub_boxes = [(0, 0, bw, bh)]
        if enable_ligature_split:
            sub_boxes = split_wide_ligature(comp_crop)

        for sx, sy, sw, sh in sub_boxes:
            global_x = max(0, min(img_w - 1, off_x + bx + sx))
            global_y = max(0, min(img_h - 1, off_y + by + sy))
            global_w = min(img_w - global_x, sw)
            global_h = min(img_h - global_y, sh)

            if global_w < min_w or global_h < min_h:
                continue

            # Crop character binary mask
            char_bin = full_binary_mask[global_y:global_y+global_h, global_x:global_x+global_w]
            if char_bin.size == 0 or (char_bin == 255).sum() < min_area:
                continue

            # Crop grayscale image if provided
            char_gray = None
            if full_gray_image is not None:
                char_gray = full_gray_image[global_y:global_y+global_h, global_x:global_x+global_w]

            char_crop = CharacterCrop(
                char_id=char_counter,
                bbox=(global_x, global_y, global_w, global_h),
                binary_crop=char_bin.copy(),
                gray_crop=char_gray.copy() if char_gray is not None else None,
                word_id=word_id,
                line_id=line_id,
            )
            final_crops.append(char_crop)
            char_counter += 1

    # Sort strictly left-to-right
    final_crops.sort(key=lambda c: c.bbox[0])
    for idx, c in enumerate(final_crops):
        c.char_id = idx

    return final_crops


# -------------------------------------------------------------------------
# Step 4: End-to-End Hierarchical Document Segmentation
# -------------------------------------------------------------------------

def segment_document(
    binary_mask: np.ndarray,
    gray_image: Optional[np.ndarray] = None,
    smoothing_sigma: float = config.HPP_SMOOTHING_SIGMA,
    line_valley_ratio: float = config.LINE_VALLEY_THRESHOLD_RATIO,
    min_line_height: int = config.MIN_LINE_HEIGHT,
    min_line_gap: int = config.MIN_LINE_GAP,
    word_dilation_w: int = config.WORD_DILATION_KERNEL_W,
    min_char_area: int = config.MIN_CHAR_AREA,
    min_char_width: int = config.MIN_CHAR_WIDTH,
    min_char_height: int = config.MIN_CHAR_HEIGHT,
    max_char_aspect_ratio: float = config.MAX_CHAR_ASPECT_RATIO,
    enable_ligature_split: bool = True,
) -> SegmentationResult:
    """
    Executes complete hierarchical segmentation (Line -> Word -> Character).

    Args:
        binary_mask: Inverted binary image (255 = ink, 0 = paper background).
        gray_image: Optional grayscale image for grayscale character crop extraction.
        smoothing_sigma: HPP smoothing sigma.
        line_valley_ratio: Line detection threshold ratio against max HPP.
        min_line_height: Minimum line height in pixels.
        min_line_gap: Gap below which consecutive lines are merged.
        word_dilation_w: Horizontal dilation kernel width for word bridging.
        min_char_area: Noise filtering minimum area threshold.
        min_char_width: Minimum character width.
        min_char_height: Minimum character height.
        max_char_aspect_ratio: Ruling line / scratch filter ratio.
        enable_ligature_split: Whether to split joined cursive characters.

    Returns:
        SegmentationResult containing hierarchical Line, Word, and CharacterCrop objects.
    """
    if binary_mask is None or binary_mask.size == 0:
        raise ValueError("[Phase 3 Error] Input binary mask is empty or None.")

    h, w = binary_mask.shape[:2]
    result = SegmentationResult(lines=[], image_shape=(h, w))

    # 1. Detect text lines
    line_boxes = detect_lines(
        binary_mask,
        smoothing_sigma=smoothing_sigma,
        valley_threshold_ratio=line_valley_ratio,
        min_line_height=min_line_height,
        min_line_gap=min_line_gap,
    )

    global_word_id = 0

    for line_id, (lx, ly, lw, lh) in enumerate(line_boxes):
        line_binary = binary_mask[ly:ly+lh, lx:lx+lw]
        line_obj = Line(line_id=line_id, bbox=(lx, ly, lw, lh), words=[])

        # 2. Detect words within line
        word_boxes = detect_words(
            line_binary,
            line_offset=(lx, ly),
            word_dilation_w=word_dilation_w,
        )

        for wx, wy, ww, wh in word_boxes:
            word_binary = binary_mask[wy:wy+wh, wx:wx+ww]
            word_obj = Word(word_id=global_word_id, bbox=(wx, wy, ww, wh), characters=[], line_id=line_id)

            # 3. Extract characters within word
            chars = extract_characters(
                word_binary=word_binary,
                word_offset=(wx, wy),
                full_binary_mask=binary_mask,
                full_gray_image=gray_image,
                word_id=global_word_id,
                line_id=line_id,
                min_area=min_char_area,
                min_w=min_char_width,
                min_h=min_char_height,
                max_aspect_ratio=max_char_aspect_ratio,
                enable_ligature_split=enable_ligature_split,
            )

            # Only append non-empty words
            if chars:
                word_obj.characters = chars
                line_obj.words.append(word_obj)
                global_word_id += 1

        # Only append non-empty lines
        if line_obj.words:
            result.lines.append(line_obj)

    return result


# -------------------------------------------------------------------------
# Step 5: Diagnostic Visual Overlay Rendering
# -------------------------------------------------------------------------

def render_segmentation_overlay(
    base_image: Optional[np.ndarray],
    segmentation_result: SegmentationResult,
    save_path: Optional[Union[str, Path]] = None,
) -> np.ndarray:
    """
    Renders diagnostic visualization highlighting:
    - Lines in Blue (BGR: 255, 0, 0)
    - Words in Green (BGR: 0, 255, 0)
    - Characters in Red (BGR: 0, 0, 255)

    Args:
        base_image: Canvas image (BGR or grayscale). If None, renders on white canvas.
        segmentation_result: Completed SegmentationResult hierarchy.
        save_path: Optional path to save visual PNG image.

    Returns:
        3-channel BGR image with drawn bounding box overlays.
    """
    h, w = segmentation_result.image_shape
    if base_image is not None:
        if len(base_image.shape) == 2:
            canvas = cv2.cvtColor(base_image, cv2.COLOR_GRAY2BGR)
        else:
            canvas = base_image.copy()
    else:
        canvas = np.full((h, w, 3), 255, dtype=np.uint8)

    # 1. Draw Lines (Blue)
    for line in segmentation_result.lines:
        lx, ly, lw, lh = line.bbox
        cv2.rectangle(canvas, (lx, ly), (lx + lw, ly + lh), (255, 0, 0), 2)
        cv2.putText(
            canvas,
            f"L{line.line_id}",
            (lx + 4, ly + 14),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.4,
            (255, 0, 0),
            1,
            cv2.LINE_AA,
        )

        # 2. Draw Words (Green)
        for word in line.words:
            wx, wy, ww, wh = word.bbox
            cv2.rectangle(canvas, (wx, wy), (wx + ww, wy + wh), (0, 255, 0), 2)

            # 3. Draw Characters (Red)
            for char in word.characters:
                cx, cy, cw, ch = char.bbox
                cv2.rectangle(canvas, (cx, cy), (cx + cw, cy + ch), (0, 0, 255), 1)

    if save_path is not None:
        path_obj = Path(save_path)
        path_obj.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(path_obj), canvas)

    return canvas


class SegmentationEngine:
    """
    Object-oriented encapsulation of Phase 3 Segmentation for modular execution.
    """

    def __init__(
        self,
        smoothing_sigma: float = config.HPP_SMOOTHING_SIGMA,
        line_valley_ratio: float = config.LINE_VALLEY_THRESHOLD_RATIO,
        min_line_height: int = config.MIN_LINE_HEIGHT,
        min_line_gap: int = config.MIN_LINE_GAP,
        word_dilation_w: int = config.WORD_DILATION_KERNEL_W,
        min_char_area: int = config.MIN_CHAR_AREA,
        min_char_width: int = config.MIN_CHAR_WIDTH,
        min_char_height: int = config.MIN_CHAR_HEIGHT,
        max_char_aspect_ratio: float = config.MAX_CHAR_ASPECT_RATIO,
        enable_ligature_split: bool = True,
    ):
        self.smoothing_sigma = smoothing_sigma
        self.line_valley_ratio = line_valley_ratio
        self.min_line_height = min_line_height
        self.min_line_gap = min_line_gap
        self.word_dilation_w = word_dilation_w
        self.min_char_area = min_char_area
        self.min_char_width = min_char_width
        self.min_char_height = min_char_height
        self.max_char_aspect_ratio = max_char_aspect_ratio
        self.enable_ligature_split = enable_ligature_split

    def segment(
        self,
        binary_mask: np.ndarray,
        gray_image: Optional[np.ndarray] = None,
    ) -> SegmentationResult:
        """Executes hierarchical segmentation on an input binary mask."""
        return segment_document(
            binary_mask=binary_mask,
            gray_image=gray_image,
            smoothing_sigma=self.smoothing_sigma,
            line_valley_ratio=self.line_valley_ratio,
            min_line_height=self.min_line_height,
            min_line_gap=self.min_line_gap,
            word_dilation_w=self.word_dilation_w,
            min_char_area=self.min_char_area,
            min_char_width=self.min_char_width,
            min_char_height=self.min_char_height,
            max_char_aspect_ratio=self.max_char_aspect_ratio,
            enable_ligature_split=self.enable_ligature_split,
        )
