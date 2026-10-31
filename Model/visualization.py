"""
visualization.py - Phase 7: Diagnostic Visual Canvas & HUD Overlays

Renders high-resolution bounding boxes, character labels, confidence score tags,
and document-level telemetry banners directly on the original or flattened image canvas.

Key Visual Features:
1. Solid-Background Character HUDs:
   - High-contrast text labels with opaque background rectangles for maximum legibility
     against textured paper, shadows, and varied desk backgrounds.
2. Color-Coded Structural Hierarchy:
   - Words: Forest Green bounding boxes (thickness=2).
   - Characters: Cyan / Light Blue bounding boxes (thickness=1).
3. Telemetry Status Banner:
   - Summary bar rendered at the top of the canvas displaying line, word, character counts,
     and overall mean recognition confidence.
4. Optional Inverse Homography Projection:
   - Capable of projecting flattened-space bounding boxes back onto the original angled photograph.
"""

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
import cv2
import numpy as np

try:
    from . import config
except ImportError:
    import config


# -------------------------------------------------------------------------
# Coordinate Transformation Helpers
# -------------------------------------------------------------------------

def transform_bbox_inverse(
    bbox: Tuple[int, int, int, int],
    inv_homography: np.ndarray,
) -> Tuple[int, int, int, int]:
    """
    Transforms an axis-aligned bounding box from flattened space back to
    original image space using the inverse homography matrix M^-1.

    Args:
        bbox: (x, y, w, h) in flattened space.
        inv_homography: 3x3 float64 inverse perspective matrix.

    Returns:
        Axis-aligned bounding box (x, y, w, h) enclosing the transformed quad.
    """
    x, y, w, h = bbox
    corners = np.array([
        [[x, y]],
        [[x + w, y]],
        [[x + w, y + h]],
        [[x, y + h]],
    ], dtype=np.float32)

    transformed_corners = cv2.perspectiveTransform(corners, inv_homography)
    pts = transformed_corners.reshape(-1, 2)

    min_x = int(np.floor(np.min(pts[:, 0])))
    min_y = int(np.floor(np.min(pts[:, 1])))
    max_x = int(np.ceil(np.max(pts[:, 0])))
    max_y = int(np.ceil(np.max(pts[:, 1])))

    return (max(0, min_x), max(0, min_y), max(1, max_x - min_x), max(1, max_y - min_y))


# -------------------------------------------------------------------------
# Core Rendering Functions
# -------------------------------------------------------------------------

def render_annotated_document(
    image: np.ndarray,
    predictions: List[Any],
    segmentation_result: Optional[Any] = None,
    output_path: Optional[Union[str, Path]] = None,
    show_confidences: bool = config.DRAW_CONFIDENCE_SCORES,
    draw_words: bool = True,
    draw_chars: bool = True,
    draw_header_hud: bool = True,
    inverse_homography: Optional[np.ndarray] = None,
) -> np.ndarray:
    """
    Renders bounding boxes, character labels, and telemetry overlays onto the document canvas.

    Args:
        image: Base image array (untouched original or flattened paper).
        predictions: List of PredictionResult objects from Phase 6.
        segmentation_result: Optional SegmentationResult from Phase 3.
        output_path: Optional destination file path (*.png).
        show_confidences: If True, renders confidence percentages alongside characters.
        draw_words: If True, draws green bounding boxes around words.
        draw_chars: If True, draws cyan bounding boxes around characters.
        draw_header_hud: If True, renders top telemetry banner.
        inverse_homography: Optional 3x3 homography matrix to project boxes to original canvas.

    Returns:
        Annotated 3-channel BGR image array.
    """
    # 1. Ensure working on a 3-channel BGR copy
    if image.ndim == 2:
        canvas = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    elif image.ndim == 3 and image.shape[2] == 4:
        canvas = cv2.cvtColor(image, cv2.COLOR_BGRA2BGR)
    else:
        canvas = image.copy()

    canvas_h, canvas_w = canvas.shape[:2]

    # Color Palette (BGR)
    COLOR_WORD_BOX = (0, 200, 0)          # Forest green
    COLOR_CHAR_BOX = (255, 180, 0)        # Bright cyan / sky blue
    COLOR_HUD_BG = (25, 25, 25)           # Dark charcoal for label pill
    COLOR_HUD_TEXT = (255, 255, 255)      # Crisp white text
    COLOR_CONF_TEXT = (0, 220, 255)       # Gold / yellow confidence
    COLOR_BANNER_BG = (20, 20, 20)        # Dark top banner
    COLOR_BANNER_TEXT = (240, 240, 240)   # Clean banner text

    # Dynamic scaling based on canvas resolution
    base_dim = max(canvas_w, canvas_h)
    scale_factor = max(0.5, min(1.8, base_dim / 1200.0))
    char_font_scale = 0.42 * scale_factor
    banner_font_scale = 0.50 * scale_factor
    thickness = max(1, int(1.2 * scale_factor))

    # 2. Draw Word Bounding Boxes (if segmentation_result is available)
    if draw_words:
        word_boxes: List[Tuple[int, int, int, int]] = []
        if segmentation_result is not None and hasattr(segmentation_result, "lines"):
            for line in segmentation_result.lines:
                for word in line.words:
                    word_boxes.append(word.bbox)
        else:
            # Group predictions by word_id and construct word bounding boxes
            word_char_map: Dict[int, List[Any]] = {}
            for pred in predictions:
                wid = getattr(pred, "word_id", 0)
                word_char_map.setdefault(wid, []).append(pred)
            for wid, char_list in word_char_map.items():
                bboxes = [p.bbox for p in char_list if getattr(p, "bbox", None) is not None]
                if bboxes:
                    min_x = min(b[0] for b in bboxes)
                    min_y = min(b[1] for b in bboxes)
                    max_x = max(b[0] + b[2] for b in bboxes)
                    max_y = max(b[1] + b[3] for b in bboxes)
                    word_boxes.append((min_x, min_y, max_x - min_x, max_y - min_y))

        for wbox in word_boxes:
            if inverse_homography is not None:
                wbox = transform_bbox_inverse(wbox, inverse_homography)
            wx, wy, ww, wh = wbox
            cv2.rectangle(
                canvas,
                (wx, wy),
                (wx + ww, wy + wh),
                COLOR_WORD_BOX,
                thickness=max(1, int(2 * scale_factor)),
                lineType=cv2.LINE_AA,
            )

    # 3. Draw Character Bounding Boxes and HUD Labels
    for pred in predictions:
        bbox = getattr(pred, "bbox", None)
        if bbox is None:
            continue

        if inverse_homography is not None:
            bbox = transform_bbox_inverse(bbox, inverse_homography)

        cx, cy, cw, ch = bbox

        # Draw character box
        if draw_chars:
            cv2.rectangle(
                canvas,
                (cx, cy),
                (cx + cw, cy + ch),
                COLOR_CHAR_BOX,
                thickness=max(1, int(1 * scale_factor)),
                lineType=cv2.LINE_AA,
            )

        # Build label text
        char_str = getattr(pred, "predicted_char", "?")
        conf_val = getattr(pred, "confidence", 1.0)

        if show_confidences:
            label_text = f"{char_str} {int(round(conf_val * 100))}%"
        else:
            label_text = f"{char_str}"

        # Calculate text size
        (text_w, text_h), baseline = cv2.getTextSize(
            label_text,
            cv2.FONT_HERSHEY_SIMPLEX,
            char_font_scale,
            thickness=1,
        )

        # Position label above character box (or below if near top boundary)
        pad = 2
        label_x1 = max(0, cx)
        label_y2 = cy - 2
        label_y1 = label_y2 - (text_h + pad * 2)

        # If box is too close to top edge, flip label below the character box
        if label_y1 < (30 if draw_header_hud else 0):
            label_y1 = cy + ch + 2
            label_y2 = label_y1 + (text_h + pad * 2)

        label_x2 = min(canvas_w - 1, label_x1 + text_w + pad * 2)

        # Draw opaque background pill for high contrast
        cv2.rectangle(
            canvas,
            (label_x1, label_y1),
            (label_x2, label_y2),
            COLOR_HUD_BG,
            cv2.FILLED,
        )
        # Subtle border on label pill
        cv2.rectangle(
            canvas,
            (label_x1, label_y1),
            (label_x2, label_y2),
            COLOR_CHAR_BOX,
            1,
            lineType=cv2.LINE_AA,
        )

        # Render label text
        text_pos = (label_x1 + pad, label_y2 - pad - 1)
        cv2.putText(
            canvas,
            label_text,
            text_pos,
            cv2.FONT_HERSHEY_SIMPLEX,
            char_font_scale,
            COLOR_HUD_TEXT,
            1,
            lineType=cv2.LINE_AA,
        )

    # 4. Draw Header Telemetry Banner
    if draw_header_hud and canvas_h > 100:
        banner_h = int(32 * scale_factor)
        # Overlay dark banner with alpha blending
        overlay = canvas.copy()
        cv2.rectangle(overlay, (0, 0), (canvas_w, banner_h), COLOR_BANNER_BG, cv2.FILLED)
        cv2.addWeighted(overlay, 0.85, canvas, 0.15, 0, canvas)
        cv2.line(canvas, (0, banner_h), (canvas_w, banner_h), COLOR_WORD_BOX, 1, lineType=cv2.LINE_AA)

        # Compute summary stats
        total_chars = len(predictions)
        confs = [getattr(p, "confidence", 0.0) for p in predictions]
        mean_conf = float(np.mean(confs)) * 100 if confs else 0.0

        if segmentation_result is not None:
            n_lines = getattr(segmentation_result, "total_lines", 0)
            n_words = getattr(segmentation_result, "total_words", 0)
        else:
            line_ids = set(getattr(p, "line_id", 0) for p in predictions)
            word_ids = set((getattr(p, "line_id", 0), getattr(p, "word_id", 0)) for p in predictions)
            n_lines = len(line_ids)
            n_words = len(word_ids)

        banner_text = (
            f"RECOGNITION HUD | Lines: {n_lines} | Words: {n_words} | "
            f"Characters: {total_chars} | Mean Conf: {mean_conf:.1f}% | "
            f"Model: Classical CV + Shallow MLP"
        )

        cv2.putText(
            canvas,
            banner_text,
            (int(10 * scale_factor), int(20 * scale_factor)),
            cv2.FONT_HERSHEY_SIMPLEX,
            banner_font_scale,
            COLOR_BANNER_TEXT,
            1,
            lineType=cv2.LINE_AA,
        )

    # 5. Export to disk if path is provided
    if output_path is not None:
        out_p = Path(output_path).resolve()
        out_p.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(out_p), canvas)

    return canvas


def render_side_by_side_comparison(
    original_image: np.ndarray,
    annotated_image: np.ndarray,
    output_path: Optional[Union[str, Path]] = None,
) -> np.ndarray:
    """
    Renders the original photograph and annotated OCR image side-by-side.

    Args:
        original_image: Untouched input photograph.
        annotated_image: Annotated OCR canvas.
        output_path: Optional destination file path.

    Returns:
        Side-by-side comparison image array.
    """
    # Normalize channels to 3-channel BGR
    img1 = cv2.cvtColor(original_image, cv2.COLOR_GRAY2BGR) if original_image.ndim == 2 else original_image.copy()
    img2 = cv2.cvtColor(annotated_image, cv2.COLOR_GRAY2BGR) if annotated_image.ndim == 2 else annotated_image.copy()

    # Match heights
    h1, w1 = img1.shape[:2]
    h2, w2 = img2.shape[:2]
    target_h = max(h1, h2)

    if h1 != target_h:
        new_w1 = int(w1 * (target_h / h1))
        img1 = cv2.resize(img1, (new_w1, target_h), interpolation=cv2.INTER_LINEAR)
    if h2 != target_h:
        new_w2 = int(w2 * (target_h / h2))
        img2 = cv2.resize(img2, (new_w2, target_h), interpolation=cv2.INTER_LINEAR)

    comparison = np.hstack([img1, img2])

    if output_path is not None:
        out_p = Path(output_path).resolve()
        out_p.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(out_p), comparison)

    return comparison


class VisualizationEngine:
    """
    Object-oriented encapsulation of Phase 7 Visual Annotation & HUD Rendering.
    """

    def __init__(
        self,
        show_confidences: bool = config.DRAW_CONFIDENCE_SCORES,
        draw_words: bool = True,
        draw_chars: bool = True,
        draw_header_hud: bool = True,
    ):
        self.show_confidences = show_confidences
        self.draw_words = draw_words
        self.draw_chars = draw_chars
        self.draw_header_hud = draw_header_hud

    def annotate(
        self,
        image: np.ndarray,
        predictions: List[Any],
        segmentation_result: Optional[Any] = None,
        output_path: Optional[Union[str, Path]] = None,
        inverse_homography: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """Renders annotations onto the document image."""
        return render_annotated_document(
            image=image,
            predictions=predictions,
            segmentation_result=segmentation_result,
            output_path=output_path,
            show_confidences=self.show_confidences,
            draw_words=self.draw_words,
            draw_chars=self.draw_chars,
            draw_header_hud=self.draw_header_hud,
            inverse_homography=inverse_homography,
        )
