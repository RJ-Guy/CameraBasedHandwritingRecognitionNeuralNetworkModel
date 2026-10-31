"""
reconstruction.py - Phase 7: Text Reconstruction and Plaintext Export

Assembles classified character tokens into grammatically and spatially structured
words, lines, and paragraphs; computes document-level telemetry and confidence
metrics; and exports the digitized text into standard UTF-8 encoded files.

Theoretical & Algorithmic Foundations:
1. Spatial Hierarchy Traversal:
   - Lines ordered top-to-bottom: Line_0, Line_1, ...
   - Words within lines ordered left-to-right: Word_0, Word_1, ...
   - Characters within words ordered left-to-right: Char_0, Char_1, ...
2. Spatial Word Boundary Separation:
   - Words are separated with standard ASCII spaces (' ').
3. Dynamic Paragraph Break Detection:
   - Vertical gaps between lines exceeding (paragraph_gap_factor * median_line_height)
     are treated as intentional paragraph breaks and separated by double newlines ('\\n\\n').
4. UTF-8 Plaintext Export:
   - Enforces clean, lossless UTF-8 encoding across all operating systems.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np

try:
    from . import config
except ImportError:
    import config


# -------------------------------------------------------------------------
# Reconstructed Data Structures
# -------------------------------------------------------------------------

@dataclass
class ReconstructedWord:
    """Represents a reconstructed word token composed of ordered characters."""
    word_id: int
    line_id: int
    text: str
    confidence: float
    bbox: Optional[Tuple[int, int, int, int]] = None
    character_count: int = 0
    characters: List[Any] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "word_id": self.word_id,
            "line_id": self.line_id,
            "text": self.text,
            "confidence": round(float(self.confidence), 4),
            "bbox": list(self.bbox) if self.bbox is not None else None,
            "character_count": self.character_count,
        }


@dataclass
class ReconstructedLine:
    """Represents a reconstructed text line composed of ordered words."""
    line_id: int
    text: str
    words: List[ReconstructedWord] = field(default_factory=list)
    confidence: float = 0.0
    bbox: Optional[Tuple[int, int, int, int]] = None

    @property
    def word_count(self) -> int:
        return len(self.words)

    @property
    def character_count(self) -> int:
        return sum(w.character_count for w in self.words)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "line_id": self.line_id,
            "text": self.text,
            "confidence": round(float(self.confidence), 4),
            "bbox": list(self.bbox) if self.bbox is not None else None,
            "words": [w.to_dict() for w in self.words],
        }


@dataclass
class ReconstructionResult:
    """Complete document text reconstruction container."""
    full_text: str
    lines: List[ReconstructedLine] = field(default_factory=list)
    total_lines: int = 0
    total_words: int = 0
    total_characters: int = 0
    mean_confidence: float = 0.0
    output_file_path: Optional[Path] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "total_lines": self.total_lines,
            "total_words": self.total_words,
            "total_characters": self.total_characters,
            "mean_confidence": round(float(self.mean_confidence), 4),
            "full_text": self.full_text,
            "output_file_path": str(self.output_file_path) if self.output_file_path else None,
            "lines": [ln.to_dict() for ln in self.lines],
        }


# -------------------------------------------------------------------------
# Core Reconstruction Engine
# -------------------------------------------------------------------------

def compute_enclosing_bbox(bboxes: List[Tuple[int, int, int, int]]) -> Optional[Tuple[int, int, int, int]]:
    """Calculates the minimal bounding box enclosing a list of (x, y, w, h) boxes."""
    valid_boxes = [b for b in bboxes if b is not None and len(b) == 4]
    if not valid_boxes:
        return None
    min_x = min(b[0] for b in valid_boxes)
    min_y = min(b[1] for b in valid_boxes)
    max_x = max(b[0] + b[2] for b in valid_boxes)
    max_y = max(b[1] + b[3] for b in valid_boxes)
    return (int(min_x), int(min_y), int(max_x - min_x), int(max_y - min_y))


def reconstruct_text(
    predictions: List[Any],
    segmentation_result: Optional[Any] = None,
    paragraph_gap_factor: float = 1.8,
) -> ReconstructionResult:
    """
    Reconstructs recognized character predictions into structured text lines and words.

    Follows reading order hierarchy:
      - Lines ordered top-to-bottom.
      - Words ordered left-to-right within lines.
      - Characters ordered left-to-right within words.
    Detects paragraph spacing when vertical gap between lines exceeds
    paragraph_gap_factor * median_line_height.

    Args:
        predictions: List of PredictionResult objects from Phase 6.
        segmentation_result: Optional SegmentationResult from Phase 3 for line/word geometry.
        paragraph_gap_factor: Multiplier for paragraph gap detection.

    Returns:
        ReconstructionResult dataclass instance.
    """
    if not predictions:
        return ReconstructionResult(
            full_text="",
            lines=[],
            total_lines=0,
            total_words=0,
            total_characters=0,
            mean_confidence=0.0,
        )

    # 1. Organize characters into lines and words
    # Grouping structure: lines_dict[line_id][word_id] = [pred, ...]
    lines_dict: Dict[int, Dict[int, List[Any]]] = {}

    for pred in predictions:
        line_id = getattr(pred, "line_id", 0)
        word_id = getattr(pred, "word_id", 0)

        if line_id not in lines_dict:
            lines_dict[line_id] = {}
        if word_id not in lines_dict[line_id]:
            lines_dict[line_id][word_id] = []

        lines_dict[line_id][word_id].append(pred)

    # 2. Sort lines top-to-bottom
    # Sort key: line_id, or bounding box y if segmentation_result provides it
    line_y_map: Dict[int, int] = {}
    line_height_map: Dict[int, int] = {}
    if segmentation_result is not None and hasattr(segmentation_result, "lines"):
        for ln in segmentation_result.lines:
            line_y_map[ln.line_id] = ln.y
            line_height_map[ln.line_id] = ln.height

    def get_line_sort_key(lid: int) -> float:
        if lid in line_y_map:
            return float(line_y_map[lid])
        # Fallback to mean y of character bboxes in this line
        char_ys = [
            p.bbox[1] for wdict in lines_dict[lid].values() for p in wdict
            if getattr(p, "bbox", None) is not None
        ]
        return float(np.mean(char_ys)) if char_ys else float(lid)

    sorted_line_ids = sorted(lines_dict.keys(), key=get_line_sort_key)

    # 3. Assemble words and lines
    reconstructed_lines: List[ReconstructedLine] = []

    for lid in sorted_line_ids:
        words_dict = lines_dict[lid]

        # Sort words left-to-right
        def get_word_sort_key(wid: int) -> float:
            char_xs = [
                p.bbox[0] for p in words_dict[wid]
                if getattr(p, "bbox", None) is not None
            ]
            return float(min(char_xs)) if char_xs else float(wid)

        sorted_word_ids = sorted(words_dict.keys(), key=get_word_sort_key)

        reconstructed_words: List[ReconstructedWord] = []
        for wid in sorted_word_ids:
            char_preds = words_dict[wid]

            # Sort characters strictly left-to-right within the word
            char_preds_sorted = sorted(
                char_preds,
                key=lambda p: p.bbox[0] if getattr(p, "bbox", None) is not None else getattr(p, "char_id", 0)
            )

            word_text = "".join(getattr(p, "predicted_char", "") for p in char_preds_sorted)
            word_confs = [getattr(p, "confidence", 1.0) for p in char_preds_sorted]
            word_conf = float(np.mean(word_confs)) if word_confs else 0.0

            char_bboxes = [getattr(p, "bbox", None) for p in char_preds_sorted]
            word_bbox = compute_enclosing_bbox(char_bboxes)

            rec_word = ReconstructedWord(
                word_id=wid,
                line_id=lid,
                text=word_text,
                confidence=word_conf,
                bbox=word_bbox,
                character_count=len(char_preds_sorted),
                characters=char_preds_sorted,
            )
            reconstructed_words.append(rec_word)

        # Assemble full line text with single space delimiter
        line_text = " ".join(w.text for w in reconstructed_words if w.text)
        line_confs = [w.confidence for w in reconstructed_words]
        line_conf = float(np.mean(line_confs)) if line_confs else 0.0

        # Compute line bounding box
        word_bboxes = [w.bbox for w in reconstructed_words if w.bbox is not None]
        line_bbox = compute_enclosing_bbox(word_bboxes)

        rec_line = ReconstructedLine(
            line_id=lid,
            text=line_text,
            words=reconstructed_words,
            confidence=line_conf,
            bbox=line_bbox,
        )
        reconstructed_lines.append(rec_line)

    # 4. Form full text with dynamic paragraph break detection
    # Calculate median line height to determine paragraph spacing threshold
    line_heights: List[float] = []
    for rline in reconstructed_lines:
        if rline.bbox is not None:
            line_heights.append(float(rline.bbox[3]))
        elif rline.line_id in line_height_map:
            line_heights.append(float(line_height_map[rline.line_id]))

    median_height = float(np.median(line_heights)) if line_heights else 20.0
    paragraph_threshold = paragraph_gap_factor * median_height

    text_segments: List[str] = []
    for i, rline in enumerate(reconstructed_lines):
        text_segments.append(rline.text)

        # Check gap between this line and the next
        if i < len(reconstructed_lines) - 1:
            next_line = reconstructed_lines[i + 1]
            is_paragraph_break = False

            if rline.bbox is not None and next_line.bbox is not None:
                current_bottom = rline.bbox[1] + rline.bbox[3]
                next_top = next_line.bbox[1]
                gap = next_top - current_bottom
                if gap > paragraph_threshold:
                    is_paragraph_break = True

            text_segments.append("\n\n" if is_paragraph_break else "\n")

    full_reconstructed_text = "".join(text_segments)

    # 5. Document-level statistics
    total_chars = sum(ln.character_count for ln in reconstructed_lines)
    total_words = sum(ln.word_count for ln in reconstructed_lines)
    all_confs = [getattr(p, "confidence", 0.0) for p in predictions]
    mean_conf = float(np.mean(all_confs)) if all_confs else 0.0

    return ReconstructionResult(
        full_text=full_reconstructed_text,
        lines=reconstructed_lines,
        total_lines=len(reconstructed_lines),
        total_words=total_words,
        total_characters=total_chars,
        mean_confidence=mean_conf,
    )


# -------------------------------------------------------------------------
# File Export Utility
# -------------------------------------------------------------------------

def export_text_file(
    text: str,
    output_path: Union[str, Path],
    encoding: str = config.OUTPUT_TEXT_ENCODING,
) -> Path:
    """
    Exports reconstructed text string to a standard UTF-8 plaintext file.

    Args:
        text: Reconstructed document text.
        output_path: Destination file path (*.txt).
        encoding: File character encoding (default "utf-8").

    Returns:
        Resolved Path object of the exported file.
    """
    dest = Path(output_path).resolve()
    dest.parent.mkdir(parents=True, exist_ok=True)

    with open(dest, "w", encoding=encoding) as f:
        f.write(text)

    return dest


def reconstruct_and_export(
    predictions: List[Any],
    output_path: Optional[Union[str, Path]] = None,
    segmentation_result: Optional[Any] = None,
    image_name: Optional[str] = None,
    paragraph_gap_factor: float = 1.8,
) -> ReconstructionResult:
    """
    High-level entry point: Reconstructs text and exports directly to file.

    Args:
        predictions: List of PredictionResult objects from Phase 6.
        output_path: Optional explicit output path. If None, derives from image_name or config.
        segmentation_result: Optional SegmentationResult from Phase 3.
        image_name: Optional base name of source photograph.
        paragraph_gap_factor: Multiplier for paragraph gap detection.

    Returns:
        ReconstructionResult with output_file_path populated.
    """
    result = reconstruct_text(
        predictions=predictions,
        segmentation_result=segmentation_result,
        paragraph_gap_factor=paragraph_gap_factor,
    )

    if output_path is None:
        base_stem = Path(image_name).stem if image_name else "document"
        target_path = config.TEXT_OUTPUT_DIR / f"{base_stem}_recognized.txt"
    else:
        target_path = Path(output_path)

    saved_path = export_text_file(result.full_text, target_path)
    result.output_file_path = saved_path
    return result


class ReconstructionEngine:
    """
    Object-oriented encapsulation of Phase 7 Text Reconstruction.
    """

    def __init__(
        self,
        paragraph_gap_factor: float = 1.8,
        encoding: str = config.OUTPUT_TEXT_ENCODING,
    ):
        self.paragraph_gap_factor = paragraph_gap_factor
        self.encoding = encoding

    def reconstruct(
        self,
        predictions: List[Any],
        segmentation_result: Optional[Any] = None,
    ) -> ReconstructionResult:
        """Assembles character predictions into structured text."""
        return reconstruct_text(
            predictions=predictions,
            segmentation_result=segmentation_result,
            paragraph_gap_factor=self.paragraph_gap_factor,
        )

    def export(
        self,
        result: ReconstructionResult,
        output_path: Union[str, Path],
    ) -> Path:
        """Exports a ReconstructionResult to a UTF-8 file."""
        saved_path = export_text_file(result.full_text, output_path, encoding=self.encoding)
        result.output_file_path = saved_path
        return saved_path
