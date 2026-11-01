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
import json
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
    raw_text: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "word_id": self.word_id,
            "line_id": self.line_id,
            "text": self.text,
            "raw_text": self.raw_text,
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
# Lexicon Post-Correction Engine
# -------------------------------------------------------------------------

_CACHED_LEXICON_ENGINE = None


class LexiconEngine:
    """
    Loads and manages vocabulary and vectorized probability scoring for OCR post-correction.
    """

    def __init__(self, lexicon_path: Optional[Union[str, Path]] = None):
        self.lexicon_path = Path(lexicon_path or getattr(config, "LEXICON_FILE", config.MODELS_DIR / "lexicon.txt"))
        self.clean_lexicon: set = set()
        self.lexicon_by_len: Dict[int, List[str]] = {}
        self.lexicon_arrays: Dict[int, Tuple[List[str], np.ndarray]] = {}
        self.char_to_classes: Dict[str, List[int]] = {}
        self._load_lexicon()
        self._load_label_mapping()

    def _load_lexicon(self):
        if not self.lexicon_path.exists():
            sys_dict = Path("/usr/share/dict/words")
            if sys_dict.exists():
                self.lexicon_path = sys_dict

        if self.lexicon_path.exists():
            with open(self.lexicon_path, "r", encoding="utf-8", errors="ignore") as f:
                for line in f:
                    w = line.strip().lower()
                    if w and all("a" <= c <= "z" for c in w):
                        self.clean_lexicon.add(w)
                        self.lexicon_by_len.setdefault(len(w), []).append(w)

        # Incorporate domain/custom vocabulary from config
        custom_words = getattr(config, "CUSTOM_VOCABULARY", [])
        for w in custom_words:
            w_clean = w.strip().lower()
            if w_clean and all("a" <= c <= "z" for c in w_clean):
                self.clean_lexicon.add(w_clean)
                if w_clean not in self.lexicon_by_len.get(len(w_clean), []):
                    self.lexicon_by_len.setdefault(len(w_clean), []).append(w_clean)

        for L, words in self.lexicon_by_len.items():
            arr = np.array([[ord(c) - ord("a") for c in w] for w in words], dtype=np.int32)
            self.lexicon_arrays[L] = (words, arr)

    def _load_label_mapping(self):
        map_file = getattr(config, "LABEL_MAP_FILE", config.MODELS_DIR / "label_mapping.json")
        if Path(map_file).exists():
            with open(map_file, "r", encoding="utf-8") as f:
                mapping = json.load(f)
            for idx_str, ch in mapping.items():
                self.char_to_classes.setdefault(ch.lower(), []).append(int(idx_str))

    def correct_word(
        self,
        char_preds: List[Any],
        min_avg_prob: float = getattr(config, "LEXICON_MIN_AVG_PROB", 0.02),
    ) -> Tuple[str, float, str]:
        """
        Corrects an ordered list of character predictions using probability-weighted lexicon scoring.

        Returns:
            Tuple of (corrected_text, confidence, raw_text).
        """
        L = len(char_preds)
        raw_word = "".join(getattr(p, "predicted_char", "") for p in char_preds)
        raw_conf = float(np.mean([getattr(p, "confidence", 1.0) for p in char_preds])) if char_preds else 0.0

        alpha_chars = [c for c in raw_word if c.isalpha()]
        if not alpha_chars or not self.clean_lexicon:
            return raw_word, raw_conf, raw_word

        upper_ratio = sum(1 for c in alpha_chars if c.isupper()) / max(1, len(alpha_chars))
        is_upper = (upper_ratio >= 0.60)
        is_title = (not is_upper) and (raw_word[0].isupper() or raw_word[0] in "123456789")

        low_word = raw_word.lower()
        if low_word in self.clean_lexicon:
            formatted = raw_word.upper() if is_upper else (raw_word.capitalize() if is_title else raw_word)
            return formatted, raw_conf, raw_word

        # Fast digit-letter substitution (0 -> o, 1 -> i, 5 -> s, 8 -> b, etc.)
        sub_map = {"0": "o", "1": "i", "5": "s", "8": "b", "2": "z", "9": "i", "7": "t", "4": "a"}
        digit_fixed = "".join(sub_map.get(c, c) for c in low_word)
        if digit_fixed in self.clean_lexicon:
            formatted = digit_fixed.upper() if is_upper else (digit_fixed.capitalize() if is_title else digit_fixed)
            return formatted, max(raw_conf, 0.85), raw_word

        # Vectorized beam search over same-length candidate words
        if L in self.lexicon_arrays:
            cands, cand_indices = self.lexicon_arrays[L]
            logp_matrix = np.full((L, 26), -15.0, dtype=np.float32)
            for i, p in enumerate(char_preds):
                probs = getattr(p, "probabilities", None)
                if probs is not None:
                    for c_idx in range(26):
                        char_letter = chr(ord("a") + c_idx)
                        classes = self.char_to_classes.get(char_letter, [])
                        prob = sum(probs[c] for c in classes) if classes else 1e-6
                        logp_matrix[i, c_idx] = np.log(max(prob, 1e-6))
                else:
                    p_char = getattr(p, "predicted_char", "").lower()
                    if "a" <= p_char <= "z":
                        conf = getattr(p, "confidence", 0.5)
                        logp_matrix[i, ord(p_char) - ord("a")] = np.log(max(conf, 0.5))

            word_scores = np.sum(logp_matrix[np.arange(L), cand_indices], axis=1)

            # Apply unigram domain prior boost to domain vocabulary terms
            custom_set = set(getattr(config, "CUSTOM_VOCABULARY", []))
            for idx_w, w in enumerate(cands):
                if w in custom_set:
                    word_scores[idx_w] += 7.0

            best_idx = int(np.argmax(word_scores))
            best_word = cands[best_idx]
            best_score = float(word_scores[best_idx])
            avg_prob = float(np.exp(best_score / L))

            if avg_prob >= min_avg_prob:
                formatted = best_word.upper() if is_upper else (best_word.capitalize() if is_title else best_word)
                return formatted, avg_prob, raw_word

        # Fallback: Digit-to-letter conversion for predominantly alphabetic word
        if len(alpha_chars) / max(1, len(raw_word)) >= 0.5:
            fixed = "".join(sub_map.get(c, c) for c in raw_word)
            return fixed, raw_conf, raw_word

        return raw_word, raw_conf, raw_word


def get_default_lexicon_engine() -> LexiconEngine:
    """Singleton accessor for default LexiconEngine."""
    global _CACHED_LEXICON_ENGINE
    if _CACHED_LEXICON_ENGINE is None:
        _CACHED_LEXICON_ENGINE = LexiconEngine()
    return _CACHED_LEXICON_ENGINE


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
    enable_lexicon: bool = getattr(config, "ENABLE_LEXICON_CORRECTION", True),
) -> ReconstructionResult:
    """
    Reconstructs recognized character predictions into structured text lines and words.

    Follows reading order hierarchy:
      - Lines ordered top-to-bottom.
      - Words ordered left-to-right within lines.
      - Characters ordered left-to-right within words.
    Detects paragraph spacing when vertical gap between lines exceeds
    paragraph_gap_factor * median_line_height.
    Applies probability-weighted lexicon post-correction when enabled.

    Args:
        predictions: List of PredictionResult objects from Phase 6.
        segmentation_result: Optional SegmentationResult from Phase 3 for line/word geometry.
        paragraph_gap_factor: Multiplier for paragraph gap detection.
        enable_lexicon: Whether to apply lexicon dictionary post-correction.

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

            raw_word_text = "".join(getattr(p, "predicted_char", "") for p in char_preds_sorted)
            word_confs = [getattr(p, "confidence", 1.0) for p in char_preds_sorted]
            word_conf = float(np.mean(word_confs)) if word_confs else 0.0

            # Lexicon post-correction using candidate beam search
            if enable_lexicon and len(char_preds_sorted) > 0:
                lex_engine = get_default_lexicon_engine()
                word_text, word_conf, _ = lex_engine.correct_word(char_preds_sorted)
            else:
                word_text = raw_word_text

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
                raw_text=raw_word_text,
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
    enable_lexicon: bool = getattr(config, "ENABLE_LEXICON_CORRECTION", True),
) -> ReconstructionResult:
    """
    High-level entry point: Reconstructs text and exports directly to file.

    Args:
        predictions: List of PredictionResult objects from Phase 6.
        output_path: Optional explicit output path. If None, derives from image_name or config.
        segmentation_result: Optional SegmentationResult from Phase 3.
        image_name: Optional base name of source photograph.
        paragraph_gap_factor: Multiplier for paragraph gap detection.
        enable_lexicon: Whether to apply lexicon dictionary post-correction.

    Returns:
        ReconstructionResult with output_file_path populated.
    """
    result = reconstruct_text(
        predictions=predictions,
        segmentation_result=segmentation_result,
        paragraph_gap_factor=paragraph_gap_factor,
        enable_lexicon=enable_lexicon,
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
        enable_lexicon: bool = getattr(config, "ENABLE_LEXICON_CORRECTION", True),
    ):
        self.paragraph_gap_factor = paragraph_gap_factor
        self.encoding = encoding
        self.enable_lexicon = enable_lexicon

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
            enable_lexicon=self.enable_lexicon,
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
