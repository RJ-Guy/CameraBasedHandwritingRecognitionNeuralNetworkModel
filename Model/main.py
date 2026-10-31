"""
main.py - End-to-End Command Line & Programmatic Pipeline Entry Point

Orchestrates all 7 sequential phases of the classical computer vision and shallow MLP
handwriting recognition system:
  Phase 1: Ingestion & Document Flattening
  Phase 2: Illumination-Robust Preprocessing
  Phase 3: Hierarchical Segmentation (Lines -> Words -> Characters)
  Phase 4: Geometric Normalization (Slant Deskew & 16x16 Resampling)
  Phase 5: Circular DDD Feature Extraction (256-D Vectors)
  Phase 6: Shallow MLP Classification (Softmax & Top-k)
  Phase 7: Reconstruction & Visual Diagnostic Export

Usage Examples:
  python main.py --input data/samples/IMG_20261021_212517.jpg
  python main.py -i data/samples/sample_camera_capture.png --warp --debug
  python main.py -i data/samples/sample_camera_capture_synth.png --no-warp
"""

import argparse
from dataclasses import dataclass, field
import os
from pathlib import Path
import sys
import time
from typing import Any, Dict, List, Optional, Tuple, Union
import cv2
import numpy as np

# Ensure Model directory and workspace are accessible
CURRENT_DIR = Path(__file__).resolve().parent
WORKSPACE_DIR = CURRENT_DIR.parent if CURRENT_DIR.name == "Model" else CURRENT_DIR

for p in [CURRENT_DIR, WORKSPACE_DIR]:
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

try:
    from . import config
    from .ingestion import ingest_image, IngestionResult
    from .preprocessing import preprocess_image, PreprocessingResult
    from .segmentation import segment_document, SegmentationResult
    from .normalization import normalize_all_characters, NormalizationResult
    from .features import extract_features_batch
    from .predict import predict_batch, PredictionResult, MLPInferenceEngine
    from .reconstruction import reconstruct_and_export, ReconstructionResult
    from .visualization import render_annotated_document, render_side_by_side_comparison
except ImportError:
    import config
    from ingestion import ingest_image, IngestionResult
    from preprocessing import preprocess_image, PreprocessingResult
    from segmentation import segment_document, SegmentationResult
    from normalization import normalize_all_characters, NormalizationResult
    from features import extract_features_batch
    from predict import predict_batch, PredictionResult, MLPInferenceEngine
    from reconstruction import reconstruct_and_export, ReconstructionResult
    from visualization import render_annotated_document, render_side_by_side_comparison


@dataclass
class PipelineResult:
    """Comprehensive container holding artifacts across all 7 pipeline phases."""
    input_path: Path
    runtime_seconds: float
    ingestion: IngestionResult
    preprocessing: PreprocessingResult
    segmentation: SegmentationResult
    normalization: NormalizationResult
    features: np.ndarray
    predictions: List[PredictionResult]
    reconstruction: ReconstructionResult
    annotated_image: np.ndarray
    output_text_path: Path
    output_annotated_path: Path
    warnings: List[str] = field(default_factory=list)


def run_pipeline(
    input_path: Union[str, Path],
    output_dir: Optional[Union[str, Path]] = None,
    apply_warp: Optional[bool] = None,
    feature_mode: str = config.FEATURE_MODE,
    top_k_count: int = 3,
    debug: bool = False,
    quiet: bool = False,
) -> PipelineResult:
    """
    Executes the full end-to-end handwriting recognition pipeline on a single image.

    Args:
        input_path: Path to input photograph (JPEG, PNG).
        output_dir: Root directory to save outputs (defaults to config.OUTPUTS_DIR).
        apply_warp: Whether to apply perspective warp (None = use config default).
        feature_mode: Feature extraction mode ("circular_ddd" or "flatten_16x16").
        top_k_count: Number of top candidate predictions per character.
        debug: If True, saves intermediate stage masks, projections, and crops.
        quiet: If True, suppresses formatted console report.

    Returns:
        PipelineResult dataclass instance with all outputs.
    """
    t_start = time.perf_counter()
    src_path = Path(input_path).resolve()
    if not src_path.exists():
        raise FileNotFoundError(f"[Pipeline Error] Input image file not found: {src_path}")

    base_out_dir = Path(output_dir).resolve() if output_dir else config.OUTPUTS_DIR
    text_dir = base_out_dir / "text"
    annotated_dir = base_out_dir / "annotated"
    debug_dir = base_out_dir / "debug"
    for d in [text_dir, annotated_dir, debug_dir]:
        d.mkdir(parents=True, exist_ok=True)

    img_stem = src_path.stem
    warnings: List[str] = []

    # -------------------------------------------------------------------------
    # Phase 1: Ingestion & Document Flattening
    # -------------------------------------------------------------------------
    warp_setting = config.ENABLE_PERSPECTIVE_WARP if apply_warp is None else apply_warp
    ingestion_res = ingest_image(src_path, enable_warp=warp_setting)

    # -------------------------------------------------------------------------
    # Phase 2: Illumination-Robust Preprocessing
    # -------------------------------------------------------------------------
    prep_res = preprocess_image(ingestion_res.flattened_image)

    # -------------------------------------------------------------------------
    # Phase 3: Hierarchical Segmentation
    # -------------------------------------------------------------------------
    seg_res = segment_document(prep_res.binary_mask, prep_res.gray_image)

    if seg_res.total_characters == 0:
        warnings.append("No characters were detected by hierarchical segmentation.")

    # -------------------------------------------------------------------------
    # Phase 4: Geometric Normalization
    # -------------------------------------------------------------------------
    norm_res = normalize_all_characters(seg_res)

    # -------------------------------------------------------------------------
    # Phase 5: Circular DDD Feature Extraction
    # -------------------------------------------------------------------------
    if norm_res.characters:
        feature_matrix = extract_features_batch(norm_res, mode=feature_mode)
    else:
        feature_matrix = np.empty((0, config.FEATURE_VECTOR_DIM), dtype=np.float32)

    # -------------------------------------------------------------------------
    # Phase 6: Shallow MLP Inference
    # -------------------------------------------------------------------------
    engine = MLPInferenceEngine(feature_mode=feature_mode, top_k_count=top_k_count)
    if norm_res.characters:
        predictions = engine.predict_batch(norm_res.characters)
    else:
        predictions = []

    # -------------------------------------------------------------------------
    # Phase 7: Reconstruction & Visual Export
    # -------------------------------------------------------------------------
    text_out_path = text_dir / f"{img_stem}_recognized.txt"
    reconstruction_res = reconstruct_and_export(
        predictions=predictions,
        output_path=text_out_path,
        segmentation_result=seg_res,
        image_name=img_stem,
    )

    annotated_out_path = annotated_dir / f"{img_stem}_annotated.png"
    annotated_img = render_annotated_document(
        image=ingestion_res.flattened_image,
        predictions=predictions,
        segmentation_result=seg_res,
        output_path=annotated_out_path,
        show_confidences=config.DRAW_CONFIDENCE_SCORES,
        draw_words=True,
        draw_chars=True,
        draw_header_hud=True,
    )

    # Optional debug output artifacts
    if debug:
        cv2.imwrite(str(debug_dir / f"{img_stem}_01_ingested.png"), ingestion_res.flattened_image)
        cv2.imwrite(str(debug_dir / f"{img_stem}_02_binary_mask.png"), prep_res.binary_mask)
        cv2.imwrite(str(debug_dir / f"{img_stem}_03_gray_enhanced.png"), prep_res.gray_image)
        if seg_res.lines:
            from segmentation import render_segmentation_overlay
            render_segmentation_overlay(
                ingestion_res.flattened_image,
                seg_res,
                save_path=debug_dir / f"{img_stem}_04_segmentation.png"
            )
        render_side_by_side_comparison(
            ingestion_res.untouched_image,
            annotated_img,
            output_path=debug_dir / f"{img_stem}_05_comparison.png",
        )

    t_end = time.perf_counter()
    runtime_sec = t_end - t_start

    result = PipelineResult(
        input_path=src_path,
        runtime_seconds=runtime_sec,
        ingestion=ingestion_res,
        preprocessing=prep_res,
        segmentation=seg_res,
        normalization=norm_res,
        features=feature_matrix,
        predictions=predictions,
        reconstruction=reconstruction_res,
        annotated_image=annotated_img,
        output_text_path=text_out_path,
        output_annotated_path=annotated_out_path,
        warnings=warnings,
    )

    if not quiet:
        print_execution_summary(result)

    return result


def print_execution_summary(result: PipelineResult) -> None:
    """Prints the structured CLI execution summary to stdout."""
    h, w = result.ingestion.original_dimensions
    channels = 3 if result.ingestion.untouched_image.ndim == 3 else 1
    warp_status = "Applied (Homography 3x3)" if result.ingestion.warp_applied else "Bypassed (Uniform / Full Canvas)"
    mean_conf = result.reconstruction.mean_confidence

    print("======================================================================")
    print("CAMERA-BASED HANDWRITING RECOGNITION PIPELINE EXECUTION SUMMARY")
    print("======================================================================")
    print(f"Input Image              : {result.input_path}")
    print(f"Image Resolution         : {w}x{h} ({channels} channels)")
    print(f"Perspective Transform    : {warp_status}")
    print(f"Preprocessing Mode       : Morphological Bg Division + CLAHE + Sauvola")
    print(f"Detected Text Lines      : {result.reconstruction.total_lines}")
    print(f"Detected Words           : {result.reconstruction.total_words}")
    print(f"Detected Characters      : {result.reconstruction.total_characters}")
    print(f"Mean Classification Conf : {mean_conf:.3f} ({mean_conf * 100:.1f}%)")
    print(f"Output Text File         : {result.output_text_path}")
    print(f"Annotated Image File     : {result.output_annotated_path}")
    print(f"Total Pipeline Runtime   : {result.runtime_seconds:.2f}s")
    print(f"Warnings / Anomalies     : {len(result.warnings)}")
    print("======================================================================")
    print("Reconstructed Text Preview:")
    print("----------------------------------------------------------------------")
    if result.reconstruction.full_text.strip():
        print(result.reconstruction.full_text)
    else:
        print("[No text recognized]")
    print("======================================================================")


def build_arg_parser() -> argparse.ArgumentParser:
    """Constructs command line argument parser."""
    parser = argparse.ArgumentParser(
        description="Classical Computer Vision & Shallow MLP Camera-Based Handwriting Recognition System",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--input", "-i",
        type=str,
        required=True,
        help="Path to input photograph or image file (*.jpg, *.png)",
    )
    parser.add_argument(
        "--output-dir", "-o",
        type=str,
        default=None,
        help="Root directory for text, annotated images, and debug outputs",
    )
    parser.add_argument(
        "--warp",
        action="store_true",
        default=None,
        help="Force 4-point paper perspective flattening",
    )
    parser.add_argument(
        "--no-warp",
        action="store_true",
        help="Bypass perspective transformation",
    )
    parser.add_argument(
        "--debug", "-d",
        action="store_true",
        help="Save intermediate stage debugging masks and crops to outputs/debug/",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=3,
        help="Number of top candidate classes to rank per character",
    )
    parser.add_argument(
        "--quiet", "-q",
        action="store_true",
        help="Suppress console execution summary",
    )
    return parser


def main() -> int:
    """CLI Entry Point."""
    parser = build_arg_parser()
    args = parser.parse_args()

    apply_warp = None
    if args.no_warp:
        apply_warp = False
    elif args.warp:
        apply_warp = True

    try:
        run_pipeline(
            input_path=args.input,
            output_dir=args.output_dir,
            apply_warp=apply_warp,
            top_k_count=args.top_k,
            debug=args.debug,
            quiet=args.quiet,
        )
        return 0
    except Exception as e:
        print(f"\n[FATAL PIPELINE ERROR] {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
