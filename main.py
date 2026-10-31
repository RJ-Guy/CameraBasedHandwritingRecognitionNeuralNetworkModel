"""
main.py - Root CLI Entry Point for Camera-Based Handwriting Recognition System

Forwards command-line execution and API calls to Model/main.py.
"""
import importlib.util
from pathlib import Path
import sys

MODEL_DIR = Path(__file__).resolve().parent / "Model"
if str(MODEL_DIR) not in sys.path:
    sys.path.insert(0, str(MODEL_DIR))

# Load Model/main.py explicitly to avoid namespace shadowing
_spec = importlib.util.spec_from_file_location("model_main", MODEL_DIR / "main.py")
_model_main = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_model_main)

run_pipeline = _model_main.run_pipeline
PipelineResult = _model_main.PipelineResult
main = _model_main.main

if __name__ == "__main__":
    sys.exit(main())
