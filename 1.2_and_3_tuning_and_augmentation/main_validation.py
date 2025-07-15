from src.validation import run_validation
from pathlib import Path

# 4. Validierung
THIS_DIR = Path(__file__).resolve().parent

MODEL_DIR = THIS_DIR / "models" / "pipeline_models"

REPO_ROOT = THIS_DIR.parent              
VAL_PATH  = REPO_ROOT / "data" / "val_data.pt"

run_validation(
        model_dir=MODEL_DIR,
        val_data_path=VAL_PATH,
    )