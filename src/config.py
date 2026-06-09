"""
Project paths — all configurable via environment variables.

Override defaults by exporting any of these env vars, e.g.:

    export V21_DATA_DIR=/path/to/your/data
    export V21_CKPT_DIR=/path/to/checkpoints
    export V21_DB_PATH=/path/to/epiirn_v0.0.7.db
    export V21_WARMSTART_CKPT=/path/to/V20/best_model.pt

If unset, defaults assume a layout under the repo's ./data, ./checkpoints, and ./db.
"""
import os
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent

DATA_DIR = os.environ.get("V21_DATA_DIR", str(REPO_ROOT / "data"))
CKPT_DIR = os.environ.get("V21_CKPT_DIR", str(REPO_ROOT / "checkpoints"))
PRED_DIR = os.environ.get("V21_PRED_DIR", str(REPO_ROOT / "predictions"))
SURV_DIR = os.environ.get("V21_SURV_DIR", str(REPO_ROOT / "survival"))
RESULTS_DIR = os.environ.get("V21_RESULTS_DIR", str(REPO_ROOT / "results"))
DB_PATH = os.environ.get("V21_DB_PATH", str(REPO_ROOT / "db" / "epiirn_v0.0.7.db"))
WARMSTART_CKPT = os.environ.get("V21_WARMSTART_CKPT", "")  # optional
HF_BACKBONE = os.environ.get("V21_HF_BACKBONE", "facebook/dinov3-vitb16-pretrain-lvd1689m")
