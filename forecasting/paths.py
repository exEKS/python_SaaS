from __future__ import annotations

import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def model_dir() -> Path:
    override = os.environ.get("WARWATCH_MODEL_DIR", "").strip()
    if override:
        return Path(override)
    fc = ROOT / "forecasting"
    if any(fc.glob("*.pkl")):
        return fc
    return ROOT / "models"
