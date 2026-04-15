#!/usr/bin/env python3
"""
Standalone inference script (single-file, no local imports).

Environment / setup:
  - Python: 3.10+ (tested with 3.10/3.11/3.12)
  - Optional env vars:
      WARWATCH_MODEL_DIR   -> folder with .pkl files (default: script folder)
      WARWATCH_STRICT_EXIT -> "1"/"true"/"yes" to return non-zero if any model fails

Dependencies:
  pip install -r requirements_inference.txt
  # If you do not use the file, install manually:
  # pip install pandas==2.2.1 numpy==1.26.4 scikit-learn==1.4.1 joblib==1.3.2 xgboost==2.0.3 catboost==1.2.7

Run:
  1) Put this script and requirements_inference.txt into the same folder as model .pkl files
     (or set WARWATCH_MODEL_DIR to that folder).
  2) Install deps:
       pip install -r requirements_inference.txt
  3) Execute:
       python inference_catboost.py
"""

from __future__ import annotations

import json
import os
import pickle
import sys
import traceback
from pathlib import Path

import numpy as np
import pandas as pd

try:
    import joblib
except ImportError:
    joblib = None

N_TEST = 10
TEST_DATA_DICT: dict[str, list[float]] = {
    "alarm_count": [0.0, 1.0, 0.0, 2.0, 0.0, 1.0, 0.0, 3.0, 0.0, 1.0],
    "alarm_total_duration_min": [0.0, 30.0, 0.0, 90.0, 0.0, 45.0, 0.0, 120.0, 0.0, 20.0],
    "alarm_all_region": [0.0, 1.0, 0.0, 1.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0],
    "day_tempmax": [8.2, 12.1, 3.5, 15.0, 7.3, 9.8, 4.2, 11.0, 6.1, 10.5],
    "day_tempmin": [2.1, 5.0, -1.0, 8.0, 1.5, 3.0, -2.0, 4.5, 0.0, 5.5],
    "day_temp": [5.0, 8.5, 1.2, 11.5, 4.4, 6.2, 0.8, 7.8, 3.0, 8.0],
    "day_humidity": [75.0, 68.0, 82.0, 60.0, 77.0, 71.0, 85.0, 65.0, 79.0, 70.0],
    "day_windspeed": [15.0, 10.0, 22.0, 8.0, 12.0, 18.0, 25.0, 9.0, 14.0, 11.0],
    "reddit_post_count": [10.0, 24.0, 5.0, 40.0, 8.0, 15.0, 3.0, 55.0, 12.0, 20.0],
    "reddit_avg_score": [35.0, 120.0, 12.0, 200.0, 28.0, 90.0, 5.0, 150.0, 44.0, 75.0],
    "text_intensity_index": [0.12, 0.45, 0.08, 0.62, 0.15, 0.33, 0.05, 0.71, 0.22, 0.38],
    "alarm_lag1": [0.0, 1.0, 1.0, 0.0, 0.0, 2.0, 0.0, 1.0, 0.0, 1.0],
    "alarm_lag2": [1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 2.0, 1.0, 0.0],
    "alarm_roll7_mean": [0.2, 0.5, 0.1, 0.8, 0.0, 0.4, 0.15, 0.9, 0.05, 0.35],
    "duration_lag1": [0.0, 40.0, 15.0, 0.0, 0.0, 60.0, 0.0, 35.0, 0.0, 25.0],
    "region_Київ": [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    "region_Харківська_обл": [0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    "region_Львівська_обл": [0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    "region_Одеська_обл": [0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    "region_Дніпропетровська_обл": [0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    "region_Запорізька_обл": [0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0],
    "region_Донецька_обл": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0],
    "region_Вінницька_обл": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0],
    "region_Чернігівська_обл": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0],
    "region_Волинська_обл": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0],
    "weather_Clear": [0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0],
    "weather_Overcast": [1.0, 0.0, 1.0, 1.0, 0.0, 1.0, 0.0, 1.0, 1.0, 0.0],
    "weather_Partially_cloudy": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0],
    "weather_Rain": [0.0] * N_TEST,
}


def _validate_test_dict(d: dict[str, list[float]]) -> None:
    for k, v in d.items():
        if len(v) != N_TEST:
            raise ValueError(f"Column {k!r} must have length {N_TEST}, got {len(v)}")


def model_dir() -> Path:
    override = os.environ.get("WARWATCH_MODEL_DIR", "").strip()
    if override:
        return Path(override)
    return Path(__file__).resolve().parent


def list_model_pickles(mdir: Path) -> list[Path]:
    if not mdir.is_dir():
        raise FileNotFoundError(
            f"Model directory does not exist: {mdir}\n"
            "Create it, add .pkl files, or set WARWATCH_MODEL_DIR."
        )
    files = sorted(mdir.glob("*.pkl"))
    if not files:
        raise FileNotFoundError(f"No .pkl files found in {mdir.resolve()}")
    return files


def unwrap_estimator(obj, depth: int = 0):
    if depth > 6:
        return obj
    be = getattr(obj, "best_estimator_", None)
    if be is not None:
        return unwrap_estimator(be, depth + 1)
    return obj


def load_pickled_estimator(path: Path):
    try:
        with open(path, "rb") as f:
            return pickle.load(f)
    except ModuleNotFoundError as e:
        hint = "install dependencies listed in this script header"
        if "catboost" in str(e).lower():
            hint = "pip install catboost"
        elif "xgboost" in str(e).lower():
            hint = "pip install xgboost"
        raise ModuleNotFoundError(f"{e}\n  Hint: {hint}") from e
    except Exception as e:
        pickle_err = e
        if joblib is not None:
            try:
                return joblib.load(path)
            except Exception as e2:
                raise RuntimeError(
                    f"Could not load {path.name}: pickle failed ({pickle_err!r}); "
                    f"joblib failed ({e2!r})."
                ) from e2
        raise RuntimeError(
            f"Could not load {path.name} with pickle ({pickle_err!r}). Install joblib."
        ) from pickle_err


def _feature_column_names(estimator) -> list[str] | None:
    fn = getattr(estimator, "feature_names_", None)
    if fn is not None and len(fn) > 0:
        return list(fn)
    fn = getattr(estimator, "feature_names_in_", None)
    if fn is not None and len(fn) > 0:
        return list(fn)
    return None


def align_to_estimator(estimator, raw: pd.DataFrame, model_path: Path) -> pd.DataFrame:
    names = _feature_column_names(estimator)
    if not names:
        sidecar = model_path.parent / f"{model_path.name}.features.json"
        if sidecar.is_file():
            raw_list = json.loads(sidecar.read_text(encoding="utf-8"))
            if isinstance(raw_list, list) and len(raw_list) > 0:
                names = [str(x) for x in raw_list]
    if not names:
        raise RuntimeError(
            "Model has no feature_names_ / feature_names_in_. "
            "Train with pandas DataFrame or add {model}.pkl.features.json."
        )
    out = raw.reindex(columns=names, fill_value=0.0)
    missing = set(names) - set(raw.columns)
    if missing:
        print(f"  Note: {len(missing)} feature(s) missing -> filled with 0.0")
    extra = set(raw.columns) - set(names)
    if extra:
        print(f"  Note: {len(extra)} extra column(s) ignored.")
    return out.astype(np.float64)


def run_one_pkl(path: Path, raw_df: pd.DataFrame) -> bool:
    print(f"\n{'=' * 60}\n{path.name}\n{'=' * 60}")
    loaded = load_pickled_estimator(path)
    est = unwrap_estimator(loaded)
    print(f"  Estimator type: {type(est).__name__}")
    X = align_to_estimator(est, raw_df, model_path=path)
    y_hat = np.asarray(est.predict(X)).ravel()
    out = pd.DataFrame({"row": np.arange(N_TEST), "pred": y_hat})
    if hasattr(est, "predict_proba"):
        try:
            probas = est.predict_proba(X)
            if probas.ndim == 2 and probas.shape[1] >= 2:
                out["proba_class1"] = probas[:, 1]
        except (AttributeError, TypeError, ValueError):
            pass
    print(out.to_string(index=False))
    return True


def main() -> int:
    _validate_test_dict(TEST_DATA_DICT)
    raw_df = pd.DataFrame(TEST_DATA_DICT)

    mdir = model_dir()
    print(f"Model directory: {mdir.resolve()}")
    pkls = list_model_pickles(mdir)

    ok, failed = 0, 0
    for p in pkls:
        try:
            if run_one_pkl(p, raw_df):
                ok += 1
        except Exception as e:
            failed += 1
            print(f"  ERROR: {e}", file=sys.stderr)
            traceback.print_exc(limit=2)

    print(f"\nDone. OK: {ok}, failed: {failed}, total .pkl: {len(pkls)}")
    strict = os.environ.get("WARWATCH_STRICT_EXIT", "").lower() in ("1", "true", "yes")
    if strict and failed:
        return 1
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except FileNotFoundError as e:
        print(e, file=sys.stderr)
        raise SystemExit(1)
