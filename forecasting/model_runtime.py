from __future__ import annotations

import pickle
from pathlib import Path

import numpy as np
import pandas as pd

try:
    import joblib
except ImportError:
    joblib = None


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
        hint = "pip install -r requirements.txt"
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


def align_to_estimator(estimator, raw: pd.DataFrame, silent: bool = True) -> pd.DataFrame:
    names = _feature_column_names(estimator)
    if not names:
        raise RuntimeError(
            "Model has no feature_names_ / feature_names_in_. "
            "Train with a pandas DataFrame so names are stored in the pickle."
        )
    out = raw.reindex(columns=names, fill_value=0.0)
    if not silent:
        missing = set(names) - set(raw.columns)
        if missing:
            print(f"  Note: {len(missing)} feature(s) missing → filled with 0.0")
        extra = set(raw.columns) - set(names)
        if extra:
            print(f"  Note: {len(extra)} extra column(s) ignored.")
    return out.astype(np.float64)


def predict_proba_positive_or_score(estimator, X: pd.DataFrame) -> tuple[float, str]:
    pred = estimator.predict(X)
    pred = np.asarray(pred).ravel().astype(float)
    kind = "classifier"
    if hasattr(estimator, "predict_proba"):
        try:
            probas = estimator.predict_proba(X)
            if probas.ndim == 2 and probas.shape[1] >= 2:
                return float(probas[0, 1]), "classifier"
        except (AttributeError, TypeError, ValueError):
            pass
    v = float(pred[0])
    if v < 0.0 or v > 1.0:
        kind = "regression_clipped"
        v = float(np.clip(v, 0.0, 1.0))
    else:
        kind = "regression_unit"
    return v, kind
