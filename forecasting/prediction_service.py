from __future__ import annotations

import os
from pathlib import Path

from forecasting.default_feature_row import feature_dataframe_one_row
from forecasting.model_runtime import (
    align_to_estimator,
    load_pickled_estimator,
    predict_proba_positive_or_score,
    unwrap_estimator,
)
from forecasting.paths import model_dir


def _pick_model_path(mdir: Path, kind: str) -> Path | None:
    env_map = {
        "alarm": os.environ.get("WARWATCH_MODEL_ALARM", "").strip(),
        "explosion": os.environ.get("WARWATCH_MODEL_EXPLOSION", "").strip(),
        "artillery": os.environ.get("WARWATCH_MODEL_ARTILLERY", "").strip(),
    }
    if env_map.get(kind):
        p = Path(env_map[kind])
        if p.is_file():
            return p
    patterns = {
        "alarm": ("alarm", "тривог", "catboost", "randomforest", "forest", "xgb", "logistic", "linear", "ridge", "decision"),
        "explosion": ("explosion", "вибух"),
        "artillery": ("artillery", "артилер"),
    }
    files = sorted(mdir.glob("*.pkl"))
    for f in files:
        low = f.name.lower()
        if any(k in low for k in patterns.get(kind, ())):
            return f
    return None


def _default_primary(mdir: Path) -> Path | None:
    files = sorted(mdir.glob("*.pkl"))
    if not files:
        return None
    prefer = ("catboost", "randomforest", "forest", "xgb", "logistic", "linear", "ridge")
    for name in prefer:
        for f in files:
            if name in f.name.lower():
                return f
    return files[0]


def _run_model(path: Path, df):
    raw = load_pickled_estimator(path)
    est = unwrap_estimator(raw)
    X = align_to_estimator(est, df, silent=True)
    score, kind = predict_proba_positive_or_score(est, X)
    return float(score), kind, type(est).__name__


def predict_event_probabilities(region: str, date_iso: str) -> dict:
    mdir = model_dir()
    if not mdir.is_dir():
        raise FileNotFoundError(f"Model directory missing: {mdir}")

    df = feature_dataframe_one_row(region, date_iso)

    pa = _pick_model_path(mdir, "alarm") or _default_primary(mdir)
    if pa is None:
        raise FileNotFoundError(
            f"No .pkl models in {mdir}. Add models under models/ or set WARWATCH_MODEL_DIR."
        )

    pe = _pick_model_path(mdir, "explosion")
    part = _pick_model_path(mdir, "artillery")

    alarm_p, _, alarm_cls = _run_model(pa, df)
    models_used = {"alarm": f"{pa.name} ({alarm_cls})"}

    if pe is not None and pe.resolve() != pa.resolve():
        explosion_p, _, ec = _run_model(pe, df)
        models_used["explosion"] = f"{pe.name} ({ec})"
    else:
        explosion_p = alarm_p
        models_used["explosion"] = f"same_as_alarm (from {pa.name})"

    if part is None or part.resolve() == pa.resolve():
        artillery_p = alarm_p
        models_used["artillery"] = f"same_as_alarm (from {pa.name})"
    elif pe and part.resolve() == pe.resolve():
        artillery_p = explosion_p
        models_used["artillery"] = models_used["explosion"]
    else:
        artillery_p, _, ac = _run_model(part, df)
        models_used["artillery"] = f"{part.name} ({ac})"

    unique_paths = {pa.resolve()}
    if pe and pe.resolve() != pa.resolve():
        unique_paths.add(pe.resolve())
    if part:
        pr, par = part.resolve(), pa.resolve()
        per = pe.resolve() if pe else None
        if pr != par and (per is None or pr != per):
            unique_paths.add(pr)
    if len(unique_paths) == 1:
        mode = "single_model"
    elif len(unique_paths) == 2:
        mode = "two_models"
    else:
        mode = "three_models"

    return {
        "region": region.strip(),
        "date": date_iso,
        "alarm_prob": round(alarm_p, 6),
        "explosion_prob": round(explosion_p, 6),
        "artillery_prob": round(artillery_p, 6),
        "mode": mode,
        "models": models_used,
        "model_dir": str(mdir.resolve()),
    }
