from __future__ import annotations

import os
from pathlib import Path

from forecasting.default_feature_row import feature_dataframe_one_row
from forecasting.model_runtime import (
    align_to_estimator,
    binary_proba_vector,
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
    # Match by pattern priority (not by sorted filename): otherwise e.g.
    # 1__decision_tree__v1.pkl wins alphabetically over randomforest_model.pkl
    # because both match generic tokens, and a single tree saturates proba to 0/1.
    patterns = {
        "alarm": (
            "alarm",
            "тривог",
            "catboost",
            "randomforest",
            "forest",
            "xgb",
            "logistic",
            "linear",
            "ridge",
            "decision",
        ),
        "explosion": ("explosion", "вибух"),
        "artillery": ("artillery", "артилер"),
    }
    files = sorted(mdir.glob("*.pkl"))
    for token in patterns.get(kind, ()):
        for f in files:
            if token in f.name.lower():
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
    X = align_to_estimator(est, df, silent=True, model_path=path)
    score, kind = predict_proba_positive_or_score(est, X)
    pair = binary_proba_vector(est, X)
    return float(score), kind, type(est).__name__, pair


def _resolve_model_override(mdir: Path, model_arg: str | None, label: str) -> Path | None:
    if not model_arg or not str(model_arg).strip():
        return None
    raw = str(model_arg).strip().strip('"')
    candidate = Path(raw)
    if candidate.is_file():
        return candidate.resolve()
    in_dir = mdir / raw
    if not raw.endswith(".pkl"):
        in_dir = mdir / f"{raw}.pkl"
    if in_dir.is_file():
        return in_dir.resolve()
    raise FileNotFoundError(
        f"{label} not found: {raw!r} (tried absolute path and {mdir})"
    )


def predict_event_probabilities(
    region: str,
    date_iso: str,
    alarm_model: str | None = None,
    explosion_model: str | None = None,
    artillery_model: str | None = None,
    feature_overrides: dict[str, float] | None = None,
) -> dict:
    mdir = model_dir()
    if not mdir.is_dir():
        raise FileNotFoundError(f"Model directory missing: {mdir}")

    df = feature_dataframe_one_row(region, date_iso, overrides=feature_overrides)
    feature_profile = str(df.attrs.get("feature_profile", "neutral"))

    pa = _resolve_model_override(mdir, alarm_model, "alarm_model")
    if pa is None:
        pa = _pick_model_path(mdir, "alarm") or _default_primary(mdir)
    if pa is None:
        raise FileNotFoundError(
            f"No .pkl models in {mdir}. Add models under models/ or set WARWATCH_MODEL_DIR."
        )

    pe = _resolve_model_override(mdir, explosion_model, "explosion_model")
    if pe is None:
        pe = _pick_model_path(mdir, "explosion")

    part = _resolve_model_override(mdir, artillery_model, "artillery_model")
    if part is None:
        part = _pick_model_path(mdir, "artillery")

    alarm_p, _, alarm_cls, alarm_pair = _run_model(pa, df)
    models_used = {"alarm": f"{pa.name} ({alarm_cls})"}
    proba_detail: dict = {}

    def _split(p):
        return {"P_negative": p[0], "P_positive": p[1]}

    if alarm_pair is not None:
        proba_detail["alarm"] = _split(alarm_pair)

    explosion_pair = None
    if pe is not None and pe.resolve() != pa.resolve():
        explosion_p, _, ec, explosion_pair = _run_model(pe, df)
        models_used["explosion"] = f"{pe.name} ({ec})"
        if explosion_pair is not None:
            proba_detail["explosion"] = _split(explosion_pair)
    else:
        explosion_p = alarm_p
        models_used["explosion"] = f"same_as_alarm (from {pa.name})"
        if alarm_pair is not None:
            proba_detail["explosion"] = {**_split(alarm_pair), "shared_with": "alarm"}

    if part is None or part.resolve() == pa.resolve():
        artillery_p = alarm_p
        models_used["artillery"] = f"same_as_alarm (from {pa.name})"
        if alarm_pair is not None:
            proba_detail["artillery"] = {**_split(alarm_pair), "shared_with": "alarm"}
    elif pe and part.resolve() == pe.resolve():
        artillery_p = explosion_p
        models_used["artillery"] = models_used["explosion"]
        if explosion_pair is not None:
            proba_detail["artillery"] = {**_split(explosion_pair), "shared_with": "explosion"}
        elif alarm_pair is not None:
            proba_detail["artillery"] = {**_split(alarm_pair), "shared_with": "alarm"}
    else:
        artillery_p, _, ac, ar_pair = _run_model(part, df)
        models_used["artillery"] = f"{part.name} ({ac})"
        if ar_pair is not None:
            proba_detail["artillery"] = _split(ar_pair)

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

    out = {
        "region": region.strip(),
        "date": date_iso,
        "alarm_prob": round(alarm_p, 6),
        "explosion_prob": round(explosion_p, 6),
        "artillery_prob": round(artillery_p, 6),
        "mode": mode,
        "models": models_used,
        "model_dir": str(mdir.resolve()),
        "feature_profile": feature_profile,
    }
    if feature_overrides:
        out["feature_overrides"] = {k: round(float(v), 6) for k, v in feature_overrides.items()}
    if proba_detail:
        out["binary_classifier_split"] = proba_detail
    return out
