from __future__ import annotations

import os
import re
from pathlib import Path
from threading import Lock
from datetime import datetime

from forecasting.default_feature_row import (
    REGION_COLUMNS,
    feature_dataframe_one_row,
    normalize_region_column,
)
from forecasting.local_live_features import live_feature_overrides_for_prediction
from forecasting.model_runtime import (
    align_to_estimator,
    binary_proba_vector,
    load_pickled_estimator,
    predict_proba_positive_or_score,
    unwrap_estimator,
)
from forecasting.paths import model_dir

_MODEL_CACHE: dict[Path, object] = {}
_MODEL_CACHE_LOCK = Lock()


def _pick_model_path(mdir: Path, kind: str) -> Path | None:
    env_map = {
        "alarm": os.environ.get("WARWATCH_MODEL_ALARM", "").strip(),
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
    raw = _load_model_cached(path)
    est = unwrap_estimator(raw)
    X = align_to_estimator(est, df, silent=True, model_path=path)
    X = _inject_model_specific_signals(X, df)
    X = _ensure_region_signal(X, est, path, str(df.attrs.get("requested_region", "")))
    score, kind = predict_proba_positive_or_score(est, X)
    pair = binary_proba_vector(est, X)
    return float(score), kind, type(est).__name__, pair


def _load_model_cached(path: Path):
    resolved = path.resolve()
    with _MODEL_CACHE_LOCK:
        cached = _MODEL_CACHE.get(resolved)
        if cached is not None:
            return cached
    loaded = load_pickled_estimator(resolved)
    with _MODEL_CACHE_LOCK:
        _MODEL_CACHE[resolved] = loaded
    return loaded


def _model_feature_names(estimator, path: Path) -> list[str]:
    names = getattr(estimator, "feature_names_", None)
    if names is not None and len(names) > 0:
        return [str(x) for x in names]
    names = getattr(estimator, "feature_names_in_", None)
    if names is not None and len(names) > 0:
        return [str(x) for x in names]
    sidecar = path.parent / f"{path.name}.features.json"
    if sidecar.is_file():
        import json

        raw = json.loads(sidecar.read_text(encoding="utf-8"))
        if isinstance(raw, list) and raw:
            return [str(x) for x in raw]
    return []


def _slug(s: str) -> str:
    return re.sub(r"[^a-z0-9а-яіїєґ]+", "", s.lower())


def _stable_city_code(region: str) -> float:
    """
    Deterministic fallback for models that require `city_encoded`.
    Keeps inference non-constant across regions when the model has no one-hot columns.
    """
    key = _slug(region)
    table = {
        "kyiv": 12.0,
        "kharkiv": 10.0,
        "lviv": 14.0,
        "odesa": 17.0,
        "dnipropetrovsk": 6.0,
        "zaporizhzhia": 23.0,
        "donetsk": 5.0,
        "vinnytsia": 2.0,
        "chernihiv": 4.0,
        "sumy": 20.0,
        "kherson": 11.0,
        "mykolaiv": 16.0,
        "poltava": 18.0,
        "rivne": 19.0,
        "ternopil": 21.0,
        "zhytomyr": 24.0,
        "chernivtsi": 3.0,
        "cherkasy": 1.0,
        "khmelnytskyi": 13.0,
        "ivanofrankivsk": 8.0,
        "luhansk": 15.0,
        "kirovohrad": 9.0,
        "volyn": 22.0,
        "zakarpattia": 7.0,
    }
    if key in table:
        return table[key]
    return float(abs(hash(key)) % 25)


def _inject_model_specific_signals(X, raw_df):
    cols = set(X.columns)
    row = raw_df.iloc[0]
    region = str(raw_df.attrs.get("requested_region", "")).strip()

    # Models trained with label-encoded city IDs.
    if "city_encoded" in cols and float(X.loc[X.index[0], "city_encoded"]) == 0.0:
        X.loc[X.index[0], "city_encoded"] = _stable_city_code(region or "kyiv")

    # Derive hour-level weather proxies from daily template values.
    day_temp = float(row.get("day_temp", 4.0))
    day_humidity = float(row.get("day_humidity", 70.0))
    day_windspeed = float(row.get("day_windspeed", 10.0))
    hour_raw = row.get("hour", None)
    if hour_raw is None:
        hour_val = float(datetime.now().hour)
    else:
        hour_val = float(hour_raw)
    hour_int = int(max(0, min(23, round(hour_val))))

    if "hour" in cols:
        X.loc[X.index[0], "hour"] = float(hour_int)
    if "is_night" in cols:
        X.loc[X.index[0], "is_night"] = 1.0 if hour_int < 6 or hour_int >= 22 else 0.0
    if "hour_temp" in cols:
        # Mild diurnal cycle around day_temp
        shift = -2.0 if hour_int < 6 else (2.0 if 13 <= hour_int <= 17 else 0.0)
        X.loc[X.index[0], "hour_temp"] = day_temp + shift
    if "hour_humidity" in cols:
        X.loc[X.index[0], "hour_humidity"] = day_humidity
    if "hour_windspeed" in cols:
        X.loc[X.index[0], "hour_windspeed"] = day_windspeed
    if "hour_cloudcover" in cols:
        X.loc[X.index[0], "hour_cloudcover"] = 55.0
    if "hour_precip" in cols:
        X.loc[X.index[0], "hour_precip"] = 0.0

    return X


def _ensure_region_signal(X, estimator, model_path: Path, requested_region: str):
    """
    Guarantee that region information is present after column alignment.
    Some pickles have region feature names with encoding artifacts; in that case
    the direct one-hot name from default_feature_row may not match exactly.
    """
    feature_names = _model_feature_names(estimator, model_path)
    if not feature_names:
        return X
    model_region_cols = [c for c in feature_names if c.startswith("region_")]
    if not model_region_cols:
        return X
    model_region_cols = [c for c in model_region_cols if c in X.columns]
    if not model_region_cols:
        return X

    # If region signal already survived alignment, do nothing.
    if float(X.loc[X.index[0], model_region_cols].sum()) > 0.0:
        return X

    canonical_region_col = normalize_region_column(requested_region or "Kyiv")
    canonical_slug = _slug(canonical_region_col.replace("region_", ""))
    target_col = None

    # 1) Direct/fuzzy name match.
    for col in model_region_cols:
        if _slug(col.replace("region_", "")) == canonical_slug:
            target_col = col
            break
    if target_col is None:
        for col in model_region_cols:
            cslug = _slug(col.replace("region_", ""))
            if canonical_slug and (canonical_slug in cslug or cslug in canonical_slug):
                target_col = col
                break

    # 2) Positional fallback when counts match canonical region schema.
    if target_col is None and len(model_region_cols) == len(REGION_COLUMNS):
        try:
            idx = REGION_COLUMNS.index(canonical_region_col)
            target_col = model_region_cols[idx]
        except ValueError:
            target_col = None

    if target_col is None:
        return X

    X.loc[:, model_region_cols] = 0.0
    X.loc[X.index[0], target_col] = 1.0
    return X


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
    feature_overrides: dict[str, float] | None = None,
) -> dict:
    mdir = model_dir()
    if not mdir.is_dir():
        raise FileNotFoundError(f"Model directory missing: {mdir}")

    live_overrides, live_meta = live_feature_overrides_for_prediction(region, date_iso)
    merged_overrides = {**live_overrides, **(feature_overrides or {})}
    df = feature_dataframe_one_row(
        region, date_iso, overrides=merged_overrides or None
    )
    df.attrs["requested_region"] = region
    feature_profile = str(df.attrs.get("feature_profile", "neutral"))

    pa = resolve_alarm_model_path(mdir, alarm_model)
    if pa is None:
        raise FileNotFoundError(
            f"No .pkl models in {mdir}. Add models under models/ or set WARWATCH_MODEL_DIR."
        )

    alarm_p, _, alarm_cls, alarm_pair = _run_model(pa, df)
    models_used = {"alarm": f"{pa.name} ({alarm_cls})"}
    proba_detail: dict = {}

    def _split(p):
        return {"P_negative": p[0], "P_positive": p[1]}

    if alarm_pair is not None:
        proba_detail["alarm"] = _split(alarm_pair)

    out = {
        "region": region.strip(),
        "date": date_iso,
        "alarm_prob": round(alarm_p, 6),
        "mode": "single_model",
        "models": models_used,
        "model_dir": str(mdir.resolve()),
        "feature_profile": feature_profile,
    }
    if merged_overrides:
        out["feature_overrides"] = {
            k: round(float(v), 6) for k, v in merged_overrides.items()
        }
    if live_meta:
        out["live_context"] = live_meta
    if proba_detail:
        out["binary_classifier_split"] = proba_detail
    return out


def resolve_alarm_model_path(
    mdir: Path | None = None,
    alarm_model: str | None = None,
) -> Path | None:
    effective_dir = mdir or model_dir()
    pa = _resolve_model_override(effective_dir, alarm_model, "alarm_model")
    if pa is not None:
        return pa
    return _pick_model_path(effective_dir, "alarm") or _default_primary(effective_dir)


def warmup_models(alarm_model: str | None = None) -> dict:
    mdir = model_dir()
    if not mdir.is_dir():
        raise FileNotFoundError(f"Model directory missing: {mdir}")
    pa = resolve_alarm_model_path(mdir, alarm_model)
    if pa is None:
        raise FileNotFoundError(
            f"No .pkl models in {mdir}. Add models under models/ or set WARWATCH_MODEL_DIR."
        )
    raw = _load_model_cached(pa)
    est = unwrap_estimator(raw)
    return {
        "alarm_model": pa.name,
        "alarm_estimator_class": type(est).__name__,
        "model_dir": str(mdir.resolve()),
    }


def predict_hourly_alarm_profile(
    region: str,
    date_iso: str,
    alarm_model: str | None = None,
    feature_overrides: dict[str, float] | None = None,
) -> dict:
    mdir = model_dir()
    if not mdir.is_dir():
        raise FileNotFoundError(f"Model directory missing: {mdir}")
    pa = resolve_alarm_model_path(mdir, alarm_model)
    if pa is None:
        raise FileNotFoundError(
            f"No .pkl models in {mdir}. Add models under models/ or set WARWATCH_MODEL_DIR."
        )

    raw = _load_model_cached(pa)
    est = unwrap_estimator(raw)
    model_features = set(_model_feature_names(est, pa))
    hour_feature = next(
        (name for name in ("hour", "hour_of_day", "feat_hour") if name in model_features),
        None,
    )

    hours: list[dict] = []
    for h in range(24):
        effective_overrides = dict(feature_overrides or {})
        if hour_feature is not None:
            effective_overrides[hour_feature] = float(h)
        pred = predict_event_probabilities(
            region=region,
            date_iso=date_iso,
            alarm_model=str(pa),
            feature_overrides=effective_overrides or None,
        )
        hours.append({"hour": h, "alarm_prob": pred["alarm_prob"]})

    return {
        "region": region.strip(),
        "date": date_iso,
        "hourly_alarm_probabilities": hours,
        "model_name": pa.name,
        "uses_hour_feature": hour_feature is not None,
        "hour_feature_name": hour_feature,
    }
