from __future__ import annotations

import sys
import os
import json
from pathlib import Path
from datetime import datetime, timezone

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Query, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

load_dotenv(ROOT / ".env")

from forecasting.feature_query_params import (
    feature_overrides_from_query_params,
    supported_feature_query_params,
)
from forecasting.paths import model_dir
from forecasting.prediction_service import predict_event_probabilities
from forecasting.model_runtime import list_model_pickles

app = FastAPI(
    title="WarWatch Prediction API",
    description=(
        "Air alarm risk estimate per region and date. "
        "Optional feature overrides: any query param listed at "
        "`GET /predict/supported-feature-params` (e.g. feat_day_humidity, "
        "feat_alarm_total_duration_min). Unknown query keys are ignored. "
        "Loaded models (CatBoost, RandomForest, logistic, decision tree, …) "
        "each use their own feature_names from the pickle; extras are dropped, "
        "missing columns are filled with 0.0."
    ),
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)




PREDICTIONS_FILE = "predictions.json"
MODEL_META_FILE = "model_randomforest_v2.json"
LOG_FILE = "api_request_logs.json"

class ForecastRequest(BaseModel):
    region: str = "all"

def _load_json(path: str):
    if os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    return None

def _save_log(data: dict):
    logs = []
    if os.path.exists(LOG_FILE):
        with open(LOG_FILE, "r", encoding="utf-8") as f:
            try:
                logs = json.load(f)
            except json.JSONDecodeError:
                logs = []

    logs.append(data)
    with open(LOG_FILE, "w", encoding="utf-8") as f:
        json.dump(logs, f, indent=4, ensure_ascii=False)

def _get_predictions() -> dict:
    data = _load_json(PREDICTIONS_FILE)
    if data and "regions_forecast" in data:
        return data["regions_forecast"]
    return {}

def _get_meta() -> dict:
    meta = _load_json(MODEL_META_FILE) or {}
    return {
        "last_model_train_time": meta.get(
            "last_model_train_time", "2025-01-01T00:00:00Z"
        ),
        "last_prediction_time": meta.get(
            "last_prediction_time",
            datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        ),
        "model_name": meta.get("model_name", "RandomForest"),
        "model_version": meta.get("model_version", "v2"),
    }


@app.get("/health")
@app.get("/api/health")
def health():
    mdir = model_dir()
    ok = mdir.is_dir() and any(mdir.glob("*.pkl"))
    return {"status": "ok" if ok else "no_models", "model_dir": str(mdir.resolve())}


@app.get("/models")
@app.get("/api/models")
def list_models():
    mdir = model_dir()
    if not mdir.is_dir():
        raise HTTPException(404, f"Model directory not found: {mdir}")
    try:
        files = list_model_pickles(mdir)
    except FileNotFoundError:
        return {"model_dir": str(mdir.resolve()), "files": []}
    return {"model_dir": str(mdir.resolve()), "files": [f.name for f in files]}


@app.get("/predict/supported-feature-params")
@app.get("/api/predict/supported-feature-params")
def predict_supported_feature_params():
    return supported_feature_query_params()


@app.get("/predict")
@app.get("/api/predict")
def predict(
    request: Request,
    region: str = Query(..., description="Region, e.g. Kyiv, Kharkiv, or Київ"),
    date: str = Query(..., description="Date as YYYY-MM-DD"),
    alarm_model: str | None = Query(
        None,
        description="Optional: alarm head — file name in model_dir or absolute path to .pkl",
    ),
):
    try:
        fo = feature_overrides_from_query_params(request.query_params)
        return predict_event_probabilities(
            region,
            date,
            alarm_model=alarm_model,
            feature_overrides=fo,
        )
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e)) from e
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e)) from e
    except RuntimeError as e:
        raise HTTPException(status_code=422, detail=str(e)) from e
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e)) from e


@app.post("/forecast")
@app.post("/api/forecast")
def forecast(body: ForecastRequest):
    region = body.region

    all_predictions = _get_predictions()
    meta = _get_meta()

    if region and region.lower() != "all":
        matched = next(
            (r for r in all_predictions if r.lower() == region.lower()), None
        )
        if matched is None:
            raise HTTPException(
                status_code=404,
                detail={
                    "error": f"Region '{region}' not found.",
                    "available_regions": list(all_predictions.keys())
                }
            )
        regions_forecast = {matched: all_predictions[matched]}
    else:
        regions_forecast = all_predictions

    response = {
        "timestamp": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "last_model_train_time": meta["last_model_train_time"],
        "last_prediction_time": meta["last_prediction_time"],
        "model_name": meta["model_name"],
        "model_version": meta["model_version"],
        "requested_region": region,
        "regions_forecast": regions_forecast,
    }

    _save_log(response)

    return response


# Serve built React app from the same FastAPI server when available.
FRONTEND_DIST = ROOT / "frontend" / "react-app" / "dist"
if FRONTEND_DIST.is_dir():
    app.mount("/", StaticFiles(directory=str(FRONTEND_DIST), html=True), name="frontend")