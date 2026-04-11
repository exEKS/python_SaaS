from __future__ import annotations

import sys
from pathlib import Path

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

load_dotenv(ROOT / ".env")

from forecasting.paths import model_dir
from forecasting.prediction_service import predict_event_probabilities
from forecasting.model_runtime import list_model_pickles

app = FastAPI(
    title="WarWatch Prediction API",
    description="Air alarm, explosion, and artillery risk estimates per region and date.",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health")
def health():
    mdir = model_dir()
    ok = mdir.is_dir() and any(mdir.glob("*.pkl"))
    return {"status": "ok" if ok else "no_models", "model_dir": str(mdir.resolve())}


@app.get("/models")
def list_models():
    mdir = model_dir()
    if not mdir.is_dir():
        raise HTTPException(404, f"Model directory not found: {mdir}")
    try:
        files = list_model_pickles(mdir)
    except FileNotFoundError:
        return {"model_dir": str(mdir.resolve()), "files": []}
    return {"model_dir": str(mdir.resolve()), "files": [f.name for f in files]}


@app.get("/predict")
def predict(
    region: str = Query(..., description="Region, e.g. Kyiv, Kharkiv, or Київ"),
    date: str = Query(..., description="Date as YYYY-MM-DD"),
):
    try:
        return predict_event_probabilities(region, date)
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e)) from e
    except RuntimeError as e:
        raise HTTPException(status_code=422, detail=str(e)) from e
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e)) from e
