from flask import Flask, request, jsonify
from datetime import datetime, timezone
import json
import os
import random

app = Flask(__name__)


PREDICTIONS_FILE = "predictions.json"
MODEL_META_FILE = "model_meta.json"

REGIONS = [
    "Donetsk", "Luhansk", "Zaporizhzhia", "Kherson", "Kharkiv",
    "Mykolaiv", "Dnipropetrovsk", "Kyiv", "Odesa", "Zhytomyr",
    "Sumy", "Chernihiv", "Poltava", "Vinnytsia", "Lviv",
    "Ivano-Frankivsk", "Zakarpattia", "Chernivtsi", "Ternopil",
    "Khmelnytskyi", "Rivne", "Volyn", "Kirovohrad", "Cherkasy"
]

HOURS = [f"{h:02d}:00" for h in range(24)]



def _load_json(path: str):
    if os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    return None


def _generate_mock_predictions() -> dict:

    return {
        region: {hour: bool(random.random() > 0.6) for hour in HOURS}
        for region in REGIONS
    }


def _get_predictions() -> dict:
    data = _load_json(PREDICTIONS_FILE)
    if data and "regions_forecast" in data:
        return data["regions_forecast"]
    return _generate_mock_predictions()


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
        "model_version": meta.get("model_version", "v1"),
    }



@app.route("/forecast", methods=["POST"])
def forecast():

    body = request.get_json(silent=True) or {}
    region = body.get("region", "all")

    all_predictions = _get_predictions()
    meta = _get_meta()

    if region and region.lower() != "all":
        matched = next(
            (r for r in all_predictions if r.lower() == region.lower()), None
        )
        if matched is None:
            return jsonify({
                "error": f"Region '{region}' not found.",
                "available_regions": list(all_predictions.keys())
            }), 404
        regions_forecast = {matched: all_predictions[matched]}
    else:
        regions_forecast = all_predictions

    response = {
        "last_model_train_time": meta["last_model_train_time"],
        "last_prediction_time": meta["last_prediction_time"],
        "model_name": meta["model_name"],
        "model_version": meta["model_version"],
        "requested_region": region,
        "regions_forecast": regions_forecast,
    }

    return jsonify(response), 200


if __name__ == "__main__":
    print("=== Alarm Forecast API ===")
    print("Endpoints:")
    print("  POST http://localhost:5001/forecast   — main forecast endpoint")
    app.run(debug=True, host="0.0.0.0", port=5001)