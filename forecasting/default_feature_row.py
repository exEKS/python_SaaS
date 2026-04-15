from __future__ import annotations

import hashlib
import math
import os
import re
from datetime import datetime

import pandas as pd

def _template_calendar_env_enabled() -> bool:
    v = os.environ.get("WARWATCH_NO_TEMPLATE_CALENDAR", "").strip().lower()
    return v not in ("1", "true", "yes", "on")


def _apply_template_calendar_shifts(row: dict[str, float], ts: pd.Timestamp, region: str) -> None:
    """Shift template numerics by date + region so models see different X per day (no month/doy in .pkl)."""
    phase = 2.0 * math.pi * (float(ts.dayofyear) - 1.0) / 365.25
    digest = hashlib.sha256(region.strip().encode("utf-8")).digest()
    jitter = (digest[0] / 255.0 + digest[1] / 765.0) * 0.55
    p = phase + jitter

    dt = 4.5 * math.sin(p)
    row["day_temp"] = float(max(-24.0, min(40.0, row.get("day_temp", 4.0) + dt)))
    row["day_tempmax"] = float(max(-22.0, min(44.0, row.get("day_tempmax", 6.0) + dt + 2.0)))
    row["day_tempmin"] = float(max(-32.0, min(36.0, row.get("day_tempmin", 1.0) + dt - 1.0)))

    rh = 14.0 * math.cos(p * 1.07)
    row["day_humidity"] = float(max(18.0, min(98.0, row.get("day_humidity", 70.0) + rh)))

    wnd = 6.5 * math.sin(p * 0.65 + 0.3)
    row["day_windspeed"] = float(max(0.0, min(58.0, row.get("day_windspeed", 10.0) + wnd)))

    roll = 0.18 * math.sin(p * 1.25 + 0.1)
    row["alarm_roll7_mean"] = float(max(0.0, min(1.0, row.get("alarm_roll7_mean", 0.08) + roll)))

    txt = 0.12 * math.sin(p * 0.88 - 0.4)
    row["text_intensity_index"] = float(max(0.0, min(1.0, row.get("text_intensity_index", 0.12) + txt)))

    rpc = int(round(row.get("reddit_post_count", 8.0) + 8.0 * math.sin(p * 0.55)))
    row["reddit_post_count"] = float(max(0.0, min(200.0, rpc)))

N_TEST = 10

DEMO_BATCH_DICT: dict[str, list] = {
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
    "region_Івано_Франківська_обл": [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    "region_Волинська_обл": [0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    "region_Вінницька_обл": [0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    "region_Дніпропетровська_обл": [0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    "region_Донецька_обл": [0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    "region_Житомирська_обл": [0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0],
    "region_Закарпатська_обл": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0],
    "region_Запорізька_обл": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0],
    "region_Київ": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0],
    "region_Київська_обл": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0],
    "region_Кіровоградська_обл": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    "region_Львівська_обл": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    "region_Миколаївська_обл": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    "region_Одеська_обл": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    "region_Полтавська_обл": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    "region_Рівненська_обл": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    "region_Сумська_обл": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    "region_Тернопільська_обл": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    "region_Харківська_обл": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    "region_Херсонська_обл": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    "region_Хмельницька_обл": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    "region_Черкаська_обл": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    "region_Чернівецька_обл": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    "region_Чернігівська_обл": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    "weather_Clear": [0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0],
    "weather_Overcast": [1.0, 0.0, 1.0, 1.0, 0.0, 1.0, 0.0, 1.0, 1.0, 0.0],
    "weather_Partially_cloudy": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0],
    "weather_Rain": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    "weather_Rain_Freezing_Drizzle/Freezing_Rain_Overcast": [0.0] * N_TEST,
    "weather_Rain_Overcast": [0.0] * N_TEST,
    "weather_Rain_Partially_cloudy": [0.0] * N_TEST,
    "weather_Snow": [0.0] * N_TEST,
    "weather_Snow_Freezing_Drizzle/Freezing_Rain_Ice_Overcast": [0.0] * N_TEST,
    "weather_Snow_Freezing_Drizzle/Freezing_Rain_Overcast": [0.0] * N_TEST,
    "weather_Snow_Freezing_Drizzle/Freezing_Rain_Partially_cloudy": [0.0] * N_TEST,
    "weather_Snow_Overcast": [0.0] * N_TEST,
    "weather_Snow_Partially_cloudy": [0.0] * N_TEST,
    "weather_Snow_Rain": [0.0] * N_TEST,
    "weather_Snow_Rain_Freezing_Drizzle/Freezing_Rain_Overcast": [0.0] * N_TEST,
    "weather_Snow_Rain_Freezing_Drizzle/Freezing_Rain_Partially_cloudy": [0.0] * N_TEST,
    "weather_Snow_Rain_Ice_Overcast": [0.0] * N_TEST,
    "weather_Snow_Rain_Overcast": [0.0] * N_TEST,
    "weather_Snow_Rain_Partially_cloudy": [0.0] * N_TEST,
}

REGION_COLUMNS = [c for c in DEMO_BATCH_DICT if c.startswith("region_")]

_ALIAS_TO_COLUMN: dict[str, str] = {}
for col in REGION_COLUMNS:
    short = col.replace("region_", "")
    _ALIAS_TO_COLUMN[re.sub(r"[^a-z0-9]+", "", short.lower())] = col
_aliases_extra = {
    "kyiv": "region_Київ",
    "київ": "region_Київ",
    "kiev": "region_Київ",
    "kharkiv": "region_Харківська_обл",
    "харків": "region_Харківська_обл",
    "lviv": "region_Львівська_обл",
    "львів": "region_Львівська_обл",
    "odesa": "region_Одеська_обл",
    "одеса": "region_Одеська_обл",
    "dnipro": "region_Дніпропетровська_обл",
    "дніпро": "region_Дніпропетровська_обл",
    "zaporizhzhia": "region_Запорізька_обл",
    "запоріжжя": "region_Запорізька_обл",
}
_ALIAS_TO_COLUMN.update(_aliases_extra)


def normalize_region_column(region: str) -> str:
    s = region.strip()
    if s in REGION_COLUMNS:
        return s
    key = re.sub(r"[^a-z0-9а-яіїєґ]+", "", s.lower())
    if key in _ALIAS_TO_COLUMN:
        return _ALIAS_TO_COLUMN[key]
    for col in REGION_COLUMNS:
        inner = col.replace("region_", "").lower()
        if inner == s.lower() or s.lower() in inner:
            return col
    return "region_Київ"


def _neutral_feature_row() -> dict[str, float]:
    row: dict[str, float] = {}
    for k in DEMO_BATCH_DICT:
        if k.startswith("region_"):
            row[k] = 0.0
        elif k.startswith("weather_"):
            row[k] = 0.0
        elif k in (
            "alarm_count",
            "alarm_total_duration_min",
            "alarm_all_region",
            "alarm_lag1",
            "alarm_lag2",
            "duration_lag1",
        ):
            row[k] = 0.0
        elif k == "alarm_roll7_mean":
            row[k] = 0.08
        elif k == "day_tempmax":
            row[k] = 6.0
        elif k == "day_tempmin":
            row[k] = 1.0
        elif k == "day_temp":
            row[k] = 4.0
        elif k == "day_humidity":
            row[k] = 70.0
        elif k == "day_windspeed":
            row[k] = 10.0
        elif k == "reddit_post_count":
            row[k] = 8.0
        elif k == "reddit_avg_score":
            row[k] = 25.0
        elif k == "text_intensity_index":
            row[k] = 0.12
        else:
            row[k] = 0.0
    row["weather_Overcast"] = 1.0
    return row


def feature_dataframe_one_row(
    region: str,
    date_iso: str,
    overrides: dict[str, float] | None = None,
    use_calendar_template: bool | None = None,
) -> pd.DataFrame:
    clean = date_iso.replace("Z", "+00:00")
    if "+" not in clean and clean.count("-") >= 3:
        clean = clean.split("T")[0]
    ts = pd.Timestamp(datetime.fromisoformat(clean[:10]))
    row = _neutral_feature_row()
    for c in REGION_COLUMNS:
        row[c] = 0.0
    row[normalize_region_column(region)] = 1.0
    doy = float(ts.dayofyear)
    row["month"] = float(ts.month)
    row["day_of_year"] = doy
    row["day_of_week"] = float(ts.dayofweek)
    ang = 2.0 * math.pi * (doy - 1.0) / 365.25
    row["sin_day_of_year"] = math.sin(ang)
    row["cos_day_of_year"] = math.cos(ang)

    if use_calendar_template is None:
        calendar_on = _template_calendar_env_enabled()
    elif use_calendar_template is True:
        calendar_on = True
    else:
        calendar_on = False

    profile = "neutral"
    if calendar_on:
        _apply_template_calendar_shifts(row, ts, region.strip())
        profile = "neutral+calendar_template"

    if overrides:
        for key, val in overrides.items():
            if key in row:
                row[key] = float(val)

    df = pd.DataFrame([row])
    df.attrs["feature_profile"] = profile
    return df


UI_REGION_LABELS = [
    "Київ",
    "Харківська обл.",
    "Львівська обл.",
    "Одеська обл.",
    "Дніпропетровська обл.",
    "Запорізька обл.",
    "Донецька обл.",
    "Вінницька обл.",
    "Чернігівська обл.",
]
