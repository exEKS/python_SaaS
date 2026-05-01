from __future__ import annotations

import os
import sys
from datetime import date as date_type
from pathlib import Path

import requests
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from forecasting.default_feature_row import UI_REGION_LABELS

AUTO = "(auto — евристика за іменем файлу)"


def _is_auto_choice(choice: str) -> bool:
    return not choice or choice.startswith("(auto")


def _fetch_model_files(api_url: str) -> list[str]:
    try:
        mr = requests.get(f"{api_url}/models", timeout=15)
        if mr.ok:
            return mr.json().get("files") or []
    except requests.RequestException:
        pass
    return []


st.set_page_config(page_title="WarWatch", layout="centered")
st.title("WarWatch")
st.caption("Показує лише ймовірність тривоги (`alarm_prob`).")

api_url = os.environ.get("WARWATCH_API_URL", "http://127.0.0.1:8000").rstrip("/")

model_files = _fetch_model_files(api_url)
if not model_files:
    st.warning(
        f"Не вдалося отримати список `.pkl` з `{api_url}/models`. "
        "Перевір, що API запущено і в `model_dir` є моделі."
    )

options = [AUTO] + model_files
alarm_choice = st.selectbox("Модель (alarm)", options, index=0, key="alarm")

col1, col2 = st.columns(2)
with col1:
    region = st.selectbox("Регіон", UI_REGION_LABELS, index=0)
with col2:
    d = st.date_input("Дата", value=date_type.today())

if st.button("Отримати прогноз", type="primary"):
    params: dict = {"region": region, "date": d.isoformat()}
    if not _is_auto_choice(alarm_choice):
        params["alarm_model"] = alarm_choice
    try:
        r = requests.get(f"{api_url}/predict", params=params, timeout=120)
        r.raise_for_status()
        data = r.json()
    except requests.RequestException as e:
        st.error(f"Помилка API ({api_url}): {e}")
        st.info("Запусти бекенд: `uvicorn frontend.api.main:app --reload` з кореня репозиторію.")
        st.stop()

    st.metric("alarm_prob", f"{data.get('alarm_prob', 0):.3f}")
