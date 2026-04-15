from __future__ import annotations

import os
import sys
from pathlib import Path

import pandas as pd
import requests
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from forecasting.default_feature_row import UI_REGION_LABELS, feature_dataframe_one_row
from forecasting.feature_query_params import QUERY_PARAM_TO_COLUMN

AUTO = "(auto — евристика за іменем файлу)"


def _is_auto_choice(choice: str) -> bool:
    return not choice or choice.startswith("(auto")


def _neutral_defaults_row(region: str, date_iso: str) -> dict[str, float]:
    df = feature_dataframe_one_row(region, date_iso, overrides=None)
    row = df.iloc[0]
    return {col: float(row[col]) for col in QUERY_PARAM_TO_COLUMN.values()}


def _slider_range(col: str) -> tuple[float, float, float]:
    """min_value, max_value, step."""
    return {
        "alarm_roll7_mean": (0.0, 1.0, 0.01),
        "day_temp": (-20.0, 40.0, 0.5),
        "alarm_lag1": (0.0, 5.0, 0.05),
        "alarm_lag2": (0.0, 5.0, 0.05),
        "text_intensity_index": (0.0, 1.0, 0.01),
        "alarm_count": (0.0, 15.0, 1.0),
        "alarm_total_duration_min": (0.0, 240.0, 1.0),
        "alarm_all_region": (0.0, 1.0, 0.05),
        "day_tempmax": (-25.0, 45.0, 0.5),
        "day_tempmin": (-30.0, 40.0, 0.5),
        "day_humidity": (0.0, 100.0, 1.0),
        "day_windspeed": (0.0, 60.0, 0.5),
        "reddit_post_count": (0.0, 200.0, 1.0),
        "reddit_avg_score": (0.0, 400.0, 1.0),
        "duration_lag1": (0.0, 200.0, 1.0),
    }[col]


def _sl(label: str, col: str, default_row: dict[str, float], key: str) -> float:
    lo, hi, stp = _slider_range(col)
    return st.slider(
        label,
        min_value=lo,
        max_value=hi,
        value=default_row[col],
        step=stp,
        key=key,
    )


def _fetch_model_files(api_url: str) -> list[str]:
    try:
        mr = requests.get(f"{api_url}/models", timeout=15)
        if mr.ok:
            return mr.json().get("files") or []
    except requests.RequestException:
        pass
    return []


st.set_page_config(page_title="WarWatch", layout="wide")
st.title("WarWatch — event risk dashboard")
st.caption("Ймовірності тривоги, вибухів та артилерії на наступні 24 год (залежить від моделей).")

api_url = os.environ.get("WARWATCH_API_URL", "http://127.0.0.1:8000").rstrip("/")

model_files = _fetch_model_files(api_url)
if not model_files:
    st.warning(
        f"Не вдалося отримати список `.pkl` з `{api_url}/models`. "
        "Перевір, що API запущено і в `model_dir` є моделі."
    )

options = [AUTO] + model_files

col1, col2 = st.columns(2)
with col1:
    region = st.selectbox("Регіон", UI_REGION_LABELS, index=0)
with col2:
    from datetime import date as date_type

    d = st.date_input("Дата", value=date_type.today())

with st.expander("Моделі (.pkl)", expanded=True):
    use_one = st.checkbox(
        "Одна й та сама модель для тривоги, вибуху та артилерії",
        value=False,
        help="Швидке перемикання: один файл — три показники.",
    )
    if use_one:
        one = st.selectbox("Модель", options, index=0, key="model_one")
        alarm_choice = explosion_choice = artillery_choice = one
    else:
        c1, c2, c3 = st.columns(3)
        with c1:
            alarm_choice = st.selectbox("Тривога (alarm)", options, index=0, key="alarm")
        with c2:
            explosion_choice = st.selectbox("Вибух (explosion)", options, index=0, key="explosion")
        with c3:
            artillery_choice = st.selectbox("Артилерія (artillery)", options, index=0, key="artillery")

_feat_scope = f"{d.isoformat()}_{UI_REGION_LABELS.index(region)}"
_defaults = _neutral_defaults_row(region, d.isoformat())

with st.expander("Фічі для моделі", expanded=False):
    st.caption(
        "База фіч залежить від **обраної дати й регіону**: без реальних даних пайплайну додається календарний шаблон "
        "(погода/лаги/текст змінюються по добі року), щоб ймовірність не була однаковою щодня. "
        "Модель і далі бачить лише колонки з `.pkl`. "
        "`GET …/predict/supported-feature-params` — список query. "
        "Вимкнути шаблон: змінна середовища `WARWATCH_NO_TEMPLATE_CALENDAR=1`."
    )
    g1, g2 = st.columns(2)
    with g1:
        st.markdown("**Лаги / текст**")
        feat_roll = _sl("alarm_roll7_mean", "alarm_roll7_mean", _defaults, f"fr_{_feat_scope}_roll")
        feat_lag = _sl("alarm_lag1", "alarm_lag1", _defaults, f"fr_{_feat_scope}_lag1")
        feat_lag2 = _sl("alarm_lag2", "alarm_lag2", _defaults, f"fr_{_feat_scope}_lag2")
        feat_txt = _sl("text_intensity_index", "text_intensity_index", _defaults, f"fr_{_feat_scope}_txt")
        feat_dur_lag = _sl("duration_lag1", "duration_lag1", _defaults, f"fr_{_feat_scope}_dlag")
    with g2:
        st.markdown("**Тривоги**")
        feat_ac = _sl("alarm_count", "alarm_count", _defaults, f"fr_{_feat_scope}_ac")
        feat_dur = _sl(
            "alarm_total_duration_min",
            "alarm_total_duration_min",
            _defaults,
            f"fr_{_feat_scope}_dur",
        )
        feat_all_reg = _sl("alarm_all_region", "alarm_all_region", _defaults, f"fr_{_feat_scope}_ar")
    w1, w2 = st.columns(2)
    with w1:
        st.markdown("**Погода**")
        feat_temp = _sl("day_temp (°C)", "day_temp", _defaults, f"fr_{_feat_scope}_dt")
        feat_tmax = _sl("day_tempmax", "day_tempmax", _defaults, f"fr_{_feat_scope}_dtx")
        feat_tmin = _sl("day_tempmin", "day_tempmin", _defaults, f"fr_{_feat_scope}_dtn")
        feat_hum = _sl("day_humidity", "day_humidity", _defaults, f"fr_{_feat_scope}_hum")
        feat_wind = _sl("day_windspeed", "day_windspeed", _defaults, f"fr_{_feat_scope}_wnd")
    with w2:
        st.markdown("**Reddit**")
        feat_rpc = _sl("reddit_post_count", "reddit_post_count", _defaults, f"fr_{_feat_scope}_rpc")
        feat_ras = _sl("reddit_avg_score", "reddit_avg_score", _defaults, f"fr_{_feat_scope}_ras")

if st.button("Отримати прогноз", type="primary"):
    params: dict = {"region": region, "date": d.isoformat()}
    if not _is_auto_choice(alarm_choice):
        params["alarm_model"] = alarm_choice
    if not _is_auto_choice(explosion_choice):
        params["explosion_model"] = explosion_choice
    if not _is_auto_choice(artillery_choice):
        params["artillery_model"] = artillery_choice
    for qname, col in sorted(QUERY_PARAM_TO_COLUMN.items(), key=lambda x: x[0]):
        params[qname] = float(
            {
                "feat_alarm_roll7_mean": feat_roll,
                "feat_day_temp": feat_temp,
                "feat_alarm_lag1": feat_lag,
                "feat_alarm_lag2": feat_lag2,
                "feat_text_intensity_index": feat_txt,
                "feat_alarm_count": feat_ac,
                "feat_alarm_total_duration_min": feat_dur,
                "feat_alarm_all_region": feat_all_reg,
                "feat_day_tempmax": feat_tmax,
                "feat_day_tempmin": feat_tmin,
                "feat_day_humidity": feat_hum,
                "feat_day_windspeed": feat_wind,
                "feat_reddit_post_count": feat_rpc,
                "feat_reddit_avg_score": feat_ras,
                "feat_duration_lag1": feat_dur_lag,
            }[qname]
        )
    try:
        r = requests.get(f"{api_url}/predict", params=params, timeout=120)
        r.raise_for_status()
        data = r.json()
    except requests.RequestException as e:
        st.error(f"Помилка API ({api_url}): {e}")
        st.info("Запусти бекенд: `uvicorn frontend.api.main:app --reload` з кореня репозиторію.")
        st.stop()

    st.subheader("Результат")
    c1, c2, c3 = st.columns(3)
    c1.metric("Тривога P", f"{data.get('alarm_prob', 0):.3f}")
    c2.metric("Вибух P", f"{data.get('explosion_prob', 0):.3f}")
    c3.metric("Артилерія P", f"{data.get('artillery_prob', 0):.3f}")

    chart_df = pd.DataFrame(
        {
            "Event": ["Тривога", "Вибух", "Артилерія"],
            "Probability": [
                data.get("alarm_prob", 0),
                data.get("explosion_prob", 0),
                data.get("artillery_prob", 0),
            ],
        }
    )
    st.bar_chart(chart_df, x="Event", y="Probability", horizontal=True)

    st.dataframe(
        {
            "поле": ["регіон", "дата", "режим", "профіль фіч", "каталог моделей"],
            "значення": [
                data.get("region"),
                data.get("date"),
                data.get("mode"),
                data.get("feature_profile"),
                data.get("model_dir"),
            ],
        },
        hide_index=True,
        use_container_width=True,
    )
    if data.get("feature_overrides"):
        st.caption("Передані перевизначення фіч")
        st.json(data["feature_overrides"])
    with st.expander("Які файли використані"):
        st.json(data.get("models", {}))
    if data.get("binary_classifier_split"):
        with st.expander("Розклад класифікатора P(0) / P(1)"):
            st.caption("Якщо P(1)=1 і P(0)=0, потрапили в «чистий» лист дерева — це очікувано для шаблонних фіч.")
            st.json(data["binary_classifier_split"])

st.sidebar.markdown("### API")
st.sidebar.code(f"{api_url}/predict/supported-feature-params", language="text")
st.sidebar.code(
    f"{api_url}/predict?region=Kyiv&date=2024-06-01"
    f"&alarm_model=randomforest_model.pkl"
    f"&feat_alarm_total_duration_min=11",
    language="text",
)
st.sidebar.markdown("`WARWATCH_API_URL` — база FastAPI.")
