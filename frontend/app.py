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

from forecasting.default_feature_row import UI_REGION_LABELS

st.set_page_config(page_title="WarWatch", layout="wide")
st.title("WarWatch — event risk dashboard")
st.caption("Probabilities for air alarms, explosions, and artillery in the next 24h (model-dependent).")

api_url = os.environ.get("WARWATCH_API_URL", "http://127.0.0.1:8000").rstrip("/")

col1, col2 = st.columns(2)
with col1:
    region = st.selectbox("Region", UI_REGION_LABELS, index=0)
with col2:
    from datetime import date as date_type

    d = st.date_input("Date", value=date_type.today())

if st.button("Get prediction", type="primary"):
    params = {"region": region, "date": d.isoformat()}
    try:
        r = requests.get(f"{api_url}/predict", params=params, timeout=120)
        r.raise_for_status()
        data = r.json()
    except requests.RequestException as e:
        st.error(f"API request failed ({api_url}): {e}")
        st.info("Start the API: `uvicorn frontend.api.main:app --reload` from the repo root.")
        st.stop()

    st.subheader("Results")
    c1, c2, c3 = st.columns(3)
    c1.metric("Air alarm P", f"{data.get('alarm_prob', 0):.3f}")
    c2.metric("Explosion P", f"{data.get('explosion_prob', 0):.3f}")
    c3.metric("Artillery P", f"{data.get('artillery_prob', 0):.3f}")

    chart_df = pd.DataFrame(
        {
            "Event": ["Air alarm", "Explosion", "Artillery"],
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
            "field": ["region", "date", "mode", "model_dir"],
            "value": [
                data.get("region"),
                data.get("date"),
                data.get("mode"),
                data.get("model_dir"),
            ],
        },
        hide_index=True,
        use_container_width=True,
    )
    with st.expander("Models used"):
        st.json(data.get("models", {}))

st.sidebar.markdown("### API")
st.sidebar.code(f"{api_url}/predict?region=Kyiv&date=2024-06-01", language="text")
st.sidebar.markdown("Set `WARWATCH_API_URL` if the API runs elsewhere.")
