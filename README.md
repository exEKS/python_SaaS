# WarWatch — War Event Prediction SaaS

![Python](https://img.shields.io/badge/Python-3.10+-blue?logo=python&logoColor=white)
![FastAPI](https://img.shields.io/badge/FastAPI-0.110+-009688?logo=fastapi&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-1.x-FF4B4B?logo=streamlit&logoColor=white)
![XGBoost](https://img.shields.io/badge/XGBoost-ML-orange?logo=xgboost&logoColor=white)
![License](https://img.shields.io/badge/License-Academic-lightgrey)

> **WarWatch** is a Python-based SaaS application that predicts war-related events — air alarms, explosions, and artillery fire — for Ukrainian regions, based on ISW reports, historical weather data, and situational awareness from neighboring areas.

---

## Table of Contents

- [Team](#-team)
- [Project Goal](#-project-goal)
- [System Architecture](#-system-architecture)
- [Data Sources](#-data-sources--dataset-overview)
- [Data Preparation Strategy](#-data-analysis--preparation-strategy)
- [Repository Structure](#-repository-structure)
- [Setup & Installation](#-setup--installation)
- [Running the Application](#-running-the-application)
- [Environment Variables](#-environment-variables)
- [Team Responsibilities](#-team-responsibility-list)

---

## 👥 Team

| Role | Name |
|------|------|
| Team Leader | Ковальов Євген |
| Member | Тавлуй Софія |
| Member | Поляков Кирило |
| Member | Сніховська Владислава |
| Member | Барабанова Софія |

---

## 🔗 Links

| Resource | URL |
|----------|-----|
| GitHub Repo | [github.com/exEKS/python_SaaS](https://github.com/exEKS/python_SaaS) |
| Google Drive | [Project folder](https://drive.google.com/drive/folders/1JGf5MwedCwO2nMSBkIvQyJGPQQ08rK17) |

---

## 🎯 Project Goal

Build a predictive system that forecasts three types of war-related events per Ukrainian region and date:

| Event Type | Description |
|---|---|
| 🚨 **Air Alarms** | Active air raid alert in the region |
| 💥 **Explosions** | Recorded explosion events |
| 🔫 **Artillery Fire** | Artillery activity in the region |

Each prediction returns a probability score for the next 24 hours, served via a REST API and visualised in an interactive Streamlit dashboard.

---

## 🏗 System Architecture

The system consists of three core modules. External data flows from APIs and web scraping through the **Data Receiver**, is transformed and modelled in the **Forecasting** pipeline, and predictions are served via **FastAPI** and visualised in a **Streamlit** dashboard.

```
External Sources
    │
    ▼
┌─────────────────────────────┐
│      Data Receiver          │  APScheduler (daily / polling)
│  isw_scraper  │  alarms_rx  │
│  weather_api  │  data_store │
└────────────┬────────────────┘
             │ raw JSON / CSV
             ▼
┌─────────────────────────────┐
│       Forecasting           │
│  preprocessing → features   │
│  model training → .pkl      │
└────────────┬────────────────┘
             │ predictions
             ▼
┌─────────────────────────────┐
│     Frontend Layer          │
│  FastAPI /predict endpoint  │
│  Streamlit dashboard        │
└─────────────────────────────┘
```

### 1. Data Receiver

Collects raw data from external sources on a daily schedule (APScheduler).

| Script | Input | Output | Description |
|--------|-------|--------|-------------|
| `isw_scraper.py` | Date range / URL list | JSON `{date, title, text}` | Scrapes daily ISW reports from understandingwar.org |
| `weather_forecast.py` | Region coords, API key | JSON: 24h hourly forecast | Calls VisualCrossing REST API — temp, precipitation, wind, humidity |
| `alarms_receiver.py` | API key, region list | JSON: active alert snapshot | Polls devs.alerts.in.ua / api.ukrainealarm.com every N minutes |
| `data_storage.py` | All collected data | Saved to local FS / Google Drive | Handles persistent storage of raw data |

### 2. Forecasting

Prepares features and trains / runs predictive models.

| Script | Input | Output | Description |
|--------|-------|--------|-------------|
| `preprocessing.py` | Raw text, Weather CSV, Alarm logs | Cleaned merged DataFrame | Strips HTML, tokenises ISW text, normalises weather numerics, aligns by date/region |
| `feature_engineering.py` | Cleaned DataFrame | Feature matrix X, label vectors y | Adds lag features (t-1, t-2, t-7), 7-day rolling stats, TF-IDF vectors (top 200 features) |
| `model.py` | Feature matrix X, labels y | `.pkl` model files, prediction probs | Trains RandomForestClassifier and XGBClassifier with 5-fold CV |

### 3. Frontend (FastAPI + Streamlit)

A two-layer interface: FastAPI serves the model as a REST API; Streamlit provides the interactive dashboard.

| Component | Input | Output | Description |
|-----------|-------|--------|-------------|
| `api/main.py` | `GET /predict?region=Kyiv&date=…` | JSON: `alarm_prob`, `explosion_prob`, `artillery_prob` | Loads trained `.pkl` model, returns JSON predictions |
| `frontend/app.py` | User selects region + date | Interactive table + bar chart | Calls FastAPI `/predict`, renders with `st.dataframe` and `st.bar_chart` |

---

## 📁 Repository Structure

```
python_SaaS/
│
├── .streamlit/
│   └── config.toml
│
├── data/
│   ├── processed/
│   └── raw/
│
├── data_receiver/
│   ├── __init__.py
│   ├── alarms_receiver.py        # Fetches live alarm status per region
│   ├── data_storage.py           # Handles data persistence
│   ├── isw_daily_scraper.py      # Scrapes latest daily ISW report
│   ├── isw_historic_scraper.py   # Scrapes historical ISW archive
│   ├── isw_scraper.py            # Core ISW scraping logic
│   ├── reddit_scraper.py         # Scrapes Reddit for situational data
│   ├── telegram_scraper.py       # Scrapes Telegram channels
│   └── weather_forecast.py      # Fetches 24h weather forecast
│
├── eda/
│   ├── alarms_visualisation.ipynb
│   ├── check_stats.py
│   ├── heads.py
│   ├── isw_visualisation.ipynb
│   ├── make_features.py
│   └── weather_visualisation.ipynb
│
├── forecasting/
│   ├── __init__.py
│   ├── 2_linear_regression_training.py
│   ├── RandomForestModel.ipynb
│   ├── catboost_model.py         # CatBoost training pipeline
│   ├── catboost_model_new.py
│   ├── default_feature_row.py    # Default feature template for API / demos
│   ├── feature_engineering.py    # Lag features, rolling stats, TF-IDF
│   ├── feature_query_params.py
│   ├── inference_catboost.py     # CLI: run all models in the model folder
│   ├── model.py                  # Core model training logic
│   ├── model_3_fixed.ipynb
│   ├── model_logistic_regression.ipynb
│   ├── model_runtime.py          # Load .pkl, align features, predict
│   ├── nlp_tokenizer.py
│   ├── paths.py
│   ├── predict_24h.py            # 24h ahead prediction logic
│   ├── prediction_service.py     # Region/date → event probability scores
│   ├── requirements_inference.txt
│   ├── retrain_model.py          # Automated retraining pipeline
│   ├── train_models.py
│   └── weather_collector.py
│
├── frontend/
│   ├── api/
│   │   ├── __init__.py
│   │   ├── forecast_api.py
│   │   └── main.py               # FastAPI backend (/predict, /health, /models)
│   └── app.py                    # Streamlit UI
│
├── models/                       # Saved trained models (.pkl)
│
├── .env.example
├── .gitignore
├── HW2_Report.pdf
├── LICENSE
├── README.md
└── requirements.txt
```

---

## 📊 Data Sources & Dataset Overview

| Source | Type | Period | Format | Est. Size |
|--------|------|--------|--------|-----------|
| [air-alarms.in.ua](https://air-alarms.in.ua/) | War events (alarms / explosions) | 2022-02-24 – 2026-03-01 | CSV (event log) | ~300 000 rows |
| [understandingwar.org](https://www.understandingwar.org/) | ISW daily reports | 2022-02-24 – 2026-03-01 | HTML → text / JSON | ~1 500 docs |
| [visualcrossing.com](https://www.visualcrossing.com/weather-api) | Weather forecast (24h rolling) | Rolling | API / CSV (hourly) | ~500 rows/day |
| Historical weather | Weather data | 2022-02-24 – 2026-03-01 | CSV (hourly) | ~35 000 rows |
| [devs.alerts.in.ua](https://devs.alerts.in.ua/) | Live alarm status | Rolling 24h | JSON (API) | ~500 rows/day |

> ⚠️ **Data Notice:** Historical war events data provided by Богдан Васильчук via air-alarms.in.ua — for **internal use only**. Must not be uploaded publicly or distributed.

---

## 🔬 Data Analysis & Preparation Strategy

### Text Data (ISW Reports)

ISW reports are the primary unstructured source. The pipeline converts raw HTML into numeric feature vectors:

1. **Collection** — Scraper downloads each daily report and stores `{date, title, URL, full text}`
2. **Cleaning** — HTML stripped, boilerplate removed via regex, text lowercased; Ukrainian/Russian characters preserved
3. **Tokenisation** — Split on whitespace and punctuation; custom stopword list (English + Ukrainian) applied
4. **Vectorisation** — TF-IDF (max 200 features, unigrams + bigrams) → 200-dim sparse vector per report. Optional: `sentence-transformers` (`paraphrase-multilingual-MiniLM`) for 384-dim dense embeddings
5. **Feature alignment** — Each vector assigned to its publication date; regions mentioned receive a binary flag; final row keyed on `(date, region)`

### Structured Data

- **Weather:** Hourly data resampled to daily aggregates (avg / max / min) per region. Missing values forward-filled (max 3h), then mean-imputed.
- **Event logs:** Alarm / explosion counts aggregated per `(date, region, event_type)`. Lag features t-1, t-2, t-7 days and 7-day rolling mean added.
- **Labels:** Binary targets — `alarm`, `explosion`, `artillery` (1 = at least one event in next 24h for that region).
- **Train / test split:** 80% training (2022-02-24 – 2025-06-01), 20% test (2025-06-01 – 2026-03-01). Temporal order preserved — no data leakage.

---

## ⚙️ Setup & Installation

**Requirements:** Python 3.10+

```bash
# Clone the repository
git clone https://github.com/exEKS/python_SaaS.git
cd python_SaaS

# Install dependencies
pip install -r requirements.txt

# Configure environment variables
cp .env.example .env
# Fill in: VISUALCROSSING_API_KEY, UKRAINE_ALARM_TOKEN
```

---

## 🚀 Running the Application

Make sure you have completed the installation steps above and placed trained `.pkl` files under `models/` (or set `WARWATCH_MODEL_DIR`).

```bash
# 1. Start FastAPI backend
uvicorn frontend.api.main:app --reload --host 0.0.0.0 --port 8000

# 2. In a separate terminal — start Streamlit dashboard
streamlit run frontend/app.py
```

The API exposes `GET /predict?region=Kyiv&date=2024-06-01` returning:

```json
{
  "alarm_prob": 0.82,
  "explosion_prob": 0.47,
  "artillery_prob": 0.31
}
```

**Optional query parameters:**
- `&alarm_model=your_model.pkl` — specify model file inside `WARWATCH_MODEL_DIR` / `models/`
- If omitted, the first matching `.pkl` is selected by name heuristics (e.g. `catboost`, `decision`, …)
- Set `WARWATCH_MODEL_ALARM` in `.env` for a fixed default

```bash
# Optional: batch inference over all .pkl in the model folder
python forecasting/inference_catboost.py

# Run individual data collectors
python data_receiver/isw_scraper.py
python data_receiver/weather_forecast.py
```

---

## 🔑 Environment Variables

Copy `.env.example` to `.env` and fill in your values:

```env
# Required
VISUALCROSSING_API_KEY=your_key_here
UKRAINE_ALARM_TOKEN=your_token_here

# Optional — model configuration
WARWATCH_MODEL_DIR=          # Folder with .pkl models (defaults to models/ or forecasting/)
WARWATCH_API_URL=http://127.0.0.1:8000   # Streamlit → FastAPI base URL

# Optional — pin specific models per event type
WARWATCH_MODEL_ALARM=
WARWATCH_MODEL_EXPLOSION=
WARWATCH_MODEL_ARTILLERY=
```

---

## 👷 Team Responsibility List

| Team Member | Contribution |
|-------------|-------------|
| **Kovalov Evgen** | Inference script, full SaaS deployment (UI + API), prediction storage & retrieval, retraining pipeline setup, demo video, final report, frontend & backend |
| **Tavlui Sofiia** | Top model selection & motivation, model retraining, 24h prediction API endpoint, automated retraining script with historical + new data and model validation |
| **Snikhovska Vladyslava** | RandomForest model training & development, forecast update endpoint, README editing, final report & presentation, demo video |
| **Poliakov Kyrylo** | CatBoost model training & tuning, automated 24h regional forecasting via hourly Cron on AWS EC2, retraining pipeline with automated validation |
| **Barabanova Sofiia** | Decision Tree development & hyperparameter tuning, evaluation metrics & confusion matrix analysis, feature importance visualisation (Top-20), model improvement strategy |
