# WarWatch — War Event Prediction SaaS

> A Python-based SaaS application that predicts war events (air alarms, explosions, artillery fire) for Ukrainian regions based on ISW reports, weather data, and situational awareness from neighboring regions.

---

##  Team

| Role | Name |
|------|------|
| Team Leader | Ковальов Євген |
| Member | Тавлуй Софія |
| Member | Поляков Кирило |
| Member | Сніховська Владислава |
| Member | Барабанова Софія |

---

##  Links

| Resource | URL |
|----------|-----|
| GitHub Repo | https://github.com/exEKS/python_SaaS.git |
| Google Drive | https://drive.google.com/drive/folders/1JGf5MwedCwO2nMSBkIvQyJGPQQ08rK17?usp%20=drive_link |

---

##  Project Goal

Build a predictive system that forecasts three types of war events per region:
- **Air Alarms**
- **Explosions**
- **Artillery Fire**

---

##  System Architecture

The system consists of three core modules with an optional shared storage layer. External data flows from APIs and web scraping through the **Data Receiver**, is transformed and modelled in the **Forecasting** pipeline, and predictions are served via **FastAPI** and visualised in a **Streamlit** dashboard.

### 1. Data Receiver

Collects all raw data from external sources on a daily schedule (APScheduler).

| Script | Input | Output | Description |
|--------|-------|--------|-------------|
| `isw_scraper.py` | Date range / URL list | JSON `{date, title, text}` | Scrapes each daily ISW report from understandingwar.org |
| `weather_forecast.py` | Region coords, API key | JSON: 24h hourly forecast | Calls VisualCrossing REST API. Returns temp, precipitation, wind, humidity |
| `alarms_receiver.py` | API key, Region list | JSON: active alert snapshot | Polls devs.alerts.in.ua / api.ukrainealarm.com every N minutes |
| `data_storage.py` | All collected data | Saved to local FS / Google Drive | Handles persistent storage of raw data |

### 2. Forecasting

Prepares features and trains/runs predictive models.

| Script | Input | Output | Description |
|--------|-------|--------|-------------|
| `preprocessing.py` | Raw text, Weather CSV, Alarm logs | Cleaned merged DataFrame | Strips HTML, tokenises ISW text, normalises weather numerics, aligns by date/region |
| `feature_engineering.py` | Cleaned DataFrame | Feature matrix X, Label vectors y | Adds lag features (t-1, t-2, t-7), 7-day rolling stats, TF-IDF vectors (top 200 features) |
| `model.py` | Feature matrix X, Labels y | `.pkl` model files, Prediction probs | Trains RandomForestClassifier and XGBClassifier with 5-fold CV |

### 3. Frontend (Streamlit + FastAPI)

Two-layer interface: FastAPI serves the model as a REST API; Streamlit provides an interactive dashboard.

| Component | Input | Output | Description |
|-----------|-------|--------|-------------|
| `api/main.py` (FastAPI) | `GET /predict?region=Kyiv&date=…` | JSON: `alarm_prob`, `explosion_prob`, `artillery_prob` | Loads trained `.pkl` model, calls `model.predict()`, returns JSON |
| `frontend/app.py` (Streamlit) | User selects region + date | Interactive table + bar chart | Calls FastAPI `/predict`, renders probabilities with `st.dataframe` and `st.bar_chart` |

---

##  Repository Structure

```
warwatch/
│
├── data_receiver/
│   ├── isw_scraper.py          # Scrapes daily ISW reports
│   ├── weather_forecast.py     # Fetches 24h weather forecast
│   ├── alarms_receiver.py      # Fetches live alarm status per region
│   └── data_storage.py         # Handles data persistence
│
├── forecasting/
│   ├── preprocessing.py        # Feature engineering & text preprocessing (planned)
│   ├── feature_engineering.py    # Lag features, rolling stats, TF-IDF
│   ├── default_feature_row.py    # Default feature template for API / demos
│   ├── model_runtime.py          # Load .pkl, align features, predict
│   ├── prediction_service.py     # Region/date → alarm / explosion / artillery probabilities
│   └── inference_catboost.py     # CLI: run all models in the model folder
│
├── frontend/
│   ├── app.py                  # Streamlit UI
│   └── api/
│       └── main.py             # FastAPI backend (`/predict`, `/health`, `/models`)
│
├── data/
│   ├── raw/                    # Raw scraped/downloaded data
│   └── processed/              # Cleaned, feature-engineered data
│
├── models/                     # Saved trained models (.pkl)
├── notebooks/                  # EDA and experiments
├── requirements.txt
└── README.md
```

---

##  Data Sources & Dataset Overview

| Source | Type | Period | Format | Est. Size |
|--------|------|--------|--------|-----------|
| [air-alarms.in.ua](https://air-alarms.in.ua/) | War events (alarms/expl.) | 2022-02-24 – 2026-03-01 | CSV (event log) | ~300 000 rows |
| [understandingwar.org](https://www.understandingwar.org/) | ISW daily reports | 2022-02-24 – 2026-03-01 | HTML → text/JSON | ~1 500 docs |
| [visualcrossing.com](https://www.visualcrossing.com/weather-api) | Weather forecast (24h) | Rolling | API / CSV (hourly) | ~500 rows/day |
| Historical weather | Weather data | 2022-02-24 – 2026-03-01 | CSV (hourly) | ~35 000 rows |
| [devs.alerts.in.ua](https://devs.alerts.in.ua/) | Live alarm status | Rolling 24h | JSON (API) | ~500 rows/day |

>  Historical war events data provided by Богдан Васильчук via air-alarms.in.ua — for internal use only. Must not be uploaded publicly or distributed.

---

##  Data Analysis & Preparation Strategy

### Text Data (ISW Reports)

ISW reports are the primary unstructured source. The pipeline converts raw HTML into numeric feature vectors:

1. **Collection** — Scraper downloads each daily report; stores `{date, title, URL, full text}`
2. **Cleaning** — HTML tags stripped; boilerplate removed via regex; text lowercased; Ukrainian/Russian characters preserved
3. **Tokenisation** — Split on whitespace and punctuation; custom stopword list (English + Ukrainian) applied
4. **Vectorisation** — TF-IDF vectoriser (max 200 features, unigrams + bigrams) fitted on training corpus → 200-dim sparse vector per report. Optional: `sentence-transformers` (paraphrase-multilingual-MiniLM) for 384-dim dense embeddings
5. **Feature alignment** — Each vector assigned to publication date; regions mentioned receive a binary flag; final row keyed on `(date, region)`

### Structured Data

- **Weather:** Hourly data resampled to daily aggregates (avg/max/min) per region. Missing values forward-filled (max 3h), then mean-imputed.
- **Event logs:** Alarm/explosion counts aggregated per `(date, region, event_type)`. Lag features t-1, t-2, t-7 days and 7-day rolling mean added.
- **Labels:** Binary targets — `alarm`, `explosion`, `artillery` (1 = at least one event in next 24h for that region).
- **Train/test split:** 80% training (2022-02-24 – 2025-06-01), 20% test (2025-06-01 – 2026-03-01). Temporal order preserved — no data leakage.

---

##  Setup & Installation

```bash
# Clone the repository
git clone https://github.com/exEKS/python_SaaS.git
cd python_SaaS

# Install dependencies
pip install -r requirements.txt

# Set environment variables
cp .env.example .env
# Fill in your API keys: VISUALCROSSING_API_KEY, UKRAINE_ALARM_TOKEN
```

---

##  Running the Application

From the repository root (with `pip install -r requirements.txt` and optional `.env` from `.env.example`):

```bash
# Start FastAPI backend (REST API for predictions)
uvicorn frontend.api.main:app --reload --host 0.0.0.0 --port 8000

# In another terminal — Streamlit UI (calls the API; set WARWATCH_API_URL if not localhost:8000)
streamlit run frontend/app.py

# Optional: batch inference over all .pkl in the model folder
python forecasting/inference_catboost.py
```

Put trained `.pkl` files under `models/` or set `WARWATCH_MODEL_DIR` to your Drive/Colab folder. The API exposes `GET /predict?region=Kyiv&date=2024-06-01` returning `alarm_prob`, `explosion_prob`, `artillery_prob`. Optional: `&alarm_model=your_model.pkl` (file inside `WARWATCH_MODEL_DIR` / `models/`, or absolute path). Without it, the first matching `.pkl` by name heuristics is used (e.g. names containing `catboost`, `decision`, …). Set `WARWATCH_MODEL_ALARM` in `.env` for a fixed default without query params.

```bash
# Run ISW scraper
python data_receiver/isw_scraper.py

# Run weather fetcher
python data_receiver/weather_forecast.py
```

---

##  Environment Variables

Copy `.env.example` to `.env` and adjust.

```
VISUALCROSSING_API_KEY=your_key_here
UKRAINE_ALARM_TOKEN=your_token_here
WARWATCH_MODEL_DIR=          # optional: folder with .pkl models (defaults to models/ or forecasting/ if .pkl present)
WARWATCH_API_URL=http://127.0.0.1:8000   # Streamlit → FastAPI base URL
WARWATCH_MODEL_ALARM=        # optional: explicit path to alarm-focused .pkl
WARWATCH_MODEL_EXPLOSION=
WARWATCH_MODEL_ARTILLERY=
```

---

##  Team Responsibility List

| Team Member | Tasks Completed |
|-------------|----------------|
| Kovalov Evgen | Developed an inference script, deployed the model as a SaaS (UI + API), implemented prediction storage and retrieval, set up retraining, recorded a demo video, and prepared a report, frontend, backend. |
| Tavlui Sofia | Defined the top model, wrote the motivation for each model, retrained the top model, made an API endpoint for 24-hour predictions, created capability (script) to retrain the model with the historical dataset + new data on a regular basis + model validation. |
| Snikhovska Vladyslava | Trained and develoved RandomForest model, implemented endpoint to update forecast, README file editing, final report preparation, full final presentation, recorded a demo video. |
| Polyakov Kirilo | Trained/tuned CatBoost models, automated 24h regional forecasting via hourly Cron on AWS EC2, and implemented a retraining pipeline with automated validation. |
| Barabanova Sofia | Decision Tree model development and hyperparameter tuning, evaluation metrics & confusion matrix analysis, feature importance visualization (Top-20), and model improvement strategy (TODO list). |
