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
| Google Drive | https://drive.google.com/drive/folders/1ZVW1njMU2iy5T0t2-2ByaOsN6rpm6I72?usp=share_link |

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
│   ├── preprocessing.py        # Feature engineering & text preprocessing
│   ├── feature_engineering.py  # Lag features, rolling stats, TF-IDF
│   └── model.py                # Model training (RF, XGBoost) & prediction
│
├── frontend/
│   ├── app.py                  # Streamlit UI
│   └── api/
│       └── main.py             # FastAPI backend
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

```bash
# Start FastAPI backend
uvicorn frontend.api.main:app --reload

# Start Streamlit UI
streamlit run frontend/app.py

# Run ISW scraper
python data_receiver/isw_scraper.py

# Run weather fetcher
python data_receiver/weather_forecast.py
```

---

##  Environment Variables

```
VISUALCROSSING_API_KEY=your_key_here
UKRAINE_ALARM_TOKEN=your_token_here
```

---

##  Team Responsibility List

| Team Member | Tasks Completed |
|-------------|----------------|
| Kovalov Evgen | README.md, System architecture diagram, Module descriptions & I/O diagrams, Data analysis plan & text preparation strategy, Final PDF report compilation, Code Review |
| Tavlui Sofia | *(to be filled)* |
| Snikhovska Vladyslava | *(to be filled)* |
| Polyakov Kirilo | *(to be filled)* |
| Barabanova Sofia | *(to be filled)* |

>  Screenshots of collected data, Postman API responses (VisualCrossing and Alarms API), and final GitHub / Google Drive links must be inserted by the responsible members before the **06/03/2026** deadline.
