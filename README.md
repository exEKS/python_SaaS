# WarWatch — War Event Prediction SaaS

> A Python-based SaaS application that predicts war events (air alarms, explosions, artillery fire) for Ukrainian regions based on ISW reports, weather data, and situational awareness from neighboring regions.

---

## 👥 Team

| Role | Name |
|------|------|
| Team Leader | Ковальов Євген |
| Member | Барабанова Софія |
| Member | Тавлуй Софія |
| Member | Сніховська Владислава |
| Member | Поляков Кирило |

---

## 🎯 Project Goal

Build a predictive system that forecasts three types of war events per region:
- **Air Alarms**
- **Explosions**
- **Artillery Fire**

---

## 🏗️ System Architecture

The system consists of three core modules:

### 1. Data Receiver
Collects and stores all input data:
- **ISW Reports** — daily scraping from [understandingwar.org](https://www.understandingwar.org/)
- **Weather Forecasts** — next 24h via [VisualCrossing API](https://www.visualcrossing.com/weather-api)
- **Regional Alarm Status** — live situational data via [devs.alerts.in.ua](https://devs.alerts.in.ua/) or [api.ukrainealarm.com](https://api.ukrainealarm.com/)

### 2. Forecasting Module
- Prepares and cleans raw data (NLP for ISW text, feature engineering for weather/alarms)
- Runs prediction models: **Random Forest** and **XGBoost**
- Outputs per-region event probability for the next 24 hours

### 3. Frontend (UI)
- Built with **Streamlit**
- Displays predictions on a regional map of Ukraine
- Shows historical event data and model confidence scores
- Backend API powered by **FastAPI**

---

## 📁 Repository Structure

```
warwatch/
│
├── data_receiver/
│   ├── isw_scraper.py          # Scrapes daily ISW reports
│   ├── weather_fetcher.py      # Fetches 24h weather forecast
│   └── alarms_fetcher.py       # Fetches live alarm status per region
│
├── forecasting/
│   ├── data_preparation.py     # Feature engineering & text preprocessing
│   ├── train_model.py          # Model training (RF, XGBoost)
│   └── predict.py              # Run predictions
│
├── frontend/
│   ├── app.py                  # Streamlit UI
│   └── api.py                  # FastAPI backend
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

## 📦 Data Sources

| Source | Type | Access |
|--------|------|--------|
| [air-alarms.in.ua](https://air-alarms.in.ua/) | Historical war events | Provided by lecturer |
| [understandingwar.org](https://www.understandingwar.org/) | ISW daily reports (2022-02-24 to 2026-03-01) | Scraped |
| [visualcrossing.com](https://www.visualcrossing.com/weather-api) | Weather forecast (24h) | API |
| Historical weather | Weather data | Provided by lecturer |
| [devs.alerts.in.ua](https://devs.alerts.in.ua/) | Live alarm status | API |

---

## Setup & Installation

```bash
# Clone the repository
git clone https://github.com/<your-org>/warwatch.git
cd warwatch

# Install dependencies
pip install -r requirements.txt

# Set environment variables
cp .env.example .env
# Fill in your API keys: VISUALCROSSING_API_KEY, UKRAINE_ALARM_TOKEN
```

---

## Running the Application

```bash
# Start FastAPI backend
uvicorn frontend.api:app --reload

# Start Streamlit UI
streamlit run frontend/app.py

# Run ISW scraper
python data_receiver/isw_scraper.py

# Run weather fetcher
python data_receiver/weather_fetcher.py
```

---

## Environment Variables

```
VISUALCROSSING_API_KEY=your_key_here
UKRAINE_ALARM_TOKEN=your_token_here
```

---

## License

For internal academic use only. Historical war events data provided by Богдан Васильчук via air-alarms.in.ua — must not be distributed or uploaded publicly.
