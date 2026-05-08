# WarWatch — War Event Prediction SaaS

![Python](https://img.shields.io/badge/Python-3.10+-blue?logo=python&logoColor=white)
![FastAPI](https://img.shields.io/badge/FastAPI-0.110+-009688?logo=fastapi&logoColor=white)
![React](https://img.shields.io/badge/React-18+-61DAFB?logo=react&logoColor=black)
![Streamlit](https://img.shields.io/badge/Streamlit-optional-FF4B4B?logo=streamlit&logoColor=white)
![License](https://img.shields.io/badge/License-Academic-lightgrey)

> **WarWatch** — система оцінки ймовірності повітряних тривог по областях України на обрану дату. Модель (`.pkl`) обслуговується через **FastAPI**; основний інтерфейс користувача — **React** карта; опційно доступний **Streamlit** для швидких тестів API.

---

## Table of Contents

- [Team](#-team)
- [Project Goal](#-project-goal)
- [System Architecture](#-system-architecture)
- [API Overview](#-api-overview)
- [Live Data and Inference](#-live-data-and-inference)
- [Repository Structure](#-repository-structure)
- [Data Sources](#-data-sources--dataset-overview)
- [Data Preparation Strategy](#-data-analysis--preparation-strategy)
- [Setup & Installation](#-setup--installation)
- [Running Locally](#-running-locally)
- [Production (nginx + EC2)](#-production-nginx--ec2)
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

Оцінити **ймовірність повітряної тривоги** на наступні 24 години для кожного регіону (за шаблоном фіч і зафіксованою моделлю). Результат віддається через REST API і візуалізується на карті; є режим **погодинного профілю** (24 точки), який залежить від наявності ознаки `hour` у навченій моделі.

**Важливо:** модель **не перенавчається автоматично** при зборі нових JSON. Нові дані з колекторів лише **підмішуються у вхідні фічі на етапі інференсу** (див. [Live Data and Inference](#-live-data-and-inference)). Оновлення ваг моделі — окремий крок (тренування, новий `.pkl`, перезапуск сервісу).

---

## 🏗 System Architecture

```
External Sources                    Optional: local JSON (same host)
         │                                    │
         ▼                                    ▼
┌─────────────────────┐              ┌──────────────────────┐
│   Data Receiver     │──save───────▶│ data/local_live/      │
│ ISW daily scraper   │              │  isw/   alerts/       │
│ Alarm hourly poll   │              └───────────┬──────────┘
└─────────┬───────────┘                          │
          │ raw JSON                             │ read at predict time
          ▼                                      ▼
┌─────────────────────┐              ┌──────────────────────┐
│    Forecasting      │              │  prediction_service   │
│  train → .pkl       │              │  template row +       │
│  feature templates  │              │  live overrides +    │
└─────────┬───────────┘              │  cached model load    │
          │                          └───────────┬──────────┘
          │                                      ▼
          │                          ┌──────────────────────┐
          └─────────────────────────▶│   FastAPI main.py    │
                                     │  /predict /batch     │
                                     │  /predict/hourly     │
                                     │  static: react-app/… │
                                     └──────────┬──────────┘
                                                │
                    ┌───────────────────────────┼───────────────────────────┐
                    ▼                           ▼                           ▼
              React (Vite build)           Streamlit (optional)        nginx reverse proxy
              Map + batch fetch            sanity checks                /  and  /api → :8000
```

### Компоненти

| Шар | Опис |
|-----|------|
| **FastAPI** (`frontend/api/main.py`) | REST API, префікс `/api/...` дубльований для роботи за nginx (`/api` → бекенд). Модель прогрівається при старті; завантажений естіматор кешується в пам’яті. |
| **React** (`frontend/react-app`) | Карта областей, batch-запит усіх регіонів, модалка погодинного профілю. У dev Vite проксі `/api` на порт 8000. |
| **Streamlit** (`frontend/app.py`) | Опційна панель для ручного виклику `/predict`. |
| **Колектори** | `data_receiver/isw_daily_scraper.py` (щоденний звіт ISW), `data_receiver/alarm_daily.py` (щогодинний знімок api.ukrainealarm.com), `data_receiver/purge_local_live_nightly.py` (очищення JSON о 23:59:59 локального часу сервера). |

---

## 🔌 API Overview

Базовий шлях після збірки часто `https://your-host/api/...` (nginx). Прямо з uvicorn також доступні ті самі маршрути без `/api`.

| Метод | Шлях | Призначення |
|--------|------|-------------|
| GET | `/health`, `/api/health` | Перевірка наявності `.pkl` у каталозі моделей |
| GET | `/models`, `/api/models` | Список файлів моделей |
| GET | `/predict` | Один регіон + дата; query `feat_*` для оверрайду фіч (див. `GET /predict/supported-feature-params`) |
| POST | `/predict/batch` | Один запит: масив `regions` + `date` (зменшує навантаження на слабкому VPS) |
| GET | `/predict/hourly` | 24 значення `alarm_prob` по годинах; якщо в моделі немає `hour`, крива буде майже плоскою (див. доку в коді) |
| POST | `/forecast` | Статичний прогноз з `predictions.json` (legacy), якщо файл є |

Приклад відповіді `GET /predict`:

```json
{
  "region": "Kyiv",
  "date": "2026-05-08",
  "alarm_prob": 0.53,
  "mode": "single_model",
  "models": { "alarm": "rf_compressed.pkl (RandomForestClassifier)" },
  "live_context": { "isw": "isw_data_2026-05-08.json", "alerts": "alerts_2026-05-08_14-00-00.json" }
}
```

Поле `live_context` з’являється лише якщо для обраної дати знайдені локальні файли ISW/тривог.

---

## 📂 Live Data and Inference

Каталог за замовчуванням: **`data/local_live/`** (або **`WARWATCH_LOCAL_DATA_DIR`**):

- `isw/isw_data_YYYY-MM-DD.json` — текст денного звіту ISW.
- `alerts/alerts_YYYY-MM-DD_HH-MM-SS.json` — знімок API тривог (останній файл за датою використовується при прогнозі).

Модуль `forecasting/local_live_features.py` зчитує ці файли і формує числові оверрайди (наприклад `text_intensity_index`, `alarm_count`, `alarm_lag1`, …), які **зливаються** з шаблонним рядком фіч з `default_feature_row.py`. Явні query-параметри `feat_*` мають **вищий пріоритет** за live-значення.

**Щогодинний UI:** нові файли тривог оновлюють зведення по всьому дню при кожному знімку; **кожна година в модалці** не будує окремий “live артефакт на ту годину”, якщо тільки модель не використовує ознаку `hour` для зсуву профілю.

**Ретренінг:** не виконується автоматично. Для нової моделі запускайте скрипти з `forecasting/` / notebooks, покладіть новий `.pkl` у `models/` або `forecasting/`, перезапустіть uvicorn.

---

## 📁 Repository Structure (скорочено)

```
python_SaaS/
├── data/
│   ├── local_live/          # runtime JSON (isw/, alerts/) — *.json у .gitignore
│   ├── processed/
│   └── raw/
├── data_receiver/
│   ├── alarm_daily.py              # щогодинний збір тривог → local_live/alerts
│   ├── isw_daily_scraper.py        # щоденний ISW → local_live/isw
│   ├── purge_local_live_nightly.py # щодобове очищення JSON о 23:59:59
│   └── …
├── forecasting/
│   ├── default_feature_row.py      # шаблон фіч для API
│   ├── local_live_features.py     # злітання live JSON → оверрайди
│   ├── prediction_service.py       # інференс, кеш моделі, hourly helper
│   ├── model_runtime.py
│   └── …
├── frontend/
│   ├── api/main.py                 # FastAPI + (опційно) static dist
│   ├── app.py                      # Streamlit
│   └── react-app/                  # Vite + React UI
├── models/                         # .pkl (частина ігнору Git за розміром)
├── .env.example
└── requirements.txt
```

---

## 📊 Data Sources & Dataset Overview

| Source | Type | Notes |
|--------|------|-------|
| [air-alarms.in.ua](https://air-alarms.in.ua/) | Historical CSV | Навчання / аналітика |
| [understandingwar.org](https://www.understandingwar.org/) | ISW HTML → JSON | Денний скрейпер у `local_live/isw` |
| [visualcrossing.com](https://www.visualcrossing.com/weather-api) | Weather API | Додаткові колектори |
| [api.ukrainealarm.com](https://api.ukrainealarm.com/) | Live alerts JSON | `alarm_daily.py` → `local_live/alerts` |

> ⚠️ **Data Notice:** історичні дані подій з air-alarms.in.ua — для **внутрішнього використання**; не публікувати.

---

## 🔬 Data Analysis & Preparation Strategy

(Без зміни суті проєкту) ISW і структуровані ряди перетворюються на матрицю ознак з лагами, ковзними статистиками тощо; таргет — бінарна подія в наступні 24 год. Деталі тренування описані в notebooks і скриптах у `forecasting/` та `eda/`.

---

## ⚙️ Setup & Installation

**Вимоги:** Python 3.10+, Node 18+ (лише для збірки React).

```bash
git clone https://github.com/exEKS/python_SaaS.git
cd python_SaaS
pip install -r requirements.txt

cp .env.example .env
# Мінімум для колектора тривог: UKRAINE_ALARM_TOKEN
```

Збірка фронтенду (якщо віддаєте `dist` через FastAPI або nginx):

```bash
cd frontend/react-app
npm ci
npm run build
```

Покладіть модель, наприклад `rf_compressed.pkl`, у `forecasting/` або `models/` (або задайте `WARWATCH_MODEL_DIR`).

---

## 🚀 Running Locally

**Термінал 1 — API (і статичний `dist`, якщо папка існує):**

```bash
cd python_SaaS
uvicorn frontend.api.main:app --reload --host 127.0.0.1 --port 8000
```

**Термінал 2 — React у dev (опційно, якщо не використовуєте тільки `dist`):**

```bash
cd python_SaaS/frontend/react-app
npm run dev
```

**Термінал 3 — Streamlit (опційно):**

```bash
streamlit run frontend/app.py
```

**Колектори (окремі довгоживучі процеси або systemd):**

```bash
python data_receiver/alarm_daily.py
python data_receiver/isw_daily_scraper.py
python data_receiver/purge_local_live_nightly.py
```

---

## 🌐 Production (nginx + EC2)

Типовий прод на малому інстансі (наприклад `t3.micro`):

1. Один процес **`uvicorn`** на `127.0.0.1:8000` (без `--reload`).
2. **nginx** віддає статику з `frontend/react-app/dist` і проксує **`/api/`** на `http://127.0.0.1:8000/` (або з rewrite — на бекенді дубльовані маршрути `/api/...`).
3. Користувачам не потрібно відкривати порт 8000 назовні — лише 80/443.
4. Колектори та purge запускайте окремими **systemd units** або **screen/tmux**, щоб вони переживали logout.

Рекомендація: обмежити паралелізм на клієнті вже реалізовано (batch + низький fallback); на сервері не запускайте зайвих копій uvicorn на один порт.

---

## 🔑 Environment Variables

Скопіюйте `.env.example` у `.env` і заповніть:

```env
# Ключі для зовнішніх API (збір даних / Streamlit)
VISUALCROSSING_API_KEY=
UKRAINE_ALARM_TOKEN=

# Модель
WARWATCH_MODEL_DIR=
WARWATCH_MODEL_ALARM=
WARWATCH_MODEL_EXPLOSION=
WARWATCH_MODEL_ARTILLERY=

# Локальні JSON колекторів (за замовч.: data/local_live/)
WARWATCH_LOCAL_DATA_DIR=
# 1/true — ігнорувати live JSON при формуванні фіч
WARWATCH_DISABLE_LOCAL_LIVE=

# Streamlit → API
WARWATCH_API_URL=http://127.0.0.1:8000
```

Також підтримується **`ALARM_API_KEY`** як синонім для `UKRAINE_ALARM_TOKEN` у `alarm_daily.py`.

---

## 👷 Team Responsibility List

| Team Member | Contribution |
|-------------|--------------|
| **Kovalov Evgen** | Inference script, full SaaS deployment (UI + API), prediction storage & retrieval, retraining pipeline setup, demo video, final report, frontend & backend |
| **Tavlui Sofiia** | Top model selection & motivation, model retraining, 24h prediction API endpoint, automated retraining script with historical + new data and model validation |
| **Snikhovska Vladyslava** | RandomForest model training & development, forecast update endpoint, README editing, final report & presentation, demo video |
| **Poliakov Kyrylo** | CatBoost model training & tuning, automated 24h regional forecasting via hourly Cron on AWS EC2, retraining pipeline with automated validation |
| **Barabanova Sofiia** | Decision Tree development & hyperparameter tuning, evaluation metrics & confusion matrix analysis, feature importance visualisation (Top-20), model improvement strategy |
