# WarWatch — React frontend

Замінює Streamlit-дашборд. Показує інтерактивну карту України з тепловою картою ймовірності тривог.

## Структура

```
react-app/
├── src/
│   ├── main.jsx          ← entry point
│   ├── index.css         ← глобальні стилі (темна тема)
│   ├── App.jsx           ← головний компонент, контроли
│   ├── UkraineMap.jsx    ← SVG-карта з тултіпами
│   ├── RegionDetail.jsx  ← панель деталей при кліку
│   ├── useAlarmData.js   ← хук: fetch до FastAPI
│   ├── colors.js         ← probToColor, riskInfo, fmtPct
│   └── regions.js        ← SVG path'и всіх 27 областей
├── index.html
├── vite.config.js        ← dev-сервер + проксі на FastAPI
└── package.json
```

## Запуск

```bash
# 1. Встанови залежності (один раз)
cd frontend/react-app
npm install

# 2. Запусти FastAPI в окремому терміналі (з кореня репо)
uvicorn frontend.api.main:app --reload --host 127.0.0.1 --port 8000

# 3. Запусти React dev-сервер
npm run dev
# → http://localhost:5173
```

Vite автоматично проксює `/api/*` → `http://127.0.0.1:8000/*`,
тому CORS не потрібен для локальної розробки.

## Збірка для продакшну

```bash
npm run build
# Результат у dist/ — статичні файли, можна роздавати через FastAPI або nginx
```

### Роздача через FastAPI (опціонально)

Додай в `frontend/api/main.py`:

```python
from fastapi.staticfiles import StaticFiles

# В кінці файлу, після всіх роутів:
app.mount("/", StaticFiles(directory="frontend/react-app/dist", html=True), name="static")
```

Тоді `uvicorn frontend.api.main:app` роздає і API, і фронтенд на одному порту.

## Як це працює

1. При натисканні "Оновити прогноз" `useAlarmData` паралельно робить `fetch` для кожної з ~24 областей → `/api/predict?region=Kharkiv&date=2025-06-01`
2. Результат `alarm_prob` (0–1) → колір через `probToColor`
3. Якщо API недоступне — показуються демо-дані щоб карта не була порожньою
4. Клік на область → `RegionDetail` з великим % і бейджем ризику
