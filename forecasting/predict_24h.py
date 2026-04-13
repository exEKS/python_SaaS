import json
import pickle
import pandas as pd
from datetime import datetime

# --- НАЛАШТУВАННЯ ---
MODEL_PATH = "catboost_model_new.pkl"
WEATHER_DATA_FILE = "weather_history.json"
OUTPUT_FILE = "predictions.json"


def run_prediction():
    # 1. ЗАВАНТАЖЕННЯ МОДЕЛІ
    print("Завантаження моделі...")
    try:
        with open(MODEL_PATH, 'rb') as f:
            model = pickle.load(f)
        # Отримуємо назви фіч, на яких вчилася модель
        model_features = model.feature_names_
    except Exception as e:
        print(f"Помилка завантаження моделі: {e}")
        return

    # 2. ЧИТАННЯ ІСТОРІЇ ПОГОДИ
    print(f"📂 Пошук останніх даних у {WEATHER_DATA_FILE}...")
    try:
        with open(WEATHER_DATA_FILE, 'r', encoding='utf-8') as f:
            history = json.load(f)

        if not history:
            print("❌ Файл історії порожній.")
            return

        # Беремо останню доступну дату
        dates = sorted(history.keys())
        latest_date = dates[-1]
        weather_data = history[latest_date]['regions']
        print(f"✅ Використовуємо дані за: {latest_date}")
    except Exception as e:
        print(f"❌ Помилка читаться файлів: {e}")
        return

    regions_forecast = {}
    current_time = datetime.now().isoformat() + "Z"

    # 3. ГЕНЕРАЦІЯ ПРОГНОЗІВ
    print("🔮 Розрахунок прогнозів для регіонів...")
    for city, data in weather_data.items():
        # Створюємо порожній рядок з правильними назвами колонок
        input_df = pd.DataFrame(index=[0], columns=model_features)

        # --- ВИПРАВЛЕННЯ ТИПІВ ДАНИХ ---

        # 1. Текстові колонки (ОБОВ'ЯЗКОВО мають бути рядками)
        text_cols = ['unigrams', 'bigrams']
        for col in text_cols:
            if col in input_df.columns:
                input_df[col] = ""  # Поки порожньо, але тип — string

        # 2. Категоріальна колонка
        if 'day_conditions' in input_df.columns:
            input_df['day_conditions'] = str(data.get('conditions', 'N/A'))

        # 3. Числові колонки
        # Заповнюємо дані з JSON, якщо вони там є, інакше ставимо 0
        mapping = {
            'day_temp': data.get('temp'),
            'day_humidity': data.get('humidity'),
            'temp_change': 0.0,  # Ці дані з'являться пізніше в історії
            'humidity_yesterday': 0.0,  # Ці дані з'являться пізніше в історії
            'text_intensity_index': 0  # Це додасться після скрапера новин
        }

        for col, value in mapping.items():
            if col in input_df.columns:
                input_df[col] = float(value) if value is not None else 0.0

        # --- САМ ПРОГНОЗ ---
        try:
            # Отримуємо прогноз (0 або 1)
            prediction = model.predict(input_df)[0]

            # Якщо модель видає ймовірності, можна додати поріг:
            # prob = model.predict_proba(input_df)[0][1]
            # prediction = 1 if prob > 0.3 else 0

            # Формуємо структуру на 24 години
            hourly_forecast = {f"{h:02d}:00": bool(prediction) for h in range(24)}
            regions_forecast[city] = hourly_forecast

        except Exception as e:
            print(f"⚠️ Помилка прогнозу для {city}: {e}")

    # 4. ЗБЕРЕЖЕННЯ РЕЗУЛЬТАТУ
    final_output = {
        "last_model_train_time": "2026-04-13T21:00:00Z",
        "last_prediction_time": current_time,
        "regions_forecast": regions_forecast
    }

    try:
        with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
            json.dump(final_output, f, indent=4, ensure_ascii=False)
        print(f"🚀 Прогнози успішно оновлено! Файл: {OUTPUT_FILE}")
    except Exception as e:
        print(f"❌ Помилка запису JSON: {e}")


if __name__ == "__main__":
    run_prediction()