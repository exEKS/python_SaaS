import json
import joblib
import pandas as pd
import numpy as np
from datetime import datetime
import os

# --- НАЛАШТУВАННЯ ---
MODEL_PATH = "model_randomforest.pkl"
WEATHER_DATA_FILE = "weather_history.json"
OUTPUT_FILE = "predictions.json"

# Мапінг: міста -> назви регіонів у моделі
CITY_TO_REGION = {
    "Lutsk": "Volyn",
    "Dnipro": "Dnipropetrovsk",
    "Uzhhorod": "Zakarpattia",
    "Kropyvnytskyi": "Kirovohrad",
    "Simferopol": "Crimea"
}

# ПОВНИЙ СПИСОК 57 ОЗНАК (у правильному порядку для моделі)
MANUAL_FEATURES = [
    'day_temp', 'day_humidity', 'day_wind_speed', 'day_precip', 'day_pressure',
    'day_wind_dir', 'day_cloud_cover', 'hour', 'day_of_year', 'day_of_week',
    'is_weekend', 'is_holiday',
    'region_Cherkasy', 'region_Chernihiv', 'region_Chernivtsi', 'region_Crimea',
    'region_Dnipropetrovsk', 'region_Donetsk', 'region_Ivano-Frankivsk',
    'region_Kharkiv', 'region_Kherson', 'region_Khmelnytskyi', 'region_Kyiv',
    'region_Kirovohrad', 'region_Luhansk', 'region_Lviv', 'region_Mykolaiv',
    'region_Odesa', 'region_Poltava', 'region_Rivne', 'region_Sumy',
    'region_Ternopil', 'region_Vinnytsia', 'region_Volyn', 'region_Zakarpattia',
    'region_Zaporizhzhia', 'region_Zhytomyr',
    'day_temp_10_day_avg', 'day_temp_10_day_max', 'day_temp_10_day_min', 'day_temp_10_day_std',
    'day_humidity_10_day_avg', 'day_humidity_10_day_max', 'day_humidity_10_day_min', 'day_humidity_10_day_std',
    'day_wind_speed_10_day_avg', 'day_wind_speed_10_day_max', 'day_wind_speed_10_day_min', 'day_wind_speed_10_day_std',
    'day_pressure_10_day_avg', 'day_pressure_10_day_max', 'day_pressure_10_day_min', 'day_pressure_10_day_std',
    'day_cloud_cover_10_day_avg', 'day_cloud_cover_10_day_max', 'day_cloud_cover_10_day_min',
    'day_cloud_cover_10_day_std'
]


def calculate_rolling_stats(history, city, current_data):
    """Витягує дані за 10 днів з історії та рахує середнє/std"""
    keys_map = {
        'temp': 'temp',
        'humidity': 'humidity',
        'wind_speed': 'windspeed',
        'pressure': 'sealevelpressure',
        'cloud_cover': 'cloudcover'
    }
    stats_results = {}

    # Сортуємо дати, щоб взяти останні 10 днів
    all_dates = sorted(history.keys())

    for feat_name, api_key in keys_map.items():
        vals = []
        for d in all_dates:
            v = history[d]['regions'].get(city, {}).get(api_key)
            if v is not None:
                vals.append(float(v))

        # Додаємо поточне значення до списку для розрахунку
        current_val = float(current_data.get(api_key, 0))
        vals.append(current_val)

        # Беремо останні 10 доступних точок
        recent = vals[-10:]

        stats_results[f'day_{feat_name}_10_day_avg'] = np.mean(recent)
        stats_results[f'day_{feat_name}_10_day_std'] = np.std(recent)
        stats_results[f'day_{feat_name}_10_day_max'] = np.max(recent)
        stats_results[f'day_{feat_name}_10_day_min'] = np.min(recent)

    return stats_results


def run_prediction():
    print("Завантаження моделі RandomForest...")
    if not os.path.exists(MODEL_PATH):
        print(f"Файл моделі {MODEL_PATH} не знайдено!")
        return

    model = joblib.load(MODEL_PATH)

    if not os.path.exists(WEATHER_DATA_FILE):
        print(f"Файл історії {WEATHER_DATA_FILE} не знайдено!")
        return

    with open(WEATHER_DATA_FILE, 'r', encoding='utf-8') as f:
        history = json.load(f)

    # Визначаємо останню дату в історії для поточного прогнозу
    latest_date = sorted(history.keys())[-1]
    weather_data = history[latest_date]['regions']

    regions_forecast = {}
    current_time = datetime.now()

    print(f"Розрахунок для {len(weather_data)} міст за дату {latest_date}...")

    for city, data in weather_data.items():
        # Створюємо вхідний вектор (57 колонок нулів)
        input_df = pd.DataFrame(0.0, index=[0], columns=MANUAL_FEATURES)

        # 1. Заповнюємо поточні показники
        input_df['day_temp'] = float(data.get('temp', 0))
        input_df['day_humidity'] = float(data.get('humidity', 0))
        input_df['day_wind_speed'] = float(data.get('windspeed', 0))
        input_df['day_precip'] = float(data.get('precip', 0))
        input_df['day_wind_dir'] = float(data.get('winddir', 0))
        input_df['day_pressure'] = float(data.get('sealevelpressure', 1013))
        input_df['day_cloud_cover'] = float(data.get('cloudcover', 0))

        # 2. Розрахунок реальної статистики за 10 днів
        stats = calculate_rolling_stats(history, city, data)
        for stat_col, val in stats.items():
            if stat_col in input_df.columns:
                input_df[stat_col] = val

        # 3. Часові ознаки
        input_df['hour'] = current_time.hour
        input_df['day_of_year'] = current_time.timetuple().tm_yday
        input_df['day_of_week'] = current_time.weekday()
        input_df['is_weekend'] = 1.0 if current_time.weekday() >= 5 else 0.0

        # 4. Регіон
        region_name = CITY_TO_REGION.get(city, city)
        region_col = f"region_{region_name}"
        if region_col in input_df.columns:
            input_df[region_col] = 1.0

        # 5. Прогноз
        probabilities = model.predict_proba(input_df.values)[0]
        prob_true = probabilities[1]

        # ПОРІГ 0.58
        threshold = 0.58
        prediction = prob_true > threshold

        print(f"📍 {city:15} | Prob: {prob_true:.2f} | Alert: {str(prediction).upper()}")

        # Структура для фронтенду
        regions_forecast[city] = {f"{h:02d}:00": bool(prediction) for h in range(24)}

    # 6. Збереження результатів
    final_output = {
        "last_prediction_time": current_time.isoformat() + "Z",
        "regions_forecast": regions_forecast
    }

    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(final_output, f, indent=4, ensure_ascii=False)

    print(f"🚀 Прогнози успішно збережено у {OUTPUT_FILE}!")


if __name__ == "__main__":
    run_prediction()