import requests
import json
import os
from datetime import datetime

# --- НАЛАШТУВАННЯ ---
API_KEY = "A7QPDSSGWS3E44DYUR6MQEVDU"
OUTPUT_FILE = "weather_history.json"

REGIONS = [
    "Kyiv", "Vinnytsia", "Lutsk", "Dnipro", "Donetsk", "Zhytomyr",
    "Uzhhorod", "Zaporizhzhia", "Ivano-Frankivsk", "Kropyvnytskyi",
    "Luhansk", "Lviv", "Mykolaiv", "Odesa", "Poltava", "Rivne",
    "Sumy", "Ternopil", "Kharkiv", "Kherson", "Khmelnytskyi",
    "Cherkasy", "Chernivtsi", "Chernihiv", "Simferopol"
]


def update_weather_history():
    # 1. Завантажуємо існуючу історію, якщо файл є
    if os.path.exists(OUTPUT_FILE):
        with open(OUTPUT_FILE, "r", encoding="utf-8") as f:
            try:
                history = json.load(f)
            except json.JSONDecodeError:
                history = {}
    else:
        history = {}

    current_date = datetime.now().strftime("%Y-%m-%d")
    current_time = datetime.now().strftime("%H:%M:%S")

    # Якщо за сьогодні дані вже є, можемо або пропустити, або оновити
    # Для історії зазвичай достатньо одного запису на день
    daily_data = {}

    print(f"Збір даних за {current_date} для {len(REGIONS)} регіонів...")

    for city in REGIONS:
        url = f"https://weather.visualcrossing.com/VisualCrossingWebServices/rest/services/timeline/{city}/today?key={API_KEY}&unitGroup=metric&include=days"

        try:
            response = requests.get(url)
            if response.status_code == 200:
                data = response.json()
                if 'days' in data and len(data['days']) > 0:
                    day = data['days'][0]
                    daily_data[city] = {
                        "temp": day.get('temp'),
                        "humidity": day.get('humidity'),
                        "conditions": day.get('conditions', "N/A"),
                        "description": day.get('description', "N/A")
                    }
                    print(f"{city} — ОК")
            else:
                print(f"{city} — помилка {response.status_code}")
        except Exception as e:
            print(f"Помилка для {city}: {e}")

    # 2. Додаємо нові дані в історію під ключем дати
    if daily_data:
        history[current_date] = {
            "updated_at": current_time,
            "regions": daily_data
        }

        # 3. Зберігаємо оновлену історію
        with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
            json.dump(history, f, ensure_ascii=False, indent=4)

        print(f"\nІсторію оновлено! Тепер у файлі даних за стільки днів: {len(history)}")
    else:
        print("\nНе вдалося зібрати дані за сьогодні.")


if __name__ == "__main__":
    update_weather_history()