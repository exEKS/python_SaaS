import requests
import json
import os
from datetime import datetime
from dotenv import load_dotenv

# --- ЛОГІКА ШЛЯХІВ ДЛЯ CRON ---
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)

# Завантажуємо .env
load_dotenv(os.path.join(PROJECT_ROOT, ".env"))

# API Ключ
API_KEY = os.getenv("WEATHER_API")

DATA_FOLDER = os.path.join(PROJECT_ROOT, "data", "local_live")
OUTPUT_FILE = os.path.join(DATA_FOLDER, "weather_history.json")

REGIONS = [
    "Kyiv", "Vinnytsia", "Lutsk", "Dnipro", "Donetsk", "Zhytomyr",
    "Uzhhorod", "Zaporizhzhia", "Ivano-Frankivsk", "Kropyvnytskyi",
    "Luhansk", "Lviv", "Mykolaiv", "Odesa", "Poltava", "Rivne",
    "Sumy", "Ternopil", "Kharkiv", "Kherson", "Khmelnytskyi",
    "Cherkasy", "Chernivtsi", "Chernihiv", "Simferopol"
]


def collect_today_weather():
    os.makedirs(DATA_FOLDER, exist_ok=True)

    current_date = datetime.now().strftime("%Y-%m-%d")
    current_time = datetime.now().strftime("%H:%M:%S")
    daily_data = {}

    print(f"[{current_time}] Початок збору даних за {current_date}...")

    for city in REGIONS:
        url = f"https://weather.visualcrossing.com/VisualCrossingWebServices/rest/services/timeline/{city}/today?key={API_KEY}&unitGroup=metric&include=hours,days"

        try:
            response = requests.get(url, timeout=15)
            if response.status_code == 200:
                data = response.json()
                day = data['days'][0]

                daily_data[city] = {
                    "temp": day.get('temp'),
                    "humidity": day.get('humidity'),
                    "windspeed": day.get('windspeed'),
                    "precip": day.get('precip'),
                    "winddir": day.get('winddir'),
                    "sealevelpressure": day.get('pressure'),
                    "cloudcover": day.get('cloudcover'),
                    "hours": day.get('hours', [])  # Масив на 24 години
                }
                print(f"  {city:15} | OK")
            else:
                print(f"  {city:15} | Помилка {response.status_code}")
        except Exception as e:
            print(f"  {city:15} | Критична помилка: {e}")

    if daily_data:
        # Формуємо JSON ТІЛЬКИ з сьогоднішнім днем
        final_json = {
            current_date: {
                "updated_at": current_time,
                "regions": daily_data
            }
        }

        # Перезаписуємо файл (режим 'w')
        with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
            json.dump(final_json, f, ensure_ascii=False, indent=4)

        print(f"\nУспіх! Дані за {current_date} збережено у: {OUTPUT_FILE}")
    else:
        print("\nПомилка: Дані не зібрано.")


if __name__ == "__main__":
    collect_today_weather()
