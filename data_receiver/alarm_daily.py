import sys
from pathlib import Path

import requests
import time
import json
import os
from datetime import datetime, timedelta

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

try:
    from dotenv import load_dotenv

    load_dotenv(ROOT / ".env")
except ImportError:
    pass

from forecasting.local_live_features import alerts_dir

BASE_URL = "https://api.ukrainealarm.com/api/v3/alerts"
API_KEY = (os.getenv("UKRAINE_ALARM_TOKEN") or os.getenv("ALARM_API_KEY") or "").strip()

def fetch_and_save_alerts():
    if not API_KEY:
        print("Помилка: задай UKRAINE_ALARM_TOKEN або ALARM_API_KEY у .env")
        return
    headers = {
        "Authorization": API_KEY,
        "accept": "application/json"
    }
    now = datetime.now()
    timestamp = now.strftime("%Y-%m-%d_%H-%M-%S")
    
    try:
        print(f"[{timestamp}] Виконую запит...")
        response = requests.get(BASE_URL, headers=headers, timeout=30)
        
        if response.status_code == 200:
            data = response.json()
            file_name = f"alerts_{timestamp}.json"
            file_path = alerts_dir() / file_name
            
            with open(file_path, "w", encoding="utf-8") as f:
                json.dump(data, f, ensure_ascii=False, indent=4)
            print(f"Дані збережено: {file_name}")
        else:
            print(f"Помилка API: {response.status_code}")
    except Exception as e:
        print(f"Помилка: {e}")

def wait_until_next_hour():
    now = datetime.now()
    # Вираховуємо початок наступної години
    next_hour = (now + timedelta(hours=1)).replace(minute=0, second=0, microsecond=0)
    # Скільки секунд залишилося
    wait_seconds = (next_hour - now).total_seconds()
    
    print(f"Спимо {int(wait_seconds // 60)} хв і {int(wait_seconds % 60)} сек до {next_hour.strftime('%H:%M:%S')}...")
    time.sleep(wait_seconds)

if __name__ == "__main__":
    print("МОНІТОРИНГ СИНХРОНІЗОВАНО З ГОДИННИКОМ")
    
    # Перший запуск одразу
    fetch_and_save_alerts()
    
    while True:
        wait_until_next_hour()
        fetch_and_save_alerts()