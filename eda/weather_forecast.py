from dotenv import load_dotenv
load_dotenv()
import os
import requests
import pandas as pd  # Додали це
from datetime import datetime, timedelta

API_KEY = os.getenv("WEATHER_API_KEY")
CITY = "Kyiv"

def get_24h_forecast():
    if not API_KEY:
        print("Error: WEATHER_API_KEY not found!")
        return

    url = f"https://weather.visualcrossing.com/VisualCrossingWebServices/rest/services/timeline/{CITY}?unitGroup=metric&key={API_KEY}&contentType=json"
    
    try:
        response = requests.get(url)
        data = response.json()
        now = datetime.now()
        limit = now + timedelta(hours=24)
        
        forecast_list = [] # Створюємо список для збереження
        
        print(f"\n--- WEATHER FORECAST FOR {CITY.upper()} (NEXT 24H) ---")
        for day in data.get('days', []):
            for hr in day.get('hours', []):
                dt = datetime.fromtimestamp(hr['datetimeEpoch'])
                if now <= dt <= limit:
                    # Додаємо дані в список
                    forecast_list.append({
                        'datetime': dt,
                        'temp': hr['temp'],
                        'conditions': hr['conditions']
                    })
                    print(f"{dt.strftime('%H:%M')} | {hr['temp']}°C | {hr['conditions']}")
        
        # ЗБЕРЕЖЕННЯ У ФАЙЛ (ось те, чого не вистачало)
        df = pd.DataFrame(forecast_list)
        df.to_csv("weather_v2.csv", index=False)
        print("\n--- ФАЙЛ weather_v2.csv УСПІШНО СТВОРЕНО! ---")

    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    get_24h_forecast()