import os
import requests
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
        
        print(f"\n--- WEATHER FORECAST FOR {CITY.upper()} (NEXT 24H) ---")
        for day in data.get('days', []):
            for hr in day.get('hours', []):
                dt = datetime.fromtimestamp(hr['datetimeEpoch'])
                if now <= dt <= limit:
                    print(f"{dt.strftime('%H:%M')} | {hr['temp']}°C | {hr['conditions']}")
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    get_24h_forecast()
