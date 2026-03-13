from dotenv import load_dotenv
load_dotenv()
import os
import requests
import pandas as pd

API_KEY = os.getenv("ALARM_API_KEY")

def fetch_ukraine_alarms():
    if not API_KEY:
        print("Помилка: Ключ ALARM_API_KEY не знайдено в системі!")
        return

    url = "https://api.ukrainealarm.com/api/v3/alerts"
    headers = {"Authorization": API_KEY}
    
    try:
        response = requests.get(url, headers=headers)
        
        if response.status_code == 200:
            data = response.json()
            print(f"--- Успішно отримано дані для {len(data)} регіонів ---")
            
            alarms_list = []
            for region in data:
                alarms_list.append({
                    'region': region.get('regionName'),
                    'active_alerts': 1 if region.get('activeAlerts') else 0,
                    'datetime': pd.Timestamp.now().floor('h') # Для стикування з погодою
                })
            
            # ЗБЕРЕЖЕННЯ (те, що ми шукали!)
            df = pd.DataFrame(alarms_list)
            df.to_csv("alarms.csv", index=False)
            print("--- ФАЙЛ alarms.csv УСПІШНО СТВОРЕНО! ---")
            
        else:
            print(f"Помилка API: {response.status_code}")
            
    except Exception as e:
        print(f"Сталася помилка: {e}")

if __name__ == "__main__":
    fetch_ukraine_alarms()