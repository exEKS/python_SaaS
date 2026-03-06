import os
import requests

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
            
            for region in data[:5]: 
                name = region.get('regionName', 'Невідомо')
                status = "🔴 ТРИВОГА" if region.get('activeAlerts') else "🟢 Спокійно"
                print(f"{name}: {status}")
        else:
            print(f"Помилка API: {response.status_code}")
            
    except Exception as e:
        print(f"Сталася помилка при виконанні запиту: {e}")

if __name__ == "__main__":
    fetch_ukraine_alarms()
