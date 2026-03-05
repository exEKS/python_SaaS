import requests
from bs4 import BeautifulSoup
import datetime
import time
import json
import os

OUTPUT_FOLDER = "isw_data"
SCHEDULED_HOUR = 23
SCHEDULED_MINUTE = 51

if not os.path.exists(OUTPUT_FOLDER):
    os.makedirs(OUTPUT_FOLDER)
    print(f"Created new directory: {OUTPUT_FOLDER}")

def fetch_daily_report(target_date):
    
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
    }
    
    month_str = target_date.strftime("%B").lower()
    day = str(target_date.day)
    year = str(target_date.year)
    date_str = target_date.strftime('%Y-%m-%d')
    
    urls_to_try = [
        f"https://www.understandingwar.org/research/russia-ukraine/russian-offensive-campaign-assessment-{month_str}-{day}-{year}",
        f"https://www.understandingwar.org/backgrounder/russian-offensive-campaign-assessment-{month_str}-{day}-{year}",
        f"https://www.understandingwar.org/research/russia-ukraine/russian-offensive-campaign-assessment-{month_str}-{day}",
        f"https://www.understandingwar.org/backgrounder/russian-offensive-campaign-assessment-{month_str}-{day}"
    ]
    
    for url in urls_to_try:
        try:
            response = requests.get(url, headers=headers, timeout=10)
            if response.status_code == 200:
                soup = BeautifulSoup(response.text, 'html.parser')
                
                content = soup.find('div', class_='field-name-body') or soup.find('article')
                
                if content:
                    paragraphs = content.find_all('p')
                    full_text = " ".join([p.get_text(strip=True) for p in paragraphs if p.get_text(strip=True)])
                    
                    if full_text:
                        json_filename = os.path.join(OUTPUT_FOLDER, f"isw_data_{date_str}.json")
                        
                        daily_data = {
                            "date": date_str,
                            "url": url,
                            "text": full_text
                        }
                        
                        with open(json_filename, "w", encoding="utf-8") as f:
                            json.dump(daily_data, f, ensure_ascii=False, indent=4)
                            
                        print(f"[SUCCESS] Saved today's report to {json_filename}")
                        return True
        except Exception:
            pass 
            
    print(f"[FAILED] Could not find or download the report for {date_str}")
    return False

def get_seconds_until_next_run():
    now = datetime.datetime.now()
    target_time = now.replace(hour=SCHEDULED_HOUR, minute=SCHEDULED_MINUTE, second=0, microsecond=0)
    
    if target_time <= now:
        target_time += datetime.timedelta(days=1)
        
    return (target_time - now).total_seconds()

if __name__ == "__main__":
    print(f"Daily ISW Scraper activated. Scheduled to run every day at {SCHEDULED_HOUR:02d}:{SCHEDULED_MINUTE:02d}.")
    
    while True:
        seconds_to_wait = get_seconds_until_next_run()
        hours = int(seconds_to_wait // 3600)
        minutes = int((seconds_to_wait % 3600) // 60)
        
        print(f"Waiting {hours}h {minutes}m for the next scheduled run...")
        
        time.sleep(seconds_to_wait)
        
        today = datetime.datetime.now()
        print(f"\n--- Waking up to fetch report for {today.strftime('%Y-%m-%d')} ---")
        fetch_daily_report(today)
        
        time.sleep(120)
