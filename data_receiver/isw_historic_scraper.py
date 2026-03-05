import requests
from bs4 import BeautifulSoup
import datetime
import time
import json
import os

START_DATE = datetime.datetime(2022, 2, 24)
END_DATE = datetime.datetime(2026, 3, 1)
OUTPUT_FILE = "isw_historical_data.json"

def get_isw_report(date):
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
    }
    
    date_str = date.strftime("%Y-%m-%d")


    month_str = date.strftime("%B").lower()
    day = str(date.day)
    year = str(date.year)
        
    urls = [
        f"https://www.understandingwar.org/research/russia-ukraine/russian-offensive-campaign-assessment-{month_str}-{day}-{year}",
        f"https://www.understandingwar.org/backgrounder/russian-offensive-campaign-assessment-{month_str}-{day}-{year}",
        f"https://www.understandingwar.org/backgrounder/russian-offensive-campaign-assessment-{month_str}-{day}"
    ]
    
    for url in urls:
        try:
            response = requests.get(url, headers=headers, timeout=10)
            if response.status_code == 200:
                soup = BeautifulSoup(response.text, 'html.parser')
                
                content = soup.find('div', class_='field-name-body') or soup.find('article')
                
                if content:
                    paragraphs = content.find_all('p')
                    text = " ".join([p.get_text(strip=True) for p in paragraphs])
                    return text, url
        except Exception as e:
            print(f"Error fetching {url}: {e}")
            
    return None, None

def crawl_historical():
    current_date = START_DATE
    data_records = []

    print("Starting historical data collection...")
    
    while current_date <= END_DATE:
        text, url = get_isw_report(current_date)
        
        if text:
            print(f"[SUCCESS] {current_date.strftime('%Y-%m-%d')}")
            data_records.append({
                "date": current_date.strftime("%Y-%m-%d"),
                "url": url,
                "text": text
            })
            
            with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
                json.dump(data_records, f, ensure_ascii=False, indent=4)
                
        else:
            print(f"[NOT FOUND] Report for {current_date.strftime('%Y-%m-%d')}")
            
        current_date += datetime.timedelta(days=1)
        time.sleep(1.5)

    print(f"\nScraping complete! Data safely stored in {OUTPUT_FILE}")

def patch_dataset():
    if not os.path.exists(OUTPUT_FILE):
        print(f"Error: Could not find {OUTPUT_FILE}. Please make sure the file exists.")
        return

    with open(OUTPUT_FILE, "r", encoding="utf-8") as f:
        try:
            data_records = json.load(f)
        except json.JSONDecodeError:
            print("Error: JSON file is corrupted or empty.")
            return

    existing_dates = {record["date"] for record in data_records}
    print(f"Loaded {len(existing_dates)} existing records from the file.")

    missing_dates = []
    current_date = START_DATE
    while current_date <= END_DATE:
        date_str = current_date.strftime("%Y-%m-%d")
        if date_str not in existing_dates:
            missing_dates.append(current_date)
        current_date += datetime.timedelta(days=1)

    if not missing_dates:
        print("Dataset is 100% complete! No missing dates found.")
        return

    print(f"Found {len(missing_dates)} missing dates. Starting patch process...\n")

    for target_date in missing_dates:
        date_str = target_date.strftime("%Y-%m-%d")
        print(f"Attempting to patch: {date_str}...", end=" ")
        
        text, url = get_isw_report(target_date)
        
        if text:
            print("SUCCESS")
            data_records.append({
                "date": date_str,
                "url": url,
                "text": text
            })
            
            data_records.sort(key=lambda x: x["date"])
            with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
                json.dump(data_records, f, ensure_ascii=False, indent=4)
        else:
            print("FAILED (Still cannot find report)")
            
        time.sleep(1.5)

    print(f"\nPatch complete! Data saved and sorted chronologically in {OUTPUT_FILE}")
