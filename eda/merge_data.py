import pandas as pd
import json
import os
import glob

def finish_job():
    input_folder = 'Дані'
    print("Оновлюю дані: об'єдную звіти ISW та додаю регіони...")

    try:
        # 1. ISW: Збираємо всі файли isw_data_*.json
        isw_files = glob.glob(os.path.join(input_folder, 'isw_data_*.json'))
        main_isw = os.path.join(input_folder, 'isw_historical_data.json')
        if os.path.exists(main_isw):
            isw_files.append(main_isw)
        
        all_isw_data = []
        for file_path in isw_files:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                if isinstance(data, list):
                    all_isw_data.extend(data)
                else:
                    all_isw_data.append(data)
        
        # Створюємо DataFrame
        df_isw = pd.DataFrame(all_isw_data).drop_duplicates()
        
        # ВИДАЛЯЄМО URL ПЕРЕД ЗБЕРЕЖЕННЯМ
        if 'url' in df_isw.columns:
            df_isw = df_isw.drop(columns=['url'])
            
        df_isw.to_csv('2_isw.csv', index=False)
        print(f" 2_isw.csv готовий (об'єднано {len(isw_files)} файлів)")

        # 2. ALERTS + REGIONS
        regions_df = pd.read_csv(os.path.join(input_folder, 'regions.csv'))
        alerts_df = pd.read_csv(os.path.join(input_folder, 'alarms-240222-010325.csv'), sep=';')
        alerts_final = pd.merge(alerts_df, regions_df, left_on='region_city', right_on='region', how='left')
        alerts_final.to_csv('1_alerts.csv', index=False)
        print(" 1_alerts.csv оновлено")

        # 3. WEATHER
        weather_path = os.path.join(input_folder, 'all_weather_by_hour_2023-2025_v1.csv')
        pd.read_csv(weather_path).to_csv('3_weather.csv', index=False)
        print("3_weather.csv готовий")

        # 4. TELEGRAM
        tg_path = os.path.join(input_folder, 'telegram_data.json')
        with open(tg_path, 'r', encoding='utf-8') as f:
            pd.DataFrame(json.load(f)).to_csv('4_telegram.csv', index=False)
        print("4_telegram.csv готовий")

        print("\n Файли готові до відправки.")

    except Exception as e:
        print(f"Помилка: {e}")

if __name__ == "__main__":
    finish_job()
