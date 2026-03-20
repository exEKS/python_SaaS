import pandas as pd
import os
import ast


def run_warwatch_engineering(input_file):
    if not os.path.exists(input_file):
        print(f" Помилка: Файл {input_file} не знайдено!")
        return

    # 1. Завантаження даних
    df = pd.read_csv(input_file)
    print(f" Завантажено рядків: {len(df)}")
    print(f" Наявні колонки: {df.columns.tolist()}")

    try:
        # Підготовка дати та сортування
        df['date'] = pd.to_datetime(df['date'])
        # Важливо: якщо у вас багато регіонів, сортуємо за регіоном та датою
        if 'region_city' in df.columns:
            df = df.sort_values(by=['region_city', 'date'])
        else:
            df = df.sort_values(by=['date'])

        print(" Розрахунок математичних фіч...")

        # --- ФІЧІ ТРИВОГ (якщо колонки збереглися) ---
        if 'alarm_all_region' in df.columns:
            df['total_regions_with_alarm_day'] = df.groupby('date')['alarm_all_region'].transform('sum')

        if 'alarm_count' in df.columns:
            # Створюємо лаг (значення за вчора)
            df['alarms_yesterday'] = df.groupby('region_city')['alarm_count'].shift(1).fillna(
                0) if 'region_city' in df.columns else df['alarm_count'].shift(1).fillna(0)

        if 'alarm_total_duration_min' in df.columns:
            # Ковзне середнє за тиждень
            df['duration_7d_avg'] = df.groupby('region_city')['alarm_total_duration_min'].transform(
                lambda x: x.rolling(window=7, min_periods=1).mean()) if 'region_city' in df.columns else df[
                'alarm_total_duration_min'].rolling(window=7, min_periods=1).mean()

        # --- ПОГОДНІ ФІЧІ ---
        if 'day_temp' in df.columns:
            df['temp_change'] = df['day_temp'].diff().fillna(0)

        if 'day_humidity' in df.columns:
            df['humidity_yesterday'] = df['day_humidity'].shift(1).fillna(df['day_humidity'].mean())

        # --- ТЕКСТОВІ ФІЧІ (на основі токенізації) ---
        def get_token_count(val):
            try:
                if pd.isna(val): return 0
                tokens = ast.literal_eval(val) if isinstance(val, str) and val.startswith('[') else val
                return len(tokens) if isinstance(tokens, list) else 0
            except:
                return 0

        if 'unigrams' in df.columns:
            df['text_intensity_index'] = df['unigrams'].apply(get_token_count)
            print(" Розраховано індекс інтенсивності тексту.")

        # 2. ЗБЕРЕЖЕННЯ
        output_path = "features.csv"
        df.to_csv(output_path, index=False, encoding='utf-8-sig')

        print(f" Успіх! Файл збережено: {output_path}")
        print("-" * 40)

    except Exception as e:
        print(f" Виникла помилка: {e}")


if __name__ == "__main__":
    input_data = "tokenizer_data.csv"
    run_warwatch_engineering(input_data)