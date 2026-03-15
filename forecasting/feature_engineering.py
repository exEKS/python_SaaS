import pandas as pd
import os


def check_and_run_engineering(input_file):
    if not os.path.exists(input_file):
        print(f"Помилка: Файл {input_file} не знайдено!")
        return

    # 1. ДІАГНОСТИКА (інтегрований check_columns)
    df = pd.read_csv(input_file)
    print("\n" + "=" * 40)
    print("ДІАГНОСТИКА СТРУКТУРИ ДАТАФРЕЙМУ")
    print("=" * 40)
    for i, col in enumerate(df.columns):
        print(f"{i}. {col}")
    print("=" * 40 + "\n")

    # 2. ПІДГОТОВКА ДАНИХ
    # Використовуємо твої назви стовпців (date, region_city тощо)
    try:
        df['date'] = pd.to_datetime(df['date'])
        df = df.sort_values(by=['region_city', 'date'])

        print("Запуск генерації фіч ...")

        # --- Фіча 1: Загальна кількість областей з тривогою в цей день ---
        df['total_regions_with_alarm_day'] = df.groupby('date')['alarm_all_region'].transform('sum')

        # --- Фіча 2: Кількість тривог у регіоні за минулу добу ---
        # Оскільки дані денні, shift(1) бере значення за попередні 24 години
        df['alarms_yesterday'] = df.groupby('region_city')['alarm_count'].shift(1).fillna(0)

        # --- Фіча 3: Brainstorm (Тренд тривалості) ---
        # Середня тривалість за останні 7 днів показує накопичену інтенсивність
        df['duration_7d_avg'] = df.groupby('region_city')['alarm_total_duration_min'].transform(
            lambda x: x.rolling(window=7, min_periods=1).mean()
        )

        # 3. ЗБЕРЕЖЕННЯ
        output_path = "features.csv"
        df.to_csv(output_path, index=False)

        print(f"Успіх! Розрахунки завершені.")
        print(f"Файл збережено як: {output_path}")
        print("-" * 40)

    except KeyError as e:
        print(f"Помилка: Не знайдено стовпець {e}")
        print("Перевірте назви колонок у діагностичному списку вище.")


if __name__ == "__main__":
    run_file = "tokenized_data.csv"
    check_and_run_engineering(run_file)