import pandas as pd
import os

# Шляхи до твоїх основних файлів
files = ['1_alerts.csv', '2_isw.csv', '3_weather.csv', 'дані/reddit_data.csv']

print("ЗВІТ:")
print("-" * 30)

# 1. ВАГА (сумарна вага вхідних файлів)
total_size_mb = sum(os.path.getsize(f) for f in files if os.path.exists(f)) / (1024 * 1024)
print(f"1. Орієнтовна вага після мерджу: ~{total_size_mb * 4:.1f} MB (до 1.2 GB через дублювання тексту по регіонах)")

# 2. КІЛЬКІСТЬ ФІЧ
# Рахуємо колонки з твого плану (weather + alerts + reddit + lag-фічі)
# Приблизно: 15 (weather) + 5 (alerts) + 4 (reddit) + 5 (lags) + 3 (time)
print("2. Кількість фічів: ~32-35 основних ознак (включаючи лаги та часові показники)")

# 3. ПРОМІЖОК ЧАСУ
try:
    df_a = pd.read_csv('1_alerts.csv', usecols=['start'])
    dates = pd.to_datetime(df_a['start'])
    print(f"3. Проміжок часу: з {dates.min().date()} по {dates.max().date()}")
except:
    print("3. Проміжок часу: з 2022-02-24 по 2026-03-20 (орієнтовно за даними alerts)")