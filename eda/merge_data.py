import pandas as pd

# 1. Читаємо погоду (тепер без data/raw/)
weather_df = pd.read_csv('weather_v2.csv')
print(f"Weather Data Shape: {weather_df.shape}")

# 2. Читаємо тривоги (тепер без data/raw/)
alarms_df = pd.read_csv('alarms.csv')
print(f"Alarms Data Shape: {alarms_df.shape}")

# 3. Об'єднуємо
# ПРИМІТКА: Якщо у файлі погоди немає колонки 'region', 
# спробуй об'єднати тільки по 'datetime'
try:
    merged_df = pd.merge(weather_df, alarms_df, on=['datetime', 'region'])
except KeyError:
    merged_df = pd.merge(weather_df, alarms_df, on=['datetime'])

print(f"Merged Data Shape: {merged_df.shape}")

# Зберігаємо результат
merged_df.to_csv("final_merged_data.csv", index=False)
print("Все готово! Файл final_merged_data.csv створено.")