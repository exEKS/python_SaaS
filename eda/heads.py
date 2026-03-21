import pandas as pd
import os

# Словник з правильними шляхами (враховуючи папку 'дані')
files = {
    "Weather": "3_weather.csv",
    "Alerts": "1_alerts.csv",
    "ISW": "2_isw.csv",
    "Reddit": "дані/reddit_data.csv"
}

print("\n" + "="*60)
print("📝 ЗВІТ ПО СТРУКТУРІ ТАБЛИЦЬ (БЕЗ МЕРДЖУ)")
print("="*60)

for name, path in files.items():
    if os.path.exists(path):
        try:
            # Читаємо тільки заголовки для швидкості
            df = pd.read_csv(path, nrows=0)
            print(f"\n📄 ФАЙЛ: {path} ({name})")
            print(f"Кількість колонок: {len(df.columns)}")
            print("-" * 30)
            # Список колонок у рядок для економії місця в терміналі
            print(df.columns.tolist())
            print("-" * 60)
        except Exception as e:
            print(f"⚠️ Не вдалося прочитати {name}: {e}")
    else:
        print(f"❌ Файл {name} не знайдено за шляхом: {path}")

print("\n✅ ГОТОВО! Просто скопіюй цей список колонок для Claude.")