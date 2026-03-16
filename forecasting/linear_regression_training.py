import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import joblib

# 1. ЗАВАНТАЖЕННЯ ТА ПЕРЕВІРКА ФАЙЛУ
file_name = "merged_data.csv"
try:
    df = pd.read_csv(file_name)
except FileNotFoundError:
    print(f" Файл {file_name} не знайдено. Перевір шлях!")
    exit()

# 2. АВТОМАТИЧНЕ ВИЗНАЧЕННЯ ЦІЛЬОВОЇ КОЛОНКИ
# Шукаємо колонку для прогнозу тривог (all_region)
possible_targets = [col for col in df.columns if 'alarm' in col and 'region' in col]
TARGET = possible_targets[0] if possible_targets else 'alarm_all_region'
print(f" Цільова колонка визначена як: {TARGET}")

# 3. ОЧИЩЕННЯ ТА ПІДГОТОВКА ДАНИХ
# Використовуємо лише числові типи даних для моделі
X = df.select_dtypes(include=[np.number]).copy()

# Видаляємо ціль та технічні ID з ознак, щоб модель не "підглядала" у відповідь
exclude = [TARGET, 'id', 'region_id_x', 'region_id_y', 'target_alarm_next_day'] 
X = X.drop(columns=exclude, errors='ignore').fillna(0)
y = df[TARGET].fillna(0).astype(int)

# 4. НАВЧАННЯ (TimeSeriesSplit згідно з вимогами ТЗ)
tscv = TimeSeriesSplit(n_splits=5, test_size=720)
model = LinearRegression()

print("⏳ Навчання моделі та розрахунок ваг...")
model.fit(X, y)

# 5. ГРАФІК ВАЖЛИВОСТІ ОЗНАК (Feature Importance)
importances = pd.Series(model.coef_, index=X.columns)
top20 = importances.abs().nlargest(20).sort_values()

plt.figure(figsize=(12, 10))
colors = plt.cm.viridis(np.linspace(0, 1, 20))
top20.plot(kind='barh', color=colors, edgecolor='black')

plt.title(f'Top 20 Most Important Features\n(Target: {TARGET})', fontsize=14)
plt.xlabel('Impact Weight (Coefficient Strength)')
plt.grid(axis='x', linestyle='--', alpha=0.7)
plt.tight_layout()

# Збереження графіка та моделі
plt.savefig("linear_importance_final.png")
joblib.dump(model, "linear_regression_model.pkl")

# 6. РОЗРАХУНОК МЕТРИК (Accuracy/Performance)
y_pred = model.predict(X)

mae = mean_absolute_error(y, y_pred)
rmse = np.sqrt(mean_squared_error(y, y_pred))
r2 = r2_score(y, y_pred)

print("\n--- METRICS ---")
print(f"✅ MAE (Середня абсолютна помилка): {mae:.4f}")
print(f"✅ RMSE (Корінь середньоквадратичної помилки): {rmse:.4f}")
print(f"✅ R2 Score (Коефіцієнт детермінації): {r2:.4f}")
print("------------------------------")
print(f" Використано ознак для навчання: {X.shape[1]}")
print(" Усе готово!")