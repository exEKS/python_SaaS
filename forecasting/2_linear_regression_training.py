import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import Ridge
from sklearn.model_selection import TimeSeriesSplit, GridSearchCV
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import joblib

# 1. ЗАВАНТАЖЕННЯ
file_name = "features_final.csv"
try:
    df = pd.read_csv(file_name)
except FileNotFoundError:
    print(f" Файл {file_name} не знайдено!")
    exit()

# 2. ПІДГОТОВКА ДАНИХ
TARGET = 'target_alarm_next_day'

# Видаляємо нечислові колонки та таргет
X = df.select_dtypes(include=[np.number]).drop(columns=[TARGET], errors='ignore').fillna(0)
y = df[TARGET].fillna(0)

# 3. TUNING & TRAINING (TimeSeriesSplit)
print("⏳ Починаю тюнінг та навчання (Ridge Regression)...")

tscv = TimeSeriesSplit(n_splits=5)
ridge = Ridge()

# Налаштування гіперпараметрів (alpha - сила регуляризації)
param_grid = {'alpha': [0.1, 1.0, 10.0, 100.0]}
grid_search = GridSearchCV(ridge, param_grid, cv=tscv, scoring='neg_mean_squared_error')
grid_search.fit(X, y)

model = grid_search.best_estimator_
print(f" Найкращі параметри: {grid_search.best_params_}")

# 4. МЕТРИКИ
y_pred = model.predict(X)
mae = mean_absolute_error(y, y_pred)
rmse = np.sqrt(mean_squared_error(y, y_pred))
r2 = r2_score(y, y_pred)

print(f"\n--- МЕТРИКИ ---")
print(f"MAE: {mae:.4f} | RMSE: {rmse:.4f} | R2: {r2:.4f}")

# 5. ВАЖЛИВІСТЬ ОЗНАК (Top 20)
importances = pd.Series(model.coef_, index=X.columns)
top20 = importances.abs().nlargest(20).sort_values()

plt.figure(figsize=(12, 10))
colors = plt.cm.plasma(np.linspace(0, 1, 20)) # Змінив палітру для різноманіття
top20.plot(kind='barh', color=colors, edgecolor='black')

plt.title(f'Top 20 Most Important Features\n(Linear/Ridge Model)', fontsize=14)
plt.xlabel('Impact Weight')
plt.grid(axis='x', linestyle='--', alpha=0.7)
plt.tight_layout()

# 6. ЗБЕРЕЖЕННЯ
plt.savefig("2_linear_importance_final.png")
joblib.dump(model, "2_linear_regression_model.pkl")
print("\n Модель та графік оновлено!")