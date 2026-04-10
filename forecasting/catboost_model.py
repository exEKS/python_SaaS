import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import classification_report, confusion_matrix
from catboost import CatBoostClassifier

# 1. ЗАВАНТАЖЕННЯ ДАНИХ
file_path = 'features_new.csv'
print("Завантаження датасету...")
df = pd.read_csv(file_path, encoding='utf-8-sig')

# 2. ПІДГОТОВКА ДАНИХ
target_col = 'target_alarm_next_day'

if target_col not in df.columns:
    raise ValueError(f" Колонку '{target_col}' не знайдено у файлі!")

cols_to_drop = [target_col, 'date', 'date_dt', 'isw_text', 'reddit_text', 'unigrams', 'bigrams', 'city_address', 'region_city', 'day_conditions']
X = df.drop(columns=[col for col in cols_to_drop if col in df.columns])

# Залишаємо ТІЛЬКИ числові колонки
X = X.select_dtypes(include=['number', 'bool'])
y = df[target_col]

X = X.fillna(0)

print(f"Кількість фіч, які йдуть у модель: {len(X.columns)}")

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# 3. ТЮНІНГ ГІПЕРПАРАМЕТРІВ
print("Пошук найкращих параметрів (Grid Search)... ")

# allow_writing_files=False забороняє створювати конфліктні папки
cb = CatBoostClassifier(verbose=0, random_state=42, allow_writing_files=False)

param_grid = {
    'iterations': [100, 200],
    'depth': [4, 6],
    'learning_rate': [0.05, 0.1]
}

grid_search = GridSearchCV(estimator=cb, param_grid=param_grid, cv=3, scoring='f1', n_jobs=-1)
grid_search.fit(X_train, y_train)

best_model = grid_search.best_estimator_
print(f"Найкращі параметри CatBoost: {grid_search.best_params_}")

# 4. ОЦІНКА ТА ПІДГОТОВКА ДАНИХ ДЛЯ ГРАФІКІВ
print("\nОцінка на тестовій вибірці...")
y_pred = best_model.predict(X_test)

# Текстовий звіт
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# Дані для Confusion Matrix
conf_matrix = confusion_matrix(y_test, y_pred)

# Дані для Feature Importances
feature_importances = best_model.get_feature_importance()
fi_df = pd.DataFrame({
    'Feature': X.columns,
    'Importance': feature_importances
})
top_20_fi = fi_df.sort_values(by='Importance', ascending=False).head(20)

# 5. ВІЗУАЛІЗАЦІЯ
print("Малювання об'єднаного графіка...")

# Створюємо одне полотно (fig) з двома графіками (axes), 1 рядок, 2 колонки
fig, axes = plt.subplots(1, 2, figsize=(16, 7))

# --- Графік зліва (axes[0]): Confusion Matrix ---
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', cbar=False, ax=axes[0], annot_kws={"size": 14})
axes[0].set_title('Confusion Matrix - CatBoost', fontsize=16)
axes[0].set_xlabel('Predicted Label', fontsize=12)
axes[0].set_ylabel('True Label', fontsize=12)

# --- Графік справа (axes[1]): Top 20 Features ---
sns.barplot(x='Importance', y='Feature', data=top_20_fi, hue='Feature', palette='viridis', legend=False, ax=axes[1])
axes[1].set_title('Top 20 Features - CatBoost', fontsize=16)
axes[1].set_xlabel('Feature Importance', fontsize=12)
axes[1].set_ylabel('Feature', fontsize=12)

# Вирівнюємо та зберігаємо
plt.tight_layout()
combined_plot_filename = 'catboost.png'
plt.savefig(combined_plot_filename, dpi=300) # dpi=300 робить картинку високої якості
print(f"Обидва графіки збережено в один файл: '{combined_plot_filename}'")

# 6. ЗБЕРЕЖЕННЯ МОДЕЛІ
model_filename = 'catboost_model.pkl'
with open(model_filename, 'wb') as file:
    pickle.dump(best_model, file)
print(f"\nМодель успішно збережена у файл: {model_filename}")