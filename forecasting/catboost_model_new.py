import pandas as pd
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
from catboost import CatBoostClassifier, Pool
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix

# 1. ЗАВАНТАЖЕННЯ ТА ПІДГОТОВКА ДАНИХ
df = pd.read_csv('features.csv')

# Очищення текстових колонок (прибираємо дужки та лапки, щоб CatBoost бачив чисті слова)
for col in ['unigrams', 'bigrams']:
    df[col] = df[col].apply(lambda x: str(x).replace('[', '').replace(']', '').replace("'", "").replace(',', ''))

# Визначаємо колонки
target = 'target_alarm_next_day'
text_features = ['unigrams', 'bigrams']
cat_features = ['day_conditions']
num_features = ['day_temp', 'day_humidity', 'temp_change', 'humidity_yesterday', 'text_intensity_index']

df[target] = df[target].astype(int)

# 2. ПОДІЛ НА ВИБІРКИ
X = df[text_features + cat_features + num_features]
y = df[target]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Створюємо Pool (спеціальний формат CatBoost для швидкості)
train_pool = Pool(data=X_train, label=y_train, cat_features=cat_features, text_features=text_features)
test_pool = Pool(data=X_test, label=y_test, cat_features=cat_features, text_features=text_features)

# 3. НАЛАШТУВАННЯ ТА НАВЧАННЯ МОДЕЛІ
model = CatBoostClassifier(
    iterations=1500,
    learning_rate=0.03,
    depth=6,
    auto_class_weights='Balanced', # Щоб модель бачила тривоги, а не тільки спокійні днi
    eval_metric='F1',
    verbose=100
)

print("Починаю навчання моделі...")
model.fit(train_pool, eval_set=test_pool, early_stopping_rounds=100)

# 4. РОЗРАХУНОК МЕТРИК ДЛЯ ГРАФІКІВ
y_pred = model.predict(X_test)
conf_matrix = confusion_matrix(y_test, y_pred)

# Отримуємо важливість ознак
feature_importance = model.get_feature_importance()
fi_df = pd.DataFrame({'Feature': X.columns, 'Importance': feature_importance})
top_20_fi = fi_df.sort_values(by='Importance', ascending=False).head(20)

# 5. ВІЗУАЛІЗАЦІЯ (Твій блок)
print("Малювання об'єднаного графіка...")

fig, axes = plt.subplots(1, 2, figsize=(16, 7))

# --- Графік зліва (axes[0]): Confusion Matrix ---
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', cbar=False, ax=axes[0], annot_kws={"size": 14})
axes[0].set_title('Confusion Matrix - CatBoost Advanced', fontsize=16)
axes[0].set_xlabel('Predicted Label', fontsize=12)
axes[0].set_ylabel('True Label', fontsize=12)

# --- Графік справа (axes[1]): Top 20 Features ---
sns.barplot(x='Importance', y='Feature', data=top_20_fi, hue='Feature', palette='viridis', legend=False, ax=axes[1])
axes[1].set_title('Top Features Importance', fontsize=16)
axes[1].set_xlabel('Importance Score', fontsize=12)
axes[1].set_ylabel('Feature Name', fontsize=12)

plt.tight_layout()
combined_plot_filename = 'catboost_new.png'
plt.savefig(combined_plot_filename, dpi=300)
print(f"Обидва графіки збережено у файл: '{combined_plot_filename}'")

# 6. ЗБЕРЕЖЕННЯ МОДЕЛІ
with open("catboost_model_new.pkl", "wb") as f:
    pickle.dump(model, f)

print("\nЗвіт про класифікацію")
print(classification_report(y_test, y_pred))