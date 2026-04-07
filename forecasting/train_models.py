import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pickle
import os
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import TimeSeriesSplit, GridSearchCV
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, classification_report

DATA_PATH = r"C:\Users\Ukraine\OneDrive\Робочий стіл\2026_ukma_saas\features_final.csv"
MODEL_DIR = "models/"
os.makedirs(MODEL_DIR, exist_ok=True)

def train_and_evaluate():
    # Завантаження даних
    if not os.path.exists(DATA_PATH):
        print(f"Помилка: Файл {DATA_PATH} не знайдено!")
        return

    df = pd.read_csv(DATA_PATH)
    
    TARGET = 'target_alarm_next_day'
    print(f"\n--- АНАЛІЗ ДАНИХ ---")
    print(f"Знайдено колонок: {len(df.columns)}")
    print(f"Цільова змінна (Target): {TARGET}")
    print(f"--------------------\n")
    
    X = df.select_dtypes(include=[np.number]).drop(columns=[TARGET], errors='ignore').fillna(0)
    y = df[TARGET].astype(int)

    tscv = TimeSeriesSplit(n_splits=5)

    models_config = {
        # "RandomForest": {
        #     "model": RandomForestClassifier(random_state=42),
        #     "params": {
        #         'n_estimators': [100, 200],
        #         'max_depth': [10, 20, None],
        #         'min_samples_split': [2, 5]
        #     }
        # },
        # "XGBoost": {
        #     "model": XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42),
        #     "params": {
        #         'n_estimators': [100],
        #         'learning_rate': [0.01, 0.1],
        #         'max_depth': [3, 6]
        #     }
        # },
        "DecisionTreeBalanced": {
            "model": DecisionTreeClassifier(random_state=42),
            "params": {
                'max_depth': [5, 10, 20],
                'min_samples_leaf': [1, 5],
                'class_weight': ['balanced']
            }
        },
        "DecisionTree": {
            "model": DecisionTreeClassifier(random_state=42),
            "params": {
                'max_depth': [5, 10, 20],
                'min_samples_leaf': [1, 5]
            }
        }
    }

    best_models = {}

    for name, config in models_config.items():
        print(f"--- Навчання та тюнінг (GridSearchCV): {name} ---")
        # Тюнінг гіперпараметрів за метрикою f1
        grid = GridSearchCV(config["model"], config["params"], cv=tscv, scoring='f1', n_jobs=-1)
        grid.fit(X, y)
        
        best_model = grid.best_estimator_
        best_models[name] = best_model
        
        # Збереження моделі у формат .pkl
        model_filename = f"{MODEL_DIR}{name.lower()}_model.pkl"
        with open(model_filename, "wb") as f:
            pickle.dump(best_model, f)
        print(f"Модель {name} збережена як {model_filename}")
        
        # Оцінка та візуалізація
        y_pred = best_model.predict(X)
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
        
        # 1. Confusion Matrix (Матриця помилок)
        ConfusionMatrixDisplay.from_predictions(y, y_pred, ax=ax1, cmap='Blues')
        ax1.set_title(f"Confusion Matrix: {name}")
        
        # 2. Top 20 Features (Важливість ознак)
        importances = best_model.feature_importances_
        indices = np.argsort(importances)[-20:]
        ax2.barh(range(len(indices)), importances[indices], color='skyblue')
        ax2.set_yticks(range(len(indices)))
        ax2.set_yticklabels([X.columns[i] for i in indices])
        ax2.set_title(f"Top 20 Features: {name}")
        
        plt.tight_layout()
        print(f"Відображаю графіки для {name}. Закрийте вікно графіка, щоб продовжити...")
        plt.show()
        
        print(f"Найкращі параметри для {name}: {grid.best_params_}")
        print(classification_report(y, y_pred))

if __name__ == "__main__":
    train_and_evaluate()