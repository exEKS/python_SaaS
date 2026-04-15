import pandas as pd
import json
import os
from datetime import datetime
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split
import joblib

HISTORICAL_DATA = 'data/historical_data.csv'
NEW_DATA = 'data/new_data.csv'
MODEL_PATH = 'models/champion_model.pkl'
META_PATH = 'model_meta.json'


def load_and_merge():
    df_old = pd.read_csv(HISTORICAL_DATA)
    df_new = pd.read_csv(NEW_DATA)
    full_df = pd.concat([df_old, df_new]).drop_duplicates().reset_index(drop=True)
    full_df.to_csv(HISTORICAL_DATA, index=False)
    return full_df


def get_performance(model, X, y):
    preds = model.predict(X)
    return mean_absolute_error(y, preds)


def retrain():
    df = load_and_merge()
    X = df.drop(columns=['target_alarm_next_day'])
    y = df['target']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

    challenger_model = RandomForestRegressor(n_estimators=100)
    challenger_model.fit(X_train, y_train)
    challenger_mae = get_performance(challenger_model, X_test, y_test)

    if os.path.exists(MODEL_PATH):
        champion_model = joblib.load(MODEL_PATH)
        champion_mae = get_performance(champion_model, X_test, y_test)

        print(f"Champion MAE: {champion_mae:.4f}")
        print(f"Challenger MAE: {challenger_mae:.4f}")

        if challenger_mae <= champion_mae:
            print("Challenger is better. Promoting to Champion.")
            promote_model(challenger_model, challenger_mae)
        else:
            print("Challenger performed worse. Keeping existing model.")
    else:
        print("No existing model found. Promoting first model.")
        promote_model(challenger_model, challenger_mae)


def promote_model(model, mae):
    joblib.dump(model, MODEL_PATH)
    meta = {
        "last_model_train_time": datetime.utcnow().isoformat() + "Z",
        "model_name": "RandomForest",
        "model_version": datetime.now().strftime("%Y%m%d_%H%M"),
        "performance_mae": mae
    }
    with open(META_PATH, 'w') as f:
        json.dump(meta, f, indent=4)


if __name__ == "__main__":
    retrain()