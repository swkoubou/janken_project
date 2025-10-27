import pandas as pd
import numpy as np
import os
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
import joblib

# === パス設定 ===
DATA_DIR = "C:/Users/2321004/Desktop/janken_project/janken_prediction/move_hand/save_move/data"
MODEL_PATH = "hand_classifier.pkl"

# === データ読み込み ===
all_data = []
for file in os.listdir(DATA_DIR):
    if file.endswith(".csv"):
        df = pd.read_csv(os.path.join(DATA_DIR, file))
        # 1ファイル = 1サンプル として集約
        df_pivot = df.pivot(index="frame", columns="landmark_id", values=["x", "y", "z"])
        df_pivot = df_pivot.fillna(0).values.flatten()  # フラット化
        label = df["label"].iloc[0]
        print(label)
        all_data.append((df_pivot, label))

# === 特徴量とラベルに分割 ===
X = np.array([x for x, _ in all_data])
y = np.array([y for _, y in all_data])

# === 学習・評価 ===
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# === 評価 ===
y_pred = model.predict(X_test)
print("=== Classification Report ===")
print(classification_report(y_test, y_pred))

# === モデル保存 ===
joblib.dump(model, MODEL_PATH)
print(f"Model saved to {MODEL_PATH}")
