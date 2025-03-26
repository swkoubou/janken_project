import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import numpy as np
import joblib

# CSVファイルのパス
csv_file = 'hand_landmarks.csv'  # 実際のファイル名に置き換えてください

# CSVファイルを読み込む
df = pd.read_csv(csv_file)

# ラベルと特徴量を分離
X = df.iloc[:, 1:]  # 最初の列（label）以外を特徴量とする
y = df['label']

# データをトレーニングセットとテストセットに分割
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# ランダムフォレスト分類器を初期化
# 必要に応じて他のハイパーパラメータを調整できます
model = RandomForestClassifier(n_estimators=100, random_state=42)

# モデルをトレーニング
model.fit(X_train, y_train)

# テストデータで予測を行う
y_pred = model.predict(X_test)

# 正解率を評価
accuracy = accuracy_score(y_test, y_pred)
print(f'モデルの正解率: {accuracy:.2f}')

#モデルの保存
joblib.dump(model, 'hand_gesture_model.pkl')
print('モデルを保存しました。')

