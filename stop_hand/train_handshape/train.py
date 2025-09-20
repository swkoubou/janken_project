import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score
import joblib

# ====== 1. データの読み込みと前処理 ======
df = pd.read_csv("hand_landmarks.csv")

# 特徴量とラベルに分ける
X = df.drop('label', axis=1).values
y = df['label'].values

# ラベルエンコーディング
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)

# ワンホットエンコーディング
y_onehot = tf.keras.utils.to_categorical(y_encoded)

# データ分割
X_train, X_test, y_train, y_test = train_test_split(X, y_onehot, test_size=0.2, random_state=40)

print("X_train shape:", X_train.shape)
print("X_test shape:", X_test.shape)

# ====== 2. モデル構築 ======
model = tf.keras.models.Sequential([
    tf.keras.layers.Input(shape=(X_train.shape[1],)),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dense(32, activation='relu'),
    tf.keras.layers.Dense(32, activation='relu'),
    tf.keras.layers.Dense(16, activation='relu'),
    tf.keras.layers.Dense(3, activation='softmax')  # クラス数に応じて調整
])

# ====== 3. モデルコンパイル ======
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.01),
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# ====== 4. モデル訓練 ======
history = model.fit(X_train, y_train, epochs=1000, batch_size=32, verbose=0)

# 途中経過を出力（例：100エポックごと）
for epoch in range(0, 1000, 100):
    acc = history.history['accuracy'][epoch]
    print(f"Epoch {epoch}, Accuracy: {acc:.4f}")

# ====== 5. テスト精度の評価 ======
y_pred_probs = model.predict(X_test)
y_pred = np.argmax(y_pred_probs, axis=1)
y_true = np.argmax(y_test, axis=1)

acc = accuracy_score(y_true, y_pred)
print(f"モデルの正解率: {acc:.2f}")

# ====== 6. モデル保存 ======
model.save('hand_gesture_model_tf.h5')
joblib.dump(label_encoder, 'label_encoder.pkl')
print("モデルとエンコーダを保存しました。")
