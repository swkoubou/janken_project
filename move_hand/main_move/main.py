import cv2
import mediapipe as mp
import numpy as np
import joblib
import time

# === モデルのロード ===
MODEL_PATH = "../train_move/hand_classifier.pkl"
model = joblib.load(MODEL_PATH)

# === MediaPipe Hands初期化 ===
mp_drawing = mp.solutions.drawing_utils
mp_hands = mp.solutions.hands

# === 設定 ===
FPS = 30
SEQUENCE_LENGTH = int(1.2 * FPS)  # 1.2秒分のフレームを使用
print("モデル読み込み完了。カメラ起動中...")

# === カメラ起動 ===
cap = cv2.VideoCapture(0)

sequence = []  # フレームごとのランドマークデータを一時保存
pred_label = "判定中..."

with mp_hands.Hands(
    max_num_hands=1,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.5
) as hands:
    prev_time = time.time()

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.flip(frame, 1)
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = hands.process(rgb_frame)

        if result.multi_hand_landmarks:
            hand_landmarks = result.multi_hand_landmarks[0]

            # 手首(0)を基準に相対座標
            wrist = np.array([
                hand_landmarks.landmark[0].x,
                hand_landmarks.landmark[0].y,
                hand_landmarks.landmark[0].z
            ])

            frame_data = []
            for lm in hand_landmarks.landmark:
                frame_data.extend([
                    lm.x - wrist[0],
                    lm.y - wrist[1],
                    lm.z - wrist[2]
                ])
            sequence.append(frame_data)

            # === フレームが一定数たまったら予測 ===
            if len(sequence) >= SEQUENCE_LENGTH:
                input_data = np.array(sequence[:SEQUENCE_LENGTH]).flatten().reshape(1, -1)
                pred_label = model.predict(input_data)[0]
                sequence = []  # 使い終わったらリセット

            # ランドマーク描画
            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

        # === 結果の表示 ===
        cv2.putText(frame, f"Prediction: {pred_label}", (10, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 3)

        # FPS表示
        curr_time = time.time()
        fps = 1 / (curr_time - prev_time)
        prev_time = curr_time
        cv2.putText(frame, f"FPS: {int(fps)}", (10, 80),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

        cv2.imshow("Real-time Hand Prediction", frame)

        # ESCキーで終了
        if cv2.waitKey(1) & 0xFF == 27:
            break

cap.release()
cv2.destroyAllWindows()
