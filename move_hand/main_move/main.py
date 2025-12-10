import cv2
import mediapipe as mp
import numpy as np
import torch
import torch.nn as nn
from collections import deque

# ==============================
# LSTMモデル（学習時と同じ構造）
# ==============================

class JankenLSTM(nn.Module):
    def __init__(self, input_size=63, hidden_size=128, num_layers=2, num_classes=3):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size, hidden_size, num_layers, batch_first=True
        )
        self.fc = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        # x : (1, T, 63)
        out, (h, c) = self.lstm(x)
        out = self.fc(h[-1])  # 最後の層のhidden
        return out


# ==============================
# モデル読み込み
# ==============================
model = JankenLSTM()
model.load_state_dict(torch.load("G:\マイドライブ\ソフトウェア工房\じゃんけん\データ\janken_lstm.pth", map_location="cpu"))
model.eval()

label_map = {0: "Guu", 1: "Choki", 2: "Paa"}



# ==============================
# MediaPipe Hands 設定
# ==============================
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    max_num_hands=1,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.5
)
mp_draw = mp.solutions.drawing_utils


# ==============================
# 時系列バッファ
# ==============================
SEQ_LEN = 20        # LSTM入力フレーム数
buffer = deque(maxlen=SEQ_LEN)


# ==============================
# カメラ起動
# ==============================
cap = cv2.VideoCapture(0)

print("=== リアルタイム じゃんけん推論開始 ===")

pred_text = "----"

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    results = hands.process(rgb)

    if results.multi_hand_landmarks:
        hand_landmarks = results.multi_hand_landmarks[0]
        mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

        # 特徴量抽出（63次元）
        row = []
        for lm in hand_landmarks.landmark:
            row.extend([lm.x, lm.y, lm.z])

        buffer.append(row)

        # バッファが満タンなら推論
        if len(buffer) == SEQ_LEN:
            seq = np.array(buffer, dtype=np.float32)  # (20, 63)
            tensor = torch.tensor(seq).unsqueeze(0)   # (1,20,63)

            with torch.no_grad():
                out = model(tensor)
                pred = torch.argmax(out, dim=1).item()
                pred_text = label_map[pred]

    # 結果表示
    cv2.putText(frame, f"Predict: {pred_text}", (10, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 3)

    cv2.imshow("Real-time Janken AI", frame)

    if cv2.waitKey(1) == 27:  # ESC
        break

cap.release()
cv2.destroyAllWindows()
