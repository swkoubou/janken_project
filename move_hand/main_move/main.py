import cv2
import mediapipe as mp
import numpy as np
import torch
import torch.nn as nn
from collections import deque
import pygame
import time

# ==============================
# LSTMモデル
# ==============================
class JankenLSTM(nn.Module):
    def __init__(self, input_size=63, hidden_size=128, num_layers=2, num_classes=3):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        out, (h, c) = self.lstm(x)
        return self.fc(out[:, -1, :])

# ==============================
# 初期設定
# ==============================
pygame.mixer.init()
VOICE_PATH = "janken_voice.mp3"
MODEL_PATH = "janken_lstm.pth"
LABEL_MAP = {0: "Guu", 1: "Choki", 2: "Paa"}
SEQ_LEN = 15

model = JankenLSTM()
try:
    model.load_state_dict(torch.load(MODEL_PATH, map_location="cpu", weights_only=True))
    model.eval()
    print("LSTMモデルの読み込みに成功しました")
except Exception as e:
    print(f"モデル読み込み失敗: {e}")
    exit()

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.7, min_tracking_confidence=0.5)
mp_draw = mp.solutions.drawing_utils

# ==============================
# 状態管理
# ==============================
is_predicting = False
start_time = 0
pred_text = "READY"
buffer = [] # ここをリストとして使い、予測ごとにクリアする

cap = cv2.VideoCapture(0)

print("=== リアルタイム じゃんけん予測開始 ===")
print("[SPACE] を押して予測開始 / [ESC] 終了")

while True:
    ret, frame = cap.read()
    if not ret: break

    frame = cv2.flip(frame, 1)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb)

    status_color = (0, 255, 0) # 緑

    if is_predicting:
        elapsed = time.time() - start_time
        
        # 4.0秒〜4.7秒の間だけ「相対座標」をバッファに貯める
        if 4.0 <= elapsed <= 4.7:
            pred_text = "OBSERVING..."
            status_color = (0, 0, 255) # 赤
            if results.multi_hand_landmarks:
                hand_landmarks = results.multi_hand_landmarks[0]
                wrist = hand_landmarks.landmark[0]
                w_x, w_y, w_z = wrist.x, wrist.y, wrist.z
                
                row = []
                for lm in hand_landmarks.landmark:
                    row.extend([lm.x - w_x, lm.y - w_y, lm.z - w_z])
                buffer.append(row)
        
        # 4.7秒を過ぎたら推論実行
        elif elapsed > 4.7:
            if len(buffer) > 0:
                seq = np.array(buffer, dtype=np.float32)
                if len(seq) < SEQ_LEN:
                    pad = np.zeros((SEQ_LEN - len(seq), 63))
                    seq = np.vstack([seq, pad])
                else:
                    seq = seq[:SEQ_LEN]
                
                tensor = torch.tensor(seq).unsqueeze(0).float()
                
                with torch.no_grad():
                    output = model(tensor)
                    pred_idx = torch.argmax(output, dim=1).item()
                    pred_text = f"RESULT: {LABEL_MAP[pred_idx]}"
                
                buffer = [] # 次のために空にする
            else:
                pred_text = "NO HAND DETECTED"
            
            is_predicting = False
        else:
            pred_text = f"WAITING... ({elapsed:.1f}s)"
            status_color = (255, 255, 0)

    # ランドマーク描画（ここは常に表示）
    if results.multi_hand_landmarks:
        mp_draw.draw_landmarks(frame, results.multi_hand_landmarks[0], mp_hands.HAND_CONNECTIONS)

    # 結果表示
    cv2.putText(frame, pred_text, (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.2, status_color, 3)
    cv2.imshow("Janken Real-time Prediction", frame)

    key = cv2.waitKey(1) & 0xFF
    if key == 32: # SPACE
        if not is_predicting:
            buffer = [] # 完全にリセット
            pygame.mixer.music.load(VOICE_PATH)
            pygame.mixer.music.play()
            start_time = time.time()
            is_predicting = True
            print("予測サイクル開始...")

    elif key == 27: # ESC
        break

cap.release()
cv2.destroyAllWindows()