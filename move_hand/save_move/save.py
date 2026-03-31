import cv2
import mediapipe as mp
import csv
import os
import time
from datetime import datetime
import pygame  # 追加: pip install pygame

# ======== 設定 ========
SAVE_DIR = r"C:\Development\janken_project\move_hand\save_move\data"
os.makedirs(SAVE_DIR, exist_ok=True)
VOICE_PATH = "janken_voice.mp3"  # 音声ファイルのパス

# ======== pygame初期化 ========
pygame.mixer.init()

# ======== MediaPipe Hands ========
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    max_num_hands=1,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.5
)
mp_draw = mp.solutions.drawing_utils

# ======== 状態管理変数 ========
collecting = False
start_time = 0
current_label = None
csv_writer = None
csv_file = None

def start_new_csv(label):
    global csv_writer, csv_file
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    filename = f"{label}_{timestamp}.csv"
    path = os.path.join(SAVE_DIR, filename)
    csv_file = open(path, "w", newline="", encoding="utf-8")
    csv_writer = csv.writer(csv_file)
    
    header = []
    for i in range(21):
        header += [f"x_{i}", f"y_{i}", f"z_{i}"]
    header.append("label")
    csv_writer.writerow(header)
    return path

# ======== メインループ ========
cap = cv2.VideoCapture(0)

print("------ じゃんけん自動収集ツール ------")
print("[g] グー / [c] チョキ / [p] パー をセット")
print("[SPACE] 音声再生＆3.8s〜4.5sの間を自動保存")

while True:
    ret, frame = cap.read()
    if not ret: break

    frame = cv2.flip(frame, 1)
    img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(img_rgb)

    elapsed = 0
    if collecting:
        elapsed = time.time() - start_time
        
        # 3.8秒から4.5秒の間だけ保存
        if 3.8 <= elapsed <= 4.5:
            save_status = "RECORDING..."
            color = (0, 0, 255) # 赤
            
            if results.multi_hand_landmarks:
                hand_landmarks = results.multi_hand_landmarks[0]
                row = []
                for lm in hand_landmarks.landmark:
                    row.extend([lm.x, lm.y, lm.z])
                row.append(current_label)
                if csv_writer:
                    csv_writer.writerow(row)
        
        elif elapsed > 4.5:
            # 終了処理
            collecting = False
            if csv_file:
                csv_file.close()
                csv_file = None
                print("[INFO] 保存完了")
            save_status = "IDLE"
            color = (0, 255, 255)
        else:
            save_status = f"WAITING... ({elapsed:.1f}s)"
            color = (255, 255, 0)
    else:
        save_status = "READY"
        color = (0, 255, 0)

    # ランドマーク描画
    if results.multi_hand_landmarks:
        mp_draw.draw_landmarks(frame, results.multi_hand_landmarks[0], mp_hands.HAND_CONNECTIONS)

    # UI表示
    cv2.putText(frame, f"Label: {current_label}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)
    cv2.putText(frame, f"Status: {save_status}", (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)

    cv2.imshow("Janken Data Collector", frame)
    key = cv2.waitKey(1) & 0xFF

    if key == ord('g'): current_label = "g"; print("[SET] グー")
    elif key == ord('c'): current_label = "c"; print("[SET] チョキ")
    elif key == ord('p'): current_label = "p"; print("[SET] パー")
    
    elif key == 32:  # Space
        if current_label is None:
            print("[WARN] ラベルを先に設定してください")
        elif not collecting:
            # 再生と計測開始
            print(f"[START] {current_label} の収集開始...")
            start_new_csv(current_label)
            pygame.mixer.music.load(VOICE_PATH)
            pygame.mixer.music.play()
            start_time = time.time()
            collecting = True

    elif key == 27: break

cap.release()
cv2.destroyAllWindows()