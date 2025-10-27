import cv2
import mediapipe as mp
import numpy as np
import pandas as pd
import time
import os
import pygame

# MediaPipe Handsの初期化
mp_drawing = mp.solutions.drawing_utils
mp_hands = mp.solutions.hands

# 保存ディレクトリ
SAVE_DIR = "data"
os.makedirs(SAVE_DIR, exist_ok=True)

# 記録時間（秒）を指定
RECORD_TIME = 1.2
FPS = 30

# 1サンプルあたりのフレーム数
MAX_FRAMES = int(RECORD_TIME * FPS)

# 使用するラベル（例：グー、チョキ、パー）
LABEL = "パー"

def record_sequence():
    data_records = []
    cap = cv2.VideoCapture(0)

    with mp_hands.Hands(
        max_num_hands=1,
        min_detection_confidence=0.7,
        min_tracking_confidence=0.5
    ) as hands:
        print("Recording started!")

        start_time = time.time()
        frame_index = 0

        while frame_index < MAX_FRAMES:
            ret, frame = cap.read()
            if not ret:
                break

            frame = cv2.flip(frame, 1)
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            result = hands.process(rgb_frame)

            if result.multi_hand_landmarks:
                hand_landmarks = result.multi_hand_landmarks[0]

                # === 手首(0)を基準に相対座標を計算 ===
                wrist = np.array([
                    hand_landmarks.landmark[0].x,
                    hand_landmarks.landmark[0].y,
                    hand_landmarks.landmark[0].z
                ])

                for idx, lm in enumerate(hand_landmarks.landmark):
                    relative_x = lm.x - wrist[0]
                    relative_y = lm.y - wrist[1]
                    relative_z = lm.z - wrist[2]
                    data_records.append({
                        "frame": frame_index,
                        "landmark_id": idx,
                        "x": relative_x,
                        "y": relative_y,
                        "z": relative_z,
                        "label": LABEL
                    })
            else:
                for idx in range(21):
                    data_records.append({
                        "frame": frame_index,
                        "landmark_id": idx,
                        "x": 0.0,
                        "y": 0.0,
                        "z": 0.0,
                        "label": LABEL
                    })

                # 手のランドマーク描画
                mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            cv2.imshow("Recording", frame)
            if cv2.waitKey(1) & 0xFF == 27:
                break

            frame_index += 1

            # 時間超過チェック
            if time.time() - start_time > RECORD_TIME:
                break

        cap.release()
        cv2.destroyAllWindows()

        # === CSV保存 ===
        df = pd.DataFrame(data_records)
        save_path = os.path.join(SAVE_DIR, f"{LABEL}_{int(time.time())}.csv")
        df.to_csv(save_path, index=False)
        print(f"Saved {len(df)} rows → {save_path}")

if __name__ == "__main__":
    pygame.mixer.init()
    pygame.mixer.music.load("janken_voice.wav")
    pygame.mixer.music.play()
    record_sequence()
