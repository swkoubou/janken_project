import cv2
import mediapipe as mp
import pandas as pd
import os

csv_file = "data/hand_landmarks.csv"

# MediaPipe Handsのセットアップ
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1, min_detection_confidence=0.5)
mp_drawing = mp.solutions.drawing_utils

# カメラのセットアップ
cap = cv2.VideoCapture(0)

# 手の座標データを収集
landmarks_data = []

def collect_landmarks(label):
    global landmarks_data
    count = 0
    print(f"\n=== Collecting data for {label} ===")
    print("Press 'q' to stop collecting...")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to capture image")
            break

        # 画像保存
        cv2.imwrite(f'images/{label}/{count}.png', frame)

        frame = cv2.flip(frame, 1)  # 水平方向に反転
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)  # RGB変換

        results = hands.process(image)
        frame = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)  # BGRに戻す

        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                landmarks = []
                for lm in hand_landmarks.landmark:
                    landmarks.extend([lm.x, lm.y, lm.z])
                landmarks.append(label)
                landmarks_data.append(landmarks)
                mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            count += 1
            print(f"Collected {count} images for {label}")

        cv2.imshow('Hand Landmarks', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

        # ウィンドウが閉じられたら終了
        if cv2.getWindowProperty('Hand Landmarks', cv2.WND_PROP_VISIBLE) < 1:
            break

    print(f"\nFinished collecting {count} images for {label}.")

    cv2.destroyAllWindows()
    return

# フォルダが存在しない場合は作成
os.makedirs(os.path.dirname(csv_file), exist_ok=True)

# データ収集
input("Press Enter to collect data for Rock (グー)...")
collect_landmarks(0)

input("Press Enter to collect data for Scissors (チョキ)...")
collect_landmarks(1)

input("Press Enter to collect data for Paper (パー)...")
collect_landmarks(2)

# CSVファイルに保存
try:
    columns = [f'x{i}' for i in range(21)] + [f'y{i}' for i in range(21)] + [f'z{i}' for i in range(21)] + ['label']
    df = pd.DataFrame(landmarks_data, columns=columns)
    df.to_csv(csv_file, index=False)
    print(f'\nData saved successfully to {csv_file}')
except Exception as e:
    print(f'Error saving data: {e}')

# リソース解放
cap.release()
cv2.destroyAllWindows()
