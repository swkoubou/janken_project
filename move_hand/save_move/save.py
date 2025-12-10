import cv2
import mediapipe as mp
import csv
import os
import time
from datetime import datetime

# ======== 設定 ========
SAVE_DIR = "G:\マイドライブ\ソフトウェア工房\じゃんけん\データ\janken_dataset"   # 保存フォルダ
os.makedirs(SAVE_DIR, exist_ok=True)

# ======== MediaPipe Hands ========
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    max_num_hands=1,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.5
)
mp_draw = mp.solutions.drawing_utils


# ======== データ収集の状態管理 ========
collecting = False
current_label = None  # "g", "c", "p" のいずれか
csv_writer = None
csv_file = None


def start_new_csv(label):
    """ラベルごとにCSVファイルを新規作成"""
    global csv_writer, csv_file

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    filename = f"{label}_{timestamp}.csv"
    path = os.path.join(SAVE_DIR, filename)

    csv_file = open(path, "w", newline="", encoding="utf-8")
    csv_writer = csv.writer(csv_file)

    # ヘッダー（landmark_0_x, landmark_0_y, ...）
    header = []
    for i in range(21):
        header += [f"x_{i}", f"y_{i}", f"z_{i}"]
    header.append("label")
    csv_writer.writerow(header)

    print(f"[INFO] CSV作成: {path}")


# ======== メインループ ========
cap = cv2.VideoCapture(0)

print("------ じゃんけんデータ収集ツール ------")
print("[g] グー / [c] チョキ / [p] パー をセット")
print("[SPACE] 収集の開始・停止")
print("[ESC] 終了")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(img_rgb)

    # ランドマーク検出
    if results.multi_hand_landmarks:
        hand_landmarks = results.multi_hand_landmarks[0]
        mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

        # 収集中ならCSVに書き込み
        if collecting and current_label is not None:
            row = []
            for lm in hand_landmarks.landmark:
                row.extend([lm.x, lm.y, lm.z])
            row.append(current_label)
            csv_writer.writerow(row)

    # UI表示
    cv2.putText(frame, f"Label: {current_label}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)

    cv2.putText(frame, f"Collecting: {collecting}", (10, 70),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,255), 2)

    cv2.imshow("Janken Data Collector", frame)

    key = cv2.waitKey(1)

    # --- キー操作 ---
    if key == ord('g'):
        current_label = "g"
        print("[SET] ラベル = グー")
        #start_new_csv(current_label)

    elif key == ord('c'):
        current_label = "c"
        print("[SET] ラベル = チョキ")
        #start_new_csv(current_label)

    elif key == ord('p'):
        current_label = "p"
        print("[SET] ラベル = パー")
        #start_new_csv(current_label)

    elif key == 32:  # Space
        if current_label is None:
            print("[WARN] ラベルを先に設定してください (g/c/p)")
        else:
            collecting = not collecting
            print(f"[INFO] collecting = {collecting}")

            if collecting:
                start_new_csv(current_label)  # ★ ONになった瞬間に新しいCSV作成
            else:
                if csv_file:
                    csv_file.close()
                    print("[INFO] CSV保存完了")

    elif key == 27:  # ESC
        break

# 終了処理
if csv_file:
    csv_file.close()

cap.release()
cv2.destroyAllWindows()
