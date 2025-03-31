from flask import Flask, render_template, Response
import cv2
import mediapipe as mp
import numpy as np
import joblib
import json
from multiprocessing import shared_memory
import sys
import os

# 上位ディレクトリをPythonのモジュール検索パスに追加
sys.path.append(os.path.abspath("../train_handshape"))

from train import DeepNeuralNetwork

# Flaskアプリケーションの初期化
app = Flask(__name__)

# MediaPipe Hands 初期化
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(min_detection_confidence=0.7, min_tracking_confidence=0.7)
mp_drawing = mp.solutions.drawing_utils

# 学習済みモデルの読み込み
try:
    model = joblib.load('hand_gesture_model.pkl')
    print("学習済みモデルを読み込みました。")
except FileNotFoundError:
    print("エラー: 学習済みモデル 'hand_gesture_model.pkl' が見つかりません。先にモデルをトレーニングして保存してください。")
    exit()

# 共有メモリの作成（21点のx, y座標 * 2（float64））
shm = shared_memory.SharedMemory(create=True, size=21 * 2 * 8)  # float64 × 42
shm_array = np.ndarray((21, 2), dtype=np.float64, buffer=shm.buf)

# 手のランドマーク座標を予測用の形式に変換する関数
def preprocess_landmarks(landmarks):
    if not landmarks:
        return np.zeros((1, 42))  # ランドマークがない場合はゼロベクトルを返す
    try:
        data = np.array([[lm.x, lm.y] for lm in landmarks]).flatten().reshape(1, -1)
        return data
    except AttributeError:
        # ランドマークの形式が正しくない場合の処理
        return np.zeros((1, 42))

# 手の形を分類する関数
def classify_hand(landmarks):
    if not landmarks:
        return "検出なし"
    try:
        print("classify_hand に渡された landmarks:", landmarks)
        processed_landmarks = preprocess_landmarks(landmarks)
        print("processed_landmarks:", processed_landmarks)
        if processed_landmarks.shape == (1, 42):
            prediction = model.predict(processed_landmarks)
            # ラベルを具体的なジェスチャー名にマッピングする辞書を作成する
            gesture_map = {
                0: "グー",
                1: "チョキ",
                2: "パー",
                # 必要に応じて他のラベルとジェスチャーの対応を追加
            }
            result = gesture_map.get(prediction[0], "不明")
            print("予測結果:", prediction, "ジェスチャー:", result)
            return gesture_map.get(prediction[0], "不明")
        else:
            return "ランドマークデータの形式が不正です"
    except Exception as e:
        print(f"分類中にエラーが発生しました: {e}")
        return "分類エラー"

# Webカメラ映像をストリーミング
def generate_frames():
    cap = cv2.VideoCapture(0)
    while True:
        success, frame = cap.read()
        if not success:
            break

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(frame_rgb)

        print("results.multi_hand_landmarks:", results.multi_hand_landmarks)

        landmarks_data = []
        predicted_gesture = "検出なし"

        if results.multi_hand_landmarks:
            print("手の検出に成功しました")
            for hand_landmarks in results.multi_hand_landmarks:
                mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

                landmarks = []
                for i, landmark in enumerate(hand_landmarks.landmark):
                    shm_array[i, 0] = landmark.x
                    shm_array[i, 1] = landmark.y
                    landmarks.append(landmark) # landmarkオブジェクトをそのまま追加

                if landmarks:
                    predicted_gesture = classify_hand(landmarks)
        else:
            print("手の検出に失敗しました")

        # JSONデータも埋め込み
        _, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()

        # JSONデータに予測結果を追加
        json_data = {"landmarks": [], "gesture": predicted_gesture}

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n'
               b'Content-Type: application/json\r\n\r\n' + json.dumps(json_data).encode() + b'\r\n')

    cap.release()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == "__main__":
    app.run(debug=True, threaded=True)