from flask import Flask, render_template, Response, request, jsonify
import cv2
import mediapipe as mp
import csv
import threading
import os
import json
import pandas as pd

app = Flask(__name__)

# MediaPipe Hands 初期化
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(min_detection_confidence=0.7, min_tracking_confidence=0.7)
mp_drawing = mp.solutions.drawing_utils

# CSVファイル作成
CSV_FILE = "hand_landmarks.csv"
if not os.path.exists(CSV_FILE):
    with open(CSV_FILE, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['label'] + [f'x{i}' for i in range(21)] + [f'y{i}' for i in range(21)])

# Webカメラ映像をストリーミング
def generate_frames():
    cap = cv2.VideoCapture(0)
    while True:
        success, frame = cap.read()
        if not success:
            break

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(frame_rgb)

        landmarks_data = []
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
                for landmark in hand_landmarks.landmark:
                    landmarks_data.append({"x": landmark.x, "y": landmark.y})

        print(landmarks_data)

        _, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n'
               b'Content-Type: application/json\r\n\r\n' + json.dumps(landmarks_data).encode() + b'\r\n')

    cap.release()

# 手のランドマークをCSVに保存
def save_to_csv(label, landmarks):
    with open(CSV_FILE, 'a', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([label] + landmarks)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/save_landmarks', methods=['POST'])
def save_landmarks():
    data = request.json
    label = data.get('label')
    landmarks = data.get('landmarks')

    if label is None or landmarks is None:
        return jsonify({"error": "Invalid data"}), 400

    threading.Thread(target=save_to_csv, args=(label, landmarks)).start()

    return jsonify({"message": "Data saved successfully"}), 200

@app.route('/get_landmarks')
def get_landmarks():
    try:
        df = pd.read_csv(CSV_FILE)
        grouped = df.groupby('label', group_keys=False).apply(lambda x: x.iloc[:, 1:].values.tolist()).to_dict()
        print("Grouped data:", grouped)  # デバッグ用
        return jsonify(grouped)
    except Exception as e:
        print("Error:", str(e))
        return jsonify({"error": str(e)}), 500



if __name__ == "__main__":
    app.run(debug=True, threaded=True)
