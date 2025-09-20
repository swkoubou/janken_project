from flask import Flask, render_template, request, jsonify
from tensorflow.keras.models import load_model
import pickle
import numpy as np
import joblib

app = Flask(__name__)

labels = ["グー", "チョキ", "パー"]

# 学習済みモデルの読み込み
try:
    model = load_model('../train_handshape/hand_gesture_model_tf.h5')
    label_encoder = joblib.load('../train_handshape/label_encoder.pkl')
    print("学習済みモデルを読み込みました。")
except FileNotFoundError:
    print("エラー: 学習済みモデル 'hand_gesture_model.pkl' が見つかりません。先にモデルをトレーニングして保存してください。")
    exit()

@app.route('/')
def index():
    return render_template("index.html")

@app.route('/gesture_data', methods=["POST"])
def gesture_data():
    data = request.get_json()
    landmarks = data.get("landmarks")

    if landmarks and len(landmarks) == 63:
        X = np.array(landmarks).reshape(1, -1)
        prediction = model.predict(X)[0]  # 例: [1, 0, 0] または [0.8, 0.1, 0.1]
        predicted_index = int(np.argmax(prediction))  # 最大値のインデックス
        predicted_label = labels[predicted_index]
        return jsonify({"gesture": predicted_label})
    else:
        return jsonify({"gesture": "なし"})



if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5000, debug=True)
