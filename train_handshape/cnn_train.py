import os
import numpy as np
import cv2
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from tensorflow.keras.callbacks import EarlyStopping
import joblib
import sys

# ====== 画像の前処理 ======
def load_images_from_folder(folder_path, img_size=(64, 64)):
    X = []
    y = []

    for label in ['0', '1', '2']:  # グー・チョキ・パーを想定
        class_dir = os.path.join(folder_path, label)
        for filename in os.listdir(class_dir):
            img_path = os.path.join(class_dir, filename)
            img = cv2.imread(img_path)
            if img is not None:
                img = cv2.resize(img, img_size)
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # RGBに変換（任意）
                X.append(img / 255.0)  # 正規化
                y.append(int(label))
    
    return np.array(X), np.array(y)

# ====== モデル構築 ======
def build_cnn_model(input_shape, num_classes=3):
    model = Sequential()
    model.add(Conv2D(32, (3, 3), activation='relu', input_shape=input_shape))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    
    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Conv2D(128, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Flatten())
    model.add(Dropout(0.5))
    model.add(Dense(64, activation='relu'))
    model.add(Dense(num_classes, activation='softmax'))

    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model

# ====== 実行部分 ======
if __name__ == "__main__":
    image_dir = '../create_dataset/images'
    X, y = load_images_from_folder(image_dir, img_size=(64, 64))

    y_cat = to_categorical(y, num_classes=3)
    X_train, X_test, y_train, y_test = train_test_split(X, y_cat, test_size=0.2, random_state=42)

    model = build_cnn_model(input_shape=X_train.shape[1:], num_classes=3)

    # Early stopping (optional)
    early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

    model.fit(X_train, y_train, epochs=50, batch_size=32, validation_split=0.2, callbacks=[early_stopping])

    # テスト評価
    test_loss, test_acc = model.evaluate(X_test, y_test)
    print(f'CNNモデルのテスト精度: {test_acc:.4f}')

    # モデル保存（HDF5形式）
    model.save('cnn_hand_gesture_model.keras')
    print('CNNモデルを保存しました。')
