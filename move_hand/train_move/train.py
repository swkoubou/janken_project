import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import os
import pandas as pd
from torch.utils.data import DataLoader, TensorDataset

# --- モデル定義 (main.pyと同じ構造) ---
class JankenLSTM(nn.Module):
    def __init__(self, input_size=63, hidden_size=128, num_layers=2, num_classes=3):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        # x shape: (batch, seq_len, 63)
        out, (h, c) = self.lstm(x)
        # 最後のタイムステップの出力を全結合層へ
        return self.fc(out[:, -1, :])

# --- 設定 ---
DATA_DIR = r"C:\Development\janken_project\move_hand\save_move\data"
SEQ_LEN = 15  # 0.7秒間(3.8s-4.5s)のデータ。カメラのFPSによりますが15-20程度。
LABEL_MAP = {'g': 0, 'c': 1, 'p': 2}

# --- データ準備 ---
all_X, all_y = [], []

print("データを読み込み中...")
for file in os.listdir(DATA_DIR):
    if file.endswith(".csv"):
        path = os.path.join(DATA_DIR, file)
        df = pd.read_csv(path)
        
        if len(df) < 5: # 極端に短いデータは無視
            continue
            
        # ラベル取得 (最後の列)
        raw_label = df["label"].iloc[0]
        label = LABEL_MAP.get(raw_label)
        
        if label is None: continue

        # 特徴量取得 (x_0, y_0, z_0 ... x_20, y_20, z_20)
        # label列以外を取得
        features = df.drop(columns=["label"]).values
        
        # 長さをSEQ_LENに合わせる (足りなければパディング、多ければカット)
        if len(features) < SEQ_LEN:
            # 0でパディング
            pad = np.zeros((SEQ_LEN - len(features), 63))
            features = np.vstack([features, pad])
        else:
            # 冒頭からSEQ_LEN分だけ使用
            features = features[:SEQ_LEN]
            
        all_X.append(features)
        all_y.append(label)

if not all_X:
    print("エラー: 学習データが見つかりませんでした。")
    exit()

X_tensor = torch.tensor(np.array(all_X), dtype=torch.float32)
y_tensor = torch.tensor(np.array(all_y), dtype=torch.long)

print(f"読み込み完了: {len(all_X)} サンプル")

# --- 学習設定 ---
model = JankenLSTM()
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)
dataset = DataLoader(TensorDataset(X_tensor, y_tensor), batch_size=8, shuffle=True)

# --- 学習ループ ---
model.train()
epochs = 100
for epoch in range(epochs):
    epoch_loss = 0
    for inputs, labels in dataset:
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()
    
    if (epoch + 1) % 10 == 0:
        print(f"Epoch {epoch+1}/{epochs}, Loss: {epoch_loss/len(dataset):.4f}")

# --- 保存 ---
torch.save(model.state_dict(), "janken_lstm.pth")
print("学習完了: janken_lstm.pth として保存しました。")