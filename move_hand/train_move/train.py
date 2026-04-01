import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import os
import pandas as pd
from torch.utils.data import DataLoader, TensorDataset

# --- モデル定義 (構造は変更なし) ---
class JankenLSTM(nn.Module):
    def __init__(self, input_size=63, hidden_size=128, num_layers=2, num_classes=3):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        out, (h, c) = self.lstm(x)
        return self.fc(out[:, -1, :])

# --- データ拡張関数 ---
def augment_data(batch_X):
    """
    batch_X: (batch_size, seq_len, 63) のテンソル
    """
    # 1. 微小なノイズを加える (指の震えなどを再現)
    noise = torch.randn_like(batch_X) * 0.005
    batch_X = batch_X + noise

    # 2. 並行移動 (画面内での手の位置のズレを再現)
    # x, y, z ごとにランダムな移動量を決める (-0.05 ~ 0.05)
    for i in range(batch_X.shape[0]):
        shift_x = (np.random.rand() - 0.5) * 0.1
        shift_y = (np.random.rand() - 0.5) * 0.1
        
        # 0, 3, 6... が x座標 / 1, 4, 7... が y座標
        batch_X[i, :, 0::3] += shift_x
        batch_X[i, :, 1::3] += shift_y
        
    return batch_X

# --- 設定 ---
DATA_DIR = r"C:\Development\janken_project\move_hand\save_move\data"
SEQ_LEN = 15
LABEL_MAP = {'g': 0, 'c': 1, 'p': 2}

# --- データ準備 (読み込み部分は変更なし) ---
all_X, all_y = [], []
print("データを読み込み中...")
for file in os.listdir(DATA_DIR):
    if file.endswith(".csv"):
        path = os.path.join(DATA_DIR, file)
        df = pd.read_csv(path)
        if len(df) < 5: continue
        raw_label = df["label"].iloc[0]
        label = LABEL_MAP.get(raw_label)
        if label is None: continue
        features = df.drop(columns=["label"]).values
        # 【追加】手首相対座標への変換
        for frame_idx in range(len(features)):
            wrist_x = features[frame_idx, 0]
            wrist_y = features[frame_idx, 1]
            wrist_z = features[frame_idx, 2]
            
            for lm_idx in range(21):
                features[frame_idx, lm_idx*3] -= wrist_x
                features[frame_idx, lm_idx*3+1] -= wrist_y
                features[frame_idx, lm_idx*3+2] -= wrist_z

        if len(features) < SEQ_LEN:
            pad = np.zeros((SEQ_LEN - len(features), 63))
            features = np.vstack([features, pad])
        else:
            features = features[:SEQ_LEN]
        all_X.append(features)
        all_y.append(label)

X_tensor = torch.tensor(np.array(all_X), dtype=torch.float32)
y_tensor = torch.tensor(np.array(all_y), dtype=torch.long)
print(f"読み込み完了: {len(all_X)} サンプル")

# --- 学習設定 ---
model = JankenLSTM()
weights = torch.tensor([1.0, 1.2, 1.0], dtype=torch.float32) # チョキを2.0倍に強化
criterion = nn.CrossEntropyLoss(weight=weights)
optimizer = optim.Adam(model.parameters(), lr=0.001)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=100, gamma=0.5)
dataset = DataLoader(TensorDataset(X_tensor, y_tensor), batch_size=8, shuffle=True)

# --- 学習ループ ---
model.train()
epochs = 500 
print(f"学習開始 (Data Augmentation 有り)...")

for epoch in range(epochs):
    epoch_loss = 0
    for inputs, labels in dataset:
        # 学習時のみデータ拡張を適用
        inputs_aug = augment_data(inputs.clone())
        
        optimizer.zero_grad()
        outputs = model(inputs_aug)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()
    
    scheduler.step()
    if (epoch + 1) % 20 == 0:
        print(f"Epoch {epoch+1}/{epochs}, Loss: {epoch_loss/len(dataset):.4f}")

torch.save(model.state_dict(), "janken_lstm.pth")
print("学習完了: 拡張済みモデルを保存しました。")