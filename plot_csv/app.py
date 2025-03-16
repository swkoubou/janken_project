import pandas as pd
import matplotlib.pyplot as plt

# CSVファイルの読み込み
df = pd.read_csv("hand_landmarks.csv")  # ファイル名を適宜変更

# x, y 座標を抽出（ラベル列を除く）
num_landmarks = 21  # 手のランドマーク数
x_columns = [f"x{i}" for i in range(num_landmarks)]
y_columns = [f"y{i}" for i in range(num_landmarks)]

# 複数のサンプルをプロット
for i in range(min(5, len(df))):  # 最初の5サンプルをプロット
    x = df.loc[i, x_columns]
    y = df.loc[i, y_columns]
    
    plt.figure(figsize=(4, 4))
    plt.scatter(x, -y, c="blue")  # y軸を反転（画像座標系に合わせる）
    
    # 各ポイントに番号を表示
    for j in range(num_landmarks):
        plt.text(x[j], -y[j], str(j), fontsize=8, color="red")
    
    plt.title(f"Sample {i}")
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.xlim(0, 1)
    plt.ylim(-1, 0)
    plt.grid()
    plt.show()
