import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # 3Dプロット用

# CSVファイルの読み込み
df = pd.read_csv("hand_landmarks.csv")  # 適宜変更

# x, y, z 座標のカラム名を定義
num_landmarks = 21  # 手のランドマーク数
x_columns = [f"x{i}" for i in range(num_landmarks)]
y_columns = [f"y{i}" for i in range(num_landmarks)]
z_columns = [f"z{i}" for i in range(num_landmarks)]  # 修正

# 複数のサンプルを3Dプロット
for i in range(min(5, len(df))):  # 最初の5サンプル
    x = df.loc[i, x_columns]
    y = df.loc[i, y_columns]
    z = df.loc[i, z_columns]

    fig = plt.figure(figsize=(6, 6))  # 3Dプロットのための figure 作成
    ax = fig.add_subplot(111, projection="3d")

    ax.scatter(x, -y, z, c="blue")  # y軸を反転（画像座標系に合わせる）

    # 各ポイントに番号を表示
    for j in range(num_landmarks):
        ax.text(x[j], -y[j], z[j], str(j), fontsize=8, color="red")

    ax.set_title(f"Sample {i}")
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")  # 修正
    ax.set_xlim(0, 1)
    ax.set_ylim(-1, 0)
    ax.set_zlim(0, 1)  # 修正
    plt.show()
