import pandas as pd

# CSVファイルを読み込む
data = pd.read_csv('hand_landmarks.csv')

# 最初の数行を表示してデータを確認
print(data.head())

# 'label'列をターゲット変数 (y) として、残りの列を特徴量 (X) として分割
X = data.drop(columns=['label'])
y = data['label']

from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import joblib

# 特徴量の標準化
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

from sklearn.decomposition import PCA

# PCAを使って次元削減
pca = PCA(n_components=10)  # 主成分数を適宜調整
X_reduced = pca.fit_transform(X_scaled)

# データをトレーニングセットとテストセットに分割
X_train, X_test, y_train, y_test = train_test_split(X_reduced, y, test_size=0.2, random_state=42)

# モデルのインスタンスを作成
model = RandomForestClassifier(random_state=42)

# モデルの学習
model.fit(X_train, y_train)

# テストデータを使って予測
y_pred = model.predict(X_test)

# 精度を計算
accuracy = accuracy_score(y_test, y_pred)
print(f'Accuracy: {accuracy * 100:.2f}%')

# モデルを保存
joblib.dump(model, 'model.pkl')


from sklearn.model_selection import GridSearchCV

param_grid = {
    'n_estimators': [100, 200, 300],
    'max_depth': [10, 20, None],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4]
}

grid_search = GridSearchCV(estimator=RandomForestClassifier(random_state=42), param_grid=param_grid, cv=5)
grid_search.fit(X_train, y_train)

# 最適なパラメータ
print(f'Best parameters: {grid_search.best_params_}')

best_model = RandomForestClassifier(
    n_estimators=300,
    max_depth=10,
    min_samples_split=2,
    min_samples_leaf=1,
    random_state=42
)

best_model.fit(X_train, y_train)
joblib.dump(best_model, 'best_model.pkl')

