import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import joblib

# ====== 1. 活性化関数 ======
def relu(x):
    return np.maximum(0, x)

def relu_derivative(x):
    return (x > 0).astype(float)

def softmax(x):
    exp_x = np.exp(x - np.max(x, axis=1, keepdims=True))  # 数値的安定性
    return exp_x / np.sum(exp_x, axis=1, keepdims=True)

# ====== 2. ネットワークの初期化 ======
class DeepNeuralNetwork:
    def __init__(self, input_size=40, hidden_sizes=[64, 64, 32, 32, 16], output_size=3, learning_rate=0.01):
        self.learning_rate = learning_rate
        
        # 重みとバイアスの初期化（He初期化）
        layer_sizes = [input_size] + hidden_sizes + [output_size]
        self.weights = [np.random.randn(layer_sizes[i], layer_sizes[i+1]) * np.sqrt(2.0/layer_sizes[i]) for i in range(len(layer_sizes) - 1)]
        self.biases = [np.zeros((1, layer_sizes[i+1])) for i in range(len(layer_sizes) - 1)]
    
    # ====== 3. 順伝播 ======
    def forward(self, X):
        self.activations = [X]  
        self.z_values = []  

        for i in range(len(self.weights) - 1):  
            z = np.dot(self.activations[-1], self.weights[i]) + self.biases[i]
            self.z_values.append(z)
            self.activations.append(relu(z))

        # 出力層
        z = np.dot(self.activations[-1], self.weights[-1]) + self.biases[-1]
        self.z_values.append(z)
        self.activations.append(softmax(z))
        return self.activations[-1]

    # ====== 4. 誤差逆伝播法 ======
    def backward(self, X, y):
        m = X.shape[0]  
        grads_w = [np.zeros_like(w) for w in self.weights]
        grads_b = [np.zeros_like(b) for b in self.biases]

        # One-hot エンコーディング
        y_onehot = np.eye(self.activations[-1].shape[1])[y.astype(int)]  
        dz = (self.activations[-1] - y_onehot) / m  

        grads_w[-1] = np.dot(self.activations[-2].T, dz)
        grads_b[-1] = np.sum(dz, axis=0, keepdims=True)

        for i in range(len(self.weights) - 2, -1, -1):
            dz = np.dot(dz, self.weights[i+1].T) * relu_derivative(self.z_values[i])
            grads_w[i] = np.dot(self.activations[i].T, dz)
            grads_b[i] = np.sum(dz, axis=0, keepdims=True)

        # パラメータ更新（SGD）
        for i in range(len(self.weights)):
            self.weights[i] -= self.learning_rate * grads_w[i]
            self.biases[i] -= self.learning_rate * grads_b[i]

    # ====== 5. 訓練 ======
    def train(self, X, y, epochs=1000, batch_size=32):
        for epoch in range(epochs):
            indices = np.arange(X.shape[0])
            np.random.shuffle(indices)
            X, y = X[indices], y[indices]

            for i in range(0, X.shape[0], batch_size):
                X_batch = X[i:i+batch_size]
                y_batch = y[i:i+batch_size]
                self.forward(X_batch)
                self.backward(X_batch, y_batch)

            if epoch % 100 == 0:
                predictions = self.predict(X)
                acc = np.mean(predictions == y)
                print(f'Epoch {epoch}, Accuracy: {acc:.4f}')
    
    # ====== 6. 予測メソッドの追加 ======
    def predict(self, X):
        probabilities = self.forward(X)
        return np.argmax(probabilities, axis=1)

# ====== 7. データの準備と学習 ======
if __name__ == "__main__":
    csv_file = 'hand_landmarks.csv'
    
    df = pd.read_csv(csv_file)
    
    X = df.iloc[:, 1:].values  # NumPy 配列に変換
    y = df['label'].values

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=40)

    print("X_train shape:", X_train.shape)
    print("X_test shape:", X_test.shape)


    model = DeepNeuralNetwork(42)
    model.train(X_train, y_train, epochs=1000, batch_size=32)

    y_pred = model.predict(X_test)
    
    accuracy = accuracy_score(y_test, y_pred)
    print(f'モデルの正解率: {accuracy:.2f}')
    
    joblib.dump(model, 'hand_gesture_model.pkl')
    print('モデルを保存しました。')
