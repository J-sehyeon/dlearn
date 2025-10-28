import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# pre-processing

df = pd.read_csv('Chapter04/hw1_data.csv', sep="\s+", header=None, names=['x', 'y'])

plt.plot(df['x'], df['y'], marker="o")

X, y = df["x"].to_numpy().reshape(-1, 1), df["y"].to_numpy().reshape(-1, 1)     # X (20, 1) ,y (20, 1)


# define model and predict ftn

def predict(X, w):
    return w[0] + w[1] * X + w[2] * X**2

def compute_loss(y_pred, y_true):
    return np.mean((y_pred - y_true) ** 2)

def compute_gradients(X, y_true, w):
    y_pred = predict(X, w)
    error = y_pred - y_true

    grad_w0 = np.mean(2 * error)
    grad_w1 = np.mean(2 * error * X)
    grad_w2 = np.mean(2 * error * X**2)

    return np.array([grad_w0, grad_w1, grad_w2])

def train_model(X, y, learning_rate=0.01, epochs=10000):
    
    # 매개변수 초기화
    np.random.seed(42)
    w = np.random.rand(3) * 0.1

    history = []

    for epoch in range(epochs):
        # 예측 및 손실 계산
        y_pred = predict(X, w)
        loss = compute_loss(y_pred, y)

        # 기울기 계산
        grad = compute_gradients(X, y, w)

        # 매개변수 업데이트 (경사 하강법)
        w -= learning_rate * grad
        
        history.append(loss)

        if (epoch) % 200 == 0:
            print(f"Epoch {epoch:4f}/{epochs} | Loss: {loss:.4f} | w0: {w[0]:.4f}, w1:{w[1]:.4f}, w2: {w[2]:.4f}")
        
    return w, history
    
print("--- 학습 시작 ---")

w, history = train_model(X, y)

print(w)

y = w[0] + w[1] * X + w[2] * X**2

plt.plot(X, y, marker="x")

plt.show()