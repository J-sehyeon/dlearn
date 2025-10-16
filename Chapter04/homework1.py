import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# pre-processing

df = pd.read_csv('Chapter04/hw1_data.csv', sep="\s+", header=None, names=['x', 'y'])

plt.plot(df['x'], df['y'], marker="o")

X, y = df["x"].to_numpy().reshape(-1, 1), df["y"].to_numpy().reshape(-1, 1)     # X (20, 1) ,y (20, 1)


# problem function

def p2_ftn(x, w):
    """
    Args:
        x (scalar (N, 1)) : input
        w (ndarray (1, 3)) : weights
    Returns:
        X (ndarray (N, 3))
    """
    X = x ** np.arange(3)
    return X @ w.T


def pw_ftn_diff(x):
    """
    Returns:
        (N, 3)
    """
    return np.array([np.ones_like(x), x, x**2])


# loss

class Loss:
    def forward():
        raise NotImplementedError
    def backward():
        raise NotImplementedError

class Least_square(Loss):
    def __init__(self, n, function, diff):
        self.cost = 0
        self.function = function
        self.diff = diff
        self.N = n      # x.shape[0]
    def forward(self, x, w, y):
        return np.sum((self.function(x, w) - y) ** 2) / (2 * self.N)
    def backward(self, x, w, y):
        return np.sum((self.function(x, w) - y) * self.diff(x), axis=1).reshape(1, 3) / self.N   # (, 3)


def chi_square(x, y, std):
    pass


# backpropagation

def gb(w, gradient, lr=0.01):
    w -= lr * gradient
    return w



# Network

iterations = 10

w = np.array([[10, -10, -5]], dtype=float)   # w (1, 3)

loss_l = Least_square(X.shape[0], p2_ftn, pw_ftn_diff)

for i in range(iterations):
    y_pred = p2_ftn(X, w)
    loss_l.cost += loss_l.forward(X, w, y)
    grad = loss_l.backward(X, w, y)


    w = gb(w, grad)

print(w)

plt.plot(X, p2_ftn(X, w), marker="o")
plt.show()