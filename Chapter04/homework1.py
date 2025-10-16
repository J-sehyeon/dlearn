import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# pre-processing

df = pd.read_csv('Chapter04/hw1_data.csv', sep="\s+", header=None, names=['x', 'y'])
"""
print(df)
plt.plot(df['x'], df['y'], marker="o")
plt.show()
"""
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


def pw_ftn_diff(x, n):
    return w[n]*x**n


# loss

class Loss:
    def forward():
        raise NotImplementedError
    def backward():
        raise NotImplementedError

class least_square(Loss):
    def __init__(self, n, function, diff):
        self.cost = 0
        self.function = function
        self.diff = diff
        self.N = n      # x.shape[0]
    def forward(self, x, y):
        return np.sum((self.function(x) - y) ** 2) / (2 * self.N)
    def backward(self, x, y):
        return np.sum((self.function(x) - y)) * self.diff(x) / self.N

def chi_square(x, y, std):
    pass


# backpropagation





# Network

iterations = 10

w = np.array([[10, -10, -5]])   # w (1, 3)

for i in range(iterations):
    for idx in range(X.shape[0]):
        pass