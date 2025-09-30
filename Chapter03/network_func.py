import numpy as np

def step(x):
    return int(x > 0)

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def relu(x):
    return np.maximum(0, x)

def identity(x):
    return x

def softmax(x, overflow=False):
    if overflow: x - np.max(x)
    return np.exp(x) / np.sum(np.exp(x))