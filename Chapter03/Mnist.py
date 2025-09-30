import sys, os

parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
sys.path.append(parent_dir)
print("\n--- sys.path에 추가된 경로 ---")
print(parent_dir)

import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from dataset.mnist import load_mnist
import pickle
import network_func as nf


(x_train, t_train), (x_test, t_test) = \
    load_mnist(flatten=True, normalize=True)

print(x_train.shape)
print(x_test.shape)

#plt.imshow(x_train[0].reshape(28, 28))
#plt.show()
d = parent_dir + "/deep-learning-from-scratch/ch03/sample_weight.pkl"
def init_network():
    with open(d, "rb") as f:
        return pickle.load(f)

def predict(network, x):
    w1, w2, w3 = network["W1"], network["W2"], network["W3"]
    b1, b2, b3 = network["b1"], network["b2"], network["b3"]

    a1 = np.dot(x, w1) + b1
    z1 = nf.sigmoid(a1)
    a2 = np.dot(z1, w2) + b2
    z2 = nf.sigmoid(a2)
    a3 = np.dot(z2, w3) + b3
    y = nf.softmax(a3)

    return y

network = init_network()

accuracy_count = 0

for i in range(len(x_test)):
    y = predict(network, x_test[i])
    p = np.argmax(y)
    if p == t_test[i]:
        accuracy_count += 1

print("Accuracy : %s" % (accuracy_count / len(x_test)))