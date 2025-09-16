import numpy as np
import matplotlib.pyplot as plt

def AND_raw(x1, x2):
    """
    w1, w2, theta = 0.5, 0.5, 0.7
    """
    tmp = x1*0.5 + x2*0.5
    return int(tmp > 0.7)

def check_P(function):
    for i in range(2):
        for j in range(2):
            print(f'{function(i, j)} ({i}, {j})')
    print("---------------------------")
            

check_P(AND_raw)


"""
y = 0 if b + np.dot(x, w) <= 0
elif y = 1
"""

def AND(x1, x2):
    x = np.array([x1, x2])
    w = np.array([0.5, 0.5])
    b = -0.7
    tmp = np.dot(x, w) + b
    if tmp <= 0:
        return 0
    else:
        return 1

check_P(AND)

def NAND(x1, x2):
    return [1, 0][AND(x1, x2)]

check_P(NAND)

def OR(x1, x2):
    x = np.array([x1, x2])
    w = np.array([0.4, 0.4])
    b = -0.7
    
    tmp = np.dot(x, w) + b
    return int(tmp > 0)

check_P(OR)

w1, w2, b = (0.5, 0.5, -0.2)
x = np.arange(0, 2, 0.1)
y1 = -(w1*x + b) / w2

plt.plot(x, y1, label="perceptron")
plt.plot([0, 0, 1, 1], [0, 1, 0, 1], marker="o", linestyle="none")

plt.xlabel("x")
plt.ylabel("y")
plt.legend()
plt.show()