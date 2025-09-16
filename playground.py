import numpy as np
import matplotlib.pyplot as plt

x = np.arange(1, 6, 0.1)
def f(x):
    return 1 / 2 *(x - 3) ** 2
y = f(x)

np.random.seed(42)

x1, w, b = np.random.rand(3)
y_b = x1 * w + b

grad = (x1 - 3) * w
alpha = 2

w_n = w - alpha * grad
y_n = x1 * w_n + b


plt.figure(figsize=(3, 2))

plt.plot(x, y)
plt.plot([y_b, y_n], [f(y_b), f(y_n)], marker="o", linestyle="none")

plt.xlabel("x")
plt.ylabel("y")

plt.show()  