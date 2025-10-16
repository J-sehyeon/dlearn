import numpy as np

a = np.ones((5, 1))
print(a)

b = np.array([np.ones_like(a), a])
print(b)