import numpy as np

c1 = np.random.normal(0, 1, (5, 5))

np.where(c1 > 0.09, c1**2, 42)

print(c1[:, 3])
