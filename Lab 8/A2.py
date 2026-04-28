import numpy as np
import matplotlib.pyplot as plt

X = np.array([[1,0,0],[1,0,1],[1,1,0],[1,1,1]])
T = np.array([0,0,0,1])

w = np.array([10,0.2,-0.75])
lr = 0.05

def step(x):
    return 1 if x >= 0 else 0

errors = []

for epoch in range(1000):
    total_error = 0
    for i in range(len(X)):
        y_in = np.dot(X[i], w)
        y = step(y_in)
        e = T[i] - y
        total_error += e**2
        w = w + lr * e * X[i]
    errors.append(total_error)
    if total_error <= 0.002:
        break

plt.plot(errors)
plt.show()