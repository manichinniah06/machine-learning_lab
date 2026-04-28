import numpy as np

def step(x):
    return 1 if x >= 0 else 0

X = np.array([[1,0,0],[1,0,1],[1,1,0],[1,1,1]])
T = np.array([0,1,1,0])

w = np.array([10,0.2,-0.75])
lr = 0.05

for epoch in range(1000):
    total_error = 0
    for i in range(len(X)):
        y = step(np.dot(X[i], w))
        e = T[i] - y
        total_error += e**2
        w = w + lr * e * X[i]
    if total_error <= 0.002:
        break

print("Final Error:", total_error)
print("Weights:", w)