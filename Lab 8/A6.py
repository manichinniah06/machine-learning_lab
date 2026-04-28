import numpy as np

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

X = np.array([
[20,6,2,386],
[16,3,6,289],
[27,6,2,393],
[19,1,2,110],
[24,4,2,280],
[22,1,5,167],
[15,4,2,271],
[18,4,2,274],
[21,1,4,148],
[16,2,4,198]
])

T = np.array([1,1,1,0,1,0,1,1,0,0])

w = np.random.rand(4)
lr = 0.01

for epoch in range(1000):
    total_error = 0
    for i in range(len(X)):
        y = sigmoid(np.dot(X[i], w))
        e = T[i] - y
        total_error += e**2
        w = w + lr * e * X[i]
    if total_error <= 0.002:
        break

print("Weights:", w)