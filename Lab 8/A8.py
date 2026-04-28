import numpy as np

def sigmoid(x):
    return 1/(1+np.exp(-x))

X = np.array([[0,0],[0,1],[1,0],[1,1]])
T = np.array([[0],[0],[0],[1]])

w1 = np.random.rand(2,2)
w2 = np.random.rand(2,1)
lr = 0.05

for epoch in range(1000):
    h = sigmoid(np.dot(X, w1))
    o = sigmoid(np.dot(h, w2))
    error = T - o
    if np.mean(error**2) <= 0.002:
        break
    d2 = error * o * (1-o)
    d1 = h * (1-h) * np.dot(d2, w2.T)
    w2 += lr * np.dot(h.T, d2)
    w1 += lr * np.dot(X.T, d1)

print("Output:", o)