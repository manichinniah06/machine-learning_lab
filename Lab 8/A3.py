import numpy as np

def step(x):
    return 1 if x >= 0 else 0

def bipolar_step(x):
    return 1 if x >= 0 else -1

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def relu(x):
    return max(0, x)

def train(X, T, w, lr, activation):
    for epoch in range(1000):
        total_error = 0
        for i in range(len(X)):
            y_in = np.dot(X[i], w)
            y = activation(y_in)
            e = T[i] - y
            total_error += e**2
            w = w + lr * e * X[i]
        if total_error <= 0.002:
            return epoch + 1
    return 1000

X = np.array([
    [1,0,0],
    [1,0,1],
    [1,1,0],
    [1,1,1]
])

T = np.array([0,0,0,1])

w_init = np.array([10,0.2,-0.75])
lr = 0.05

activations = [step, bipolar_step, sigmoid, relu]

for act in activations:
    epochs = train(X, T, w_init.copy(), lr, act)
    print(act.__name__, epochs)