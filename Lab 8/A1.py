import numpy as np

def summation(x, w):
    return np.dot(x, w)

def step(x):
    return 1 if x >= 0 else 0

def bipolar_step(x):
    return 1 if x >= 0 else -1

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def tanh(x):
    return np.tanh(x)

def relu(x):
    return max(0, x)

def leaky_relu(x):
    return x if x > 0 else 0.01 * x

def error(y, t):
    return (t - y) ** 2


x = np.array([1, 2, 3])
w = np.array([0.5, -1, 2])

y_in = summation(x, w)

print("Summation:", y_in)
print("Step:", step(y_in))
print("Bipolar Step:", bipolar_step(y_in))
print("Sigmoid:", sigmoid(y_in))
print("TanH:", tanh(y_in))
print("ReLU:", relu(y_in))
print("Leaky ReLU:", leaky_relu(y_in))
print("Error:", error(step(y_in), 1))