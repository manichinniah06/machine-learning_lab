import numpy as np
import matplotlib.pyplot as plt

def step(x):
    return 1 if x >= 0 else 0

def train(X, T, w, lr):
    for epoch in range(1000):
        total_error = 0
        for i in range(len(X)):
            y_in = np.dot(X[i], w)
            y = step(y_in)
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

initial_w = np.array([10, 0.2, -0.75])

rates = [0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1]
epochs_list = []

for lr in rates:
    epochs = train(X, T, initial_w.copy(), lr)
    epochs_list.append(epochs)

print("Learning Rates:", rates)
print("Epochs:", epochs_list)

plt.plot(rates, epochs_list)
plt.xlabel("Learning Rate")
plt.ylabel("Epochs to Converge")
plt.title("Learning Rate vs Epochs")
plt.show()