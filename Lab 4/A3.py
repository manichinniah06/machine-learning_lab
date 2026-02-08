import numpy as np
import matplotlib.pyplot as plt

def generate_data():
    np.random.seed(42)
    X = np.random.randint(1, 11, 20)
    Y = np.random.randint(1, 11, 20)
    labels = np.where(X + Y > 10, 1, 0)
    return X, Y, labels

def plot_data(X, Y, labels):
    colors = ["blue" if l == 0 else "red" for l in labels]
    plt.scatter(X, Y, c=colors)
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.show()

def main():
    X, Y, labels = generate_data()
    plot_data(X, Y, labels)

if __name__ == "__main__":
    main()
