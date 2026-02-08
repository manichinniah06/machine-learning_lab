import numpy as np
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier

def generate_training_data():
    np.random.seed(42)
    X = np.random.randint(1, 11, 20)
    Y = np.random.randint(1, 11, 20)
    labels = np.where(X + Y > 10, 1, 0)
    return X, Y, labels

def plot_knn(k):
    X, Y, labels = generate_training_data()
    train_data = np.column_stack((X, Y))

    x = np.arange(0, 10.1, 0.1)
    y = np.arange(0, 10.1, 0.1)
    xx, yy = np.meshgrid(x, y)
    test_data = np.c_[xx.ravel(), yy.ravel()]

    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(train_data, labels)
    pred = knn.predict(test_data)

    colors_test = ["blue" if p == 0 else "red" for p in pred]
    colors_train = ["blue" if l == 0 else "red" for l in labels]

    plt.scatter(test_data[:, 0], test_data[:, 1], c=colors_test, s=10, alpha=0.4)
    plt.scatter(X, Y, c=colors_train, s=80, edgecolors="black")
    plt.title(f"k = {k}")
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.show()

def main():
    for k in [1, 3, 5, 7]:
        plot_knn(k)

if __name__ == "__main__":
    main()
