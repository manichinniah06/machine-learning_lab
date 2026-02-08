import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier

def load_data():
    data = pd.read_excel(r"C:\Users\manic\OneDrive\Desktop\Amrita Documents\Semester 4\Machine Learning\Lab 4\Lab Session Data.xlsx")
    target = data.iloc[:, -1]
    labels = np.where(target > target.median(), 1, 0)
    features = data.iloc[:, 1:3].values
    return features, labels

def plot_training_data(X, y):
    colors = ["blue" if l == 0 else "red" for l in y]
    plt.scatter(X[:, 0], X[:, 1], c=colors)
    plt.xlabel("Feature 1")
    plt.ylabel("Feature 2")
    plt.show()

def plot_knn_regions(X, y, k):
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1

    xx, yy = np.meshgrid(
        np.arange(x_min, x_max, 0.1),
        np.arange(y_min, y_max, 0.1)
    )

    test_data = np.c_[xx.ravel(), yy.ravel()]
    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(X, y)
    pred = knn.predict(test_data)

    colors_test = ["blue" if p == 0 else "red" for p in pred]
    colors_train = ["blue" if l == 0 else "red" for l in y]

    plt.scatter(test_data[:, 0], test_data[:, 1], c=colors_test, s=10, alpha=0.4)
    plt.scatter(X[:, 0], X[:, 1], c=colors_train, s=80, edgecolors="black")
    plt.xlabel("Feature 1")
    plt.ylabel("Feature 2")
    plt.title(f"k = {k}")
    plt.show()

def main():
    X, y = load_data()
    plot_training_data(X, y)
    for k in [3, 5, 7]:
        plot_knn_regions(X, y, k)

if __name__ == "__main__":
    main()
