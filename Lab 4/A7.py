import numpy as np
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import GridSearchCV

def generate_training_data():
    np.random.seed(42)
    X = np.random.randint(1, 11, 20)
    Y = np.random.randint(1, 11, 20)
    labels = np.where(X + Y > 10, 1, 0)
    return X, Y, labels

def find_best_k(train_data, labels):
    param_grid = {'n_neighbors': list(range(1, 16))}
    knn = KNeighborsClassifier()
    grid = GridSearchCV(knn, param_grid, cv=5, scoring='accuracy')
    grid.fit(train_data, labels)
    return grid.best_params_['n_neighbors'], grid.best_score_

def plot_best_knn(X, Y, labels, k):
    train_data = np.column_stack((X, Y))
    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(train_data, labels)

    x = np.arange(0, 10.1, 0.1)
    y = np.arange(0, 10.1, 0.1)
    xx, yy = np.meshgrid(x, y)
    test_data = np.c_[xx.ravel(), yy.ravel()]
    pred = knn.predict(test_data)

    colors_test = ["blue" if p == 0 else "red" for p in pred]
    colors_train = ["blue" if l == 0 else "red" for l in labels]

    plt.scatter(test_data[:, 0], test_data[:, 1], c=colors_test, s=10, alpha=0.4)
    plt.scatter(X, Y, c=colors_train, s=80, edgecolors="black")
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.show()

def main():
    X, Y, labels = generate_training_data()
    train_data = np.column_stack((X, Y))
    best_k, best_score = find_best_k(train_data, labels)
    print(best_k)
    print(best_score)
    plot_best_knn(X, Y, labels, best_k)

if __name__ == "__main__":
    main()
