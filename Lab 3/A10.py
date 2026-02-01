# Question-A10

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from collections import Counter
from math import sqrt

path = r"C:\Users\manic\OneDrive\Desktop\Amrita Documents\Semester 4\Machine Learning\Lab 3\Lab Session Data.xlsx"
data = pd.read_excel(path, "Purchase data")


X = data.iloc[:, 1:4].values
y = np.where(data.iloc[:, 4].values < 250, 0, 1)


X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42
)


neigh = KNeighborsClassifier(n_neighbors=3)
neigh.fit(X_train, y_train)

package_predictions = neigh.predict(X_test)
package_accuracy = neigh.score(X_test, y_test)

print("Package kNN Predictions:", package_predictions)
print("Package kNN Accuracy:", package_accuracy)



# Euclidean distance function
def euclidean_distance(x1, x2):
    distance = 0
    for i in range(len(x1)):
        distance += (x1[i] - x2[i]) ** 2
    return sqrt(distance)

# Custom kNN function
def knn_custom(X_train, y_train, test_vector, k):
    distances = []

    for i in range(len(X_train)):
        dist = euclidean_distance(X_train[i], test_vector)
        distances.append((dist, y_train[i]))

    distances.sort(key=lambda x: x[0])
    k_nearest_labels = [label for (_, label) in distances[:k]]

    predicted_class = Counter(k_nearest_labels).most_common(1)[0][0]
    return predicted_class

# Predict using custom kNN
custom_predictions = []
for test_vector in X_test:
    pred = knn_custom(X_train, y_train, test_vector, k=3)
    custom_predictions.append(pred)

custom_predictions = np.array(custom_predictions)

# Calculate accuracy manually
correct = np.sum(custom_predictions == y_test)
custom_accuracy = correct / len(y_test)

print("\nCustom kNN Predictions:", custom_predictions)
print("Custom kNN Accuracy:", custom_accuracy)

print("\nComparison of kNN Classifiers:")
print("Package kNN Accuracy:", package_accuracy)
print("Custom kNN Accuracy:", custom_accuracy)
