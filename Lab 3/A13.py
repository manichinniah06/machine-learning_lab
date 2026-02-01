# Question-A13

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression


path = r"C:\Users\manic\OneDrive\Desktop\Amrita Documents\Semester 4\Machine Learning\Lab 3\Lab Session Data.xlsx"
data = pd.read_excel(path, "Purchase data")


X = data.iloc[:, 1:4].values
y = np.where(data.iloc[:, 4].values < 250, 0, 1)


X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42
)


model = LogisticRegression()
model.fit(X_train, y_train)


y_test_pred = model.predict(X_test)


def confusion_matrix_custom(y_true, y_pred):
    TP = FP = TN = FN = 0
    for i in range(len(y_true)):
        if y_true[i] == 1 and y_pred[i] == 1:
            TP += 1
        elif y_true[i] == 0 and y_pred[i] == 1:
            FP += 1
        elif y_true[i] == 0 and y_pred[i] == 0:
            TN += 1
        elif y_true[i] == 1 and y_pred[i] == 0:
            FN += 1
    return TP, FP, TN, FN


def accuracy_custom(TP, FP, TN, FN):
    return (TP + TN) / (TP + TN + FP + FN)

def precision_custom(TP, FP):
    return TP / (TP + FP) if (TP + FP) != 0 else 0

def recall_custom(TP, FN):
    return TP / (TP + FN) if (TP + FN) != 0 else 0

def fbeta_score_custom(precision, recall, beta):
    if precision + recall == 0:
        return 0
    return (1 + beta**2) * (precision * recall) / ((beta**2 * precision) + recall)


TP, FP, TN, FN = confusion_matrix_custom(y_test, y_test_pred)

print("Custom Confusion Matrix Values:")
print("TP:", TP)
print("FP:", FP)
print("TN:", TN)
print("FN:", FN)

accuracy = accuracy_custom(TP, FP, TN, FN)
precision = precision_custom(TP, FP)
recall = recall_custom(TP, FN)
f1_score_custom = fbeta_score_custom(precision, recall, beta=1)

print("\nCustom Performance Metrics:")
print("Accuracy:", accuracy)
print("Precision:", precision)
print("Recall:", recall)
print("F1-Score:", f1_score_custom)
