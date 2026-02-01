# Question-A6
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

path = r"C:\Users\manic\OneDrive\Desktop\Amrita Documents\Semester 4\Machine Learning\Lab 3\Lab Session Data.xlsx"
data = pd.read_excel(path, "Purchase data")

X = data.iloc[:, 1:4].values
y = np.where(data.iloc[:, 4].values < 250, 0, 1)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

print(X_train)
print(X_test)
print(y_train)
print(y_test)
