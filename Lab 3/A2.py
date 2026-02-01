# Question-A2
import pandas as pd
import numpy as np

path = r"C:\Users\manic\OneDrive\Desktop\Amrita Documents\Semester 4\Machine Learning\Lab 3\Lab Session Data.xlsx"
data = pd.read_excel(path, "Purchase data")

X = data.iloc[:, 1:4].values
payment = data.iloc[:, 4].values

class1 = X[payment < 250]
class2 = X[payment >= 250]

centroid1 = class1.mean(axis=0)
centroid2 = class2.mean(axis=0)

spread1 = class1.std(axis=0)
spread2 = class2.std(axis=0)

distance = np.linalg.norm(centroid1 - centroid2)

print(centroid1)
print(spread1)
print(centroid2)
print(spread2)
print(distance)
