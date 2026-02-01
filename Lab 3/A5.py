# Question-A5
import pandas as pd
from scipy.spatial.distance import minkowski

def minkowski_dist(x, y, p):
    s = 0
    for i in range(len(x)):
        s += abs(x[i] - y[i]) ** p
    return s ** (1 / p)

path = r"C:\Users\manic\OneDrive\Desktop\Amrita Documents\Semester 4\Machine Learning\Lab 3\Lab Session Data.xlsx"
data = pd.read_excel(path, "Purchase data")

x = data.iloc[:, 2].values
y = data.iloc[:, 3].values

print(minkowski_dist(x, y, 2))
print(minkowski(x, y, 2))
