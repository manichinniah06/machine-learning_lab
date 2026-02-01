# Question-A4
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def minkowski_dist(x, y, p):
    s = 0
    for i in range(len(x)):
        s += abs(x[i] - y[i]) ** p
    return s ** (1 / p)

path = r"C:\Users\manic\OneDrive\Desktop\Amrita Documents\Semester 4\Machine Learning\Lab 3\Lab Session Data.xlsx"
data = pd.read_excel(path, "Purchase data")

f1 = data.iloc[:, 2].values
f2 = data.iloc[:, 3].values

p_vals = range(1, 11)
dist_vals = [minkowski_dist(f1, f2, p) for p in p_vals]

plt.plot(p_vals, dist_vals)
plt.show()
