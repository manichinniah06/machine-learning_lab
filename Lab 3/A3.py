# Question-A3
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

path = r"C:\Users\manic\OneDrive\Desktop\Amrita Documents\Semester 4\Machine Learning\Lab 3\Lab Session Data.xlsx"
data = pd.read_excel(path, "Purchase data")

feature = data.iloc[:, 1].values

hist = np.histogram(feature)
mean = np.mean(feature)
variance = np.var(feature)

plt.hist(feature)
plt.show()

print(hist)
print(mean)
print(variance)
