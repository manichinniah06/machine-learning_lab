import pandas as pd
import numpy as np

path = r"C:\Users\manic\OneDrive\Desktop\Amrita Documents\Semester 4\Machine Learning\Lab 2\Lab Session Data.xlsx"
data_3 = pd.read_excel(path, sheet_name="thyroid0387_UCI")

rows_1_and_2 = data_3.iloc[0:2].copy()

binary_columns = []
for col in data_3.columns:
    unique_vals = set(data_3[col].dropna().unique())
    if unique_vals.issubset({'t', 'f', '?'}):
        binary_columns.append(col)

binary_information = rows_1_and_2[binary_columns].copy()

binary_information = binary_information.replace('?', np.nan)

for col in binary_information.columns:
    binary_information[col] = binary_information[col].map({'t': 1, 'f': 0})

binary_information = binary_information.dropna(axis=1)

vector_1 = binary_information.iloc[0].to_numpy(dtype=int)
vector_2 = binary_information.iloc[1].to_numpy(dtype=int)

f11 = f10 = f01 = f00 = 0

for i, j in zip(vector_1, vector_2):
    if i == 1 and j == 1:
        f11 += 1
    elif i == 1 and j == 0:
        f10 += 1
    elif i == 0 and j == 1:
        f01 += 1
    elif i == 0 and j == 0:
        f00 += 1

jaccard = f11 / (f11 + f10 + f01)
smc = (f11 + f00) / (f11 + f10 + f01 + f00)

print("Binary columns detected:", len(binary_columns))
print("Binary columns used (after removing missing):", binary_information.shape[1])
print("f11 =", f11, "f10 =", f10, "f01 =", f01, "f00 =", f00)

print("\nJaccard Coefficient:", jaccard)
print("Simple Matching Coefficient:", smc)
