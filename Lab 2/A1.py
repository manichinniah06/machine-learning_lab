import pandas as pd
import numpy as np

path = r"C:\Users\manic\OneDrive\Desktop\Amrita Documents\Semester 4\Machine Learning\Lab 2\Lab Session Data.xlsx"

df = pd.read_excel(path, sheet_name="Purchase data")
X = df[["Candies (#)","Mangoes (Kg)","Milk Packets (#)"]].values
y = df["Payment (Rs)"].values.reshape(-1,1)

rank_X = np.linalg.matrix_rank(X)
print("Rank of X : ",rank_X)

pinv_X = np.linalg.pinv(X)
c = pinv_X @ y

print("Cost per Candy:", c[0][0])
print("Cost per Kg Mango:", c[1][0])
print("Cost per Milk Packet:", c[2][0])