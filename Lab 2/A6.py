import pandas as pd
import numpy as np

path = r"C:\Users\manic\OneDrive\Desktop\Amrita Documents\Semester 4\Machine Learning\Lab 2\Lab Session Data.xlsx"
df_3 = pd.read_excel(path, sheet_name="thyroid0387_UCI")

obj_cols = df_3.select_dtypes(include="object").columns
for col in obj_cols:
    df_3[col] = df_3[col].mask(df_3[col] == "?", np.nan)

for col in obj_cols:
    unique_vals = set(df_3[col].dropna().unique())
    if unique_vals.issubset({"t", "f"}):
        df_3[col] = df_3[col].map({"t": 1, "f": 0})

df_3_encoded = pd.get_dummies(df_3, drop_first=False)

for col in df_3_encoded.columns:
    if df_3_encoded[col].dtype != "uint8":
        df_3_encoded[col] = df_3_encoded[col].fillna(df_3_encoded[col].mean())

A = df_3_encoded.iloc[0].to_numpy(dtype=float)
B = df_3_encoded.iloc[1].to_numpy(dtype=float)

cos_sim = np.dot(A, B) / (np.linalg.norm(A) * np.linalg.norm(B))

print("Cosine Similarity:", cos_sim)
