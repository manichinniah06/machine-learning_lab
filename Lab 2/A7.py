import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from itertools import combinations

path = r"C:\Users\manic\OneDrive\Desktop\Amrita Documents\Semester 4\Machine Learning\Lab 2\Lab Session Data.xlsx"
df_3 = pd.read_excel(path, sheet_name="thyroid0387_UCI")

obj_cols = df_3.select_dtypes(include=["object"]).columns
df_3[obj_cols] = df_3[obj_cols].replace("?", np.nan)


df_features = df_3.drop(columns=["Record ID", "Condition"], errors="ignore")


if "sex" in df_features.columns:
    df_features["sex"] = df_features["sex"].map({"F": 1, "M": 0})


for col in df_features.columns:
    if df_features[col].dtype == "object":
        unique_vals = set(df_features[col].dropna().unique())
        if unique_vals.issubset({"t", "f"}):
            df_features[col] = df_features[col].map({"t": 1, "f": 0})


df_features = df_features.apply(pd.to_numeric, errors="coerce")


df_features = df_features.fillna(0)


num_df = df_features.iloc[:20]


binary_df = (num_df > 0).astype(int)


def J_C(x, y):
    f11 = np.sum((x == 1) & (y == 1))
    f10 = np.sum((x == 1) & (y == 0))
    f01 = np.sum((x == 0) & (y == 1))
    denom = f11 + f10 + f01
    return f11 / denom if denom != 0 else 0


def S_M_C(x, y):
    f11 = np.sum((x == 1) & (y == 1))
    f00 = np.sum((x == 0) & (y == 0))
    f10 = np.sum((x == 1) & (y == 0))
    f01 = np.sum((x == 0) & (y == 1))
    total = f11 + f00 + f10 + f01
    return (f11 + f00) / total if total != 0 else 0


def C_O_S(x, y):
    x = np.array(x, dtype=float)
    y = np.array(y, dtype=float)
    denom = (np.linalg.norm(x) * np.linalg.norm(y))
    return np.dot(x, y) / denom if denom != 0 else 0

results = []

for i, j in combinations(range(20), 2):
    results.append({
        "Vector Pair": f"({i+1}, {j+1})",
        "Jaccard (JC)": J_C(binary_df.iloc[i].values, binary_df.iloc[j].values),
        "SMC": S_M_C(binary_df.iloc[i].values, binary_df.iloc[j].values),
        "Cosine (COS)": C_O_S(num_df.iloc[i].values, num_df.iloc[j].values)
    })

similarity_df = pd.DataFrame(results)
print(similarity_df.head(10))


n = 20
JC = np.zeros((n, n))
SMC = np.zeros((n, n))
COS = np.zeros((n, n))

for i in range(n):
    for j in range(n):
        JC[i, j] = J_C(binary_df.iloc[i].values, binary_df.iloc[j].values)
        SMC[i, j] = S_M_C(binary_df.iloc[i].values, binary_df.iloc[j].values)
        COS[i, j] = C_O_S(num_df.iloc[i].values, num_df.iloc[j].values)


plt.figure(figsize=(8, 6))
sns.heatmap(JC, cmap="Blues", annot=True, fmt=".2f")
plt.title("Jaccard Coefficient Heatmap")
plt.xlabel("Vector Index")
plt.ylabel("Vector Index")
plt.show()

plt.figure(figsize=(8, 6))
sns.heatmap(SMC, cmap="Greens", annot=True, fmt=".2f")
plt.title("Simple Matching Coefficient Heatmap")
plt.xlabel("Vector Index")
plt.ylabel("Vector Index")
plt.show()

plt.figure(figsize=(8, 6))
sns.heatmap(COS, cmap="Reds", annot=True, fmt=".2f")
plt.title("Cosine Similarity Heatmap")
plt.xlabel("Vector Index")
plt.ylabel("Vector Index")
plt.show()
