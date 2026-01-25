import pandas as pd
import numpy as np

path = r"C:\Users\manic\OneDrive\Desktop\Amrita Documents\Semester 4\Machine Learning\Lab 2\Lab Session Data.xlsx"
df = pd.read_excel(path, sheet_name="thyroid0387_UCI")

print("Dataset Shape (rows, cols):", df.shape)
print("\nFirst 5 rows:\n", df.head())

print(df.info())

print(df.columns.tolist())

for col in df.columns:
    unique_count = df[col].nunique(dropna=True)
    if unique_count <= 15:  
        print(f"\nColumn: {col}")
        print("Unique values:", df[col].dropna().unique())


cat_cols = df.select_dtypes(include=["object", "category", "bool"]).columns.tolist()
num_cols = df.select_dtypes(include=[np.number]).columns.tolist()

print("\nCategorical Columns:", cat_cols)
print("Numeric Columns:", num_cols)

print("\nFor NOMINAL categorical variables -> One-Hot Encoding")
print("For ORDINAL categorical variables -> Label Encoding")

print("\nCategorical columns found in your dataset:")
for col in cat_cols:
    print(f"- {col}  (Unique values: {df[col].nunique(dropna=True)})")

print("\nNOTE: To decide ordinal vs nominal, check if values have natural order.")
print("Example: Low < Medium < High -> ordinal")
print("Example: Male/Female, Yes/No -> nominal")

if len(num_cols) > 0:
    numeric_range = df[num_cols].agg(["min", "max"])
    print(numeric_range)
else:
    print("No numeric columns found.")

missing_count = df.isna().sum()
missing_percent = (missing_count / len(df)) * 100

missing_table = pd.DataFrame({
    "Missing Count": missing_count,
    "Missing %": missing_percent
}).sort_values(by="Missing Count", ascending=False)

print(missing_table)


def count_outliers_iqr(series):
    series = series.dropna()
    q1 = np.percentile(series, 25)
    q3 = np.percentile(series, 75)
    iqr = q3 - q1
    lower = q1 - 1.5 * iqr
    upper = q3 + 1.5 * iqr
    outliers = series[(series < lower) | (series > upper)]
    return len(outliers), lower, upper

if len(num_cols) > 0:
    for col in num_cols:
        out_count, lower, upper = count_outliers_iqr(df[col])
        print(f"{col}: Outliers = {out_count}, Lower Bound = {lower:.3f}, Upper Bound = {upper:.3f}")
else:
    print("No numeric columns found for outlier detection.")


if len(num_cols) > 0:
    stats_table = pd.DataFrame({
        "Mean": df[num_cols].mean(),
        "Variance": df[num_cols].var(),     
        "Std Dev": df[num_cols].std()
    })
    print(stats_table)
else:
    print("No numeric columns found.")
