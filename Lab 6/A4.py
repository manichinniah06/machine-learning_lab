import pandas as pd
import numpy as np

df = pd.read_csv(r"C:\Users\manic\OneDrive\Desktop\Amrita Documents\Semester 4\Machine Learning\Lab\f1_deviation_dataset_2022_2024.csv")

# Equal Width
def equal_width_binning(column, bins=4):
    min_val = column.min()
    max_val = column.max()
    width = (max_val - min_val) / bins
    
    binned = np.floor((column - min_val) / width)
    binned[binned == bins] = bins - 1
    
    return binned.astype(int)

# Equal Frequency
def equal_frequency_binning(column, bins=4):
    return pd.qcut(column, q=bins, labels=False, duplicates='drop')

# General Function
def bin_feature(column, bins=4, method="width"):
    if method == "width":
        return equal_width_binning(column, bins)
    elif method == "frequency":
        return equal_frequency_binning(column, bins)
    else:
        raise ValueError("Invalid method")

# Apply binning to all numeric columns
for col in df.columns:
    if df[col].dtype != 'object':
        df[col] = bin_feature(df[col], bins=4, method="width")

print(df.head())