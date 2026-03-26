import pandas as pd
import numpy as np
from collections import Counter

df = pd.read_csv(r"C:\Users\manic\OneDrive\Desktop\Amrita Documents\Semester 4\Machine Learning\Lab\f1_deviation_dataset_2022_2024.csv")

def equal_width_binning(column, bins=4):
    min_val = column.min()
    max_val = column.max()
    width = (max_val - min_val) / bins
    
    binned = np.floor((column - min_val) / width)
    binned[binned == bins] = bins - 1
    
    return binned.astype(int)

def entropy(y):
    counts = Counter(y)
    total = len(y)
    
    ent = 0
    for count in counts.values():
        p = count / total
        ent -= p * np.log2(p)
    
    return ent

target_col = df.columns[-1]

if df[target_col].dtype != 'object':
    df[target_col] = equal_width_binning(df[target_col], bins=4)

print("Entropy:", entropy(df[target_col]))