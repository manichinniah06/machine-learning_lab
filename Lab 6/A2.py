import pandas as pd
from collections import Counter

df = pd.read_csv(r"C:\Users\manic\OneDrive\Desktop\Amrita Documents\Semester 4\Machine Learning\Lab\f1_deviation_dataset_2022_2024.csv")

def gini_index(y):
    counts = Counter(y)
    total = len(y)
    
    gini = 1
    for count in counts.values():
        p = count / total
        gini -= p ** 2
    
    return gini

target_col = df.columns[-1]

print("Gini Index:", gini_index(df[target_col]))