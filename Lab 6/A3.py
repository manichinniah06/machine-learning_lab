import pandas as pd
import numpy as np
from collections import Counter

df = pd.read_csv(r"C:\Users\manic\OneDrive\Desktop\Amrita Documents\Semester 4\Machine Learning\Lab\f1_deviation_dataset_2022_2024.csv")

def entropy(y):
    counts = Counter(y)
    total = len(y)
    return -sum((c/total)*np.log2(c/total) for c in counts.values())

def information_gain(df, feature, target):
    total_entropy = entropy(df[target])
    
    values = df[feature].unique()
    weighted_entropy = 0
    
    for val in values:
        subset = df[df[feature] == val]
        weight = len(subset) / len(df)
        weighted_entropy += weight * entropy(subset[target])
    
    return total_entropy - weighted_entropy

def find_best_feature(df, target):
    features = [col for col in df.columns if col != target]
    
    ig_values = {}
    for feature in features:
        ig_values[feature] = information_gain(df, feature, target)
    
    best_feature = max(ig_values, key=ig_values.get)
    
    print("Information Gain:", ig_values)
    return best_feature

target_col = df.columns[-1]
print("Root Node:", find_best_feature(df, target_col))