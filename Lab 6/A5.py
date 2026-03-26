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
    return max(features, key=lambda x: information_gain(df, x, target))

def build_tree(df, target):
    if len(df[target].unique()) == 1:
        return df[target].iloc[0]
    
    if len(df.columns) == 1:
        return df[target].mode()[0]
    
    best_feature = find_best_feature(df, target)
    tree = {best_feature: {}}
    
    for val in df[best_feature].unique():
        subset = df[df[best_feature] == val].drop(columns=[best_feature])
        tree[best_feature][val] = build_tree(subset, target)
    
    return tree

target_col = df.columns[-1]
tree = build_tree(df, target_col)

print("Decision Tree:")
print(tree)