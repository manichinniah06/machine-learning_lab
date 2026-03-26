import pandas as pd
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.preprocessing import LabelEncoder

df = pd.read_csv(r"C:\Users\manic\OneDrive\Desktop\Amrita Documents\Semester 4\Machine Learning\Lab\f1_deviation_dataset_2022_2024.csv")

df_encoded = df.copy()
le = LabelEncoder()

for col in df_encoded.columns:
    df_encoded[col] = le.fit_transform(df_encoded[col])

X = df_encoded.iloc[:, :-1]
y = df_encoded.iloc[:, -1]

model = DecisionTreeClassifier()
model.fit(X, y)

plt.figure(figsize=(12, 6))
plot_tree(model, feature_names=X.columns, filled=True)
plt.show()