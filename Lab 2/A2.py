import pandas as pd
import numpy as np

path = r"C:\Users\manic\OneDrive\Desktop\Amrita Documents\Semester 4\Machine Learning\Lab 2\Lab Session Data.xlsx"

df = pd.read_excel(path, sheet_name="Purchase data")

X = df[["Candies (#)", "Mangoes (Kg)", "Milk Packets (#)"]].values
payment = df["Payment (Rs)"].values

Y2 = np.where(payment > 200, 1, 0).reshape(-1, 1)

X_pinv = np.linalg.pinv(X)
w = X_pinv @ Y2  

print("Classifier weights (w):\n", w)

scores = X @ w  
predicted_labels = np.where(scores >= 0.5, 1, 0)

predicted_status = np.where(predicted_labels == 1, "RICH", "POOR")
actual_status = np.where(Y2 == 1, "RICH", "POOR")

accuracy = np.mean(predicted_labels == Y2)
print("\nAccuracy:", accuracy)

result_df = df[["Candies (#)", "Mangoes (Kg)", "Milk Packets (#)", "Payment (Rs)"]].copy()
result_df["Actual"] = actual_status
result_df["Predicted"] = predicted_status
result_df["Score"] = scores

print("\nFirst 10 Predictions:")
print(result_df.head(10))
