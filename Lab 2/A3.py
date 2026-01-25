import pandas as pd
import numpy as np
import time
import matplotlib.pyplot as plt

path = r"C:\Users\manic\OneDrive\Desktop\Amrita Documents\Semester 4\Machine Learning\Lab 2\Lab Session Data.xlsx"
df = pd.read_excel(path, sheet_name="IRCTC Stock Price")

price = df.iloc[:, 3].to_numpy()

chg_raw = df.iloc[:, 8]

chg = chg_raw.astype(str).str.replace("%", "", regex=False)
chg = pd.to_numeric(chg, errors="coerce")
df["Chg_num"] = chg

mean_np = np.mean(price)
var_np = np.var(price)

print("Mean (NumPy):", mean_np)
print("Variance (NumPy):", var_np)


def mean_my(arr):
    total = 0.0
    for x in arr:
        total += x
    return total / len(arr)

def var_my(arr):
    m = mean_my(arr)
    total = 0.0
    for x in arr:
        total += (x - m) ** 2
    return total / len(arr)  

mean_custom = mean_my(price)
var_custom = var_my(price)

print("\nMean (My Function):", mean_custom)
print("Variance (My Function):", var_custom)

print("\nMean difference (My - NumPy):", mean_custom - mean_np)
print("Variance difference (My - NumPy):", var_custom - var_np)


def avg_time_numpy(arr, runs=10):
    start = time.perf_counter()
    for _ in range(runs):
        np.mean(arr)
        np.var(arr)
    end = time.perf_counter()
    return (end - start) / runs

def avg_time_custom(arr, runs=10):
    start = time.perf_counter()
    for _ in range(runs):
        mean_my(arr)
        var_my(arr)
    end = time.perf_counter()
    return (end - start) / runs

print("\nAvg time (NumPy mean+var):", avg_time_numpy(price))
print("Avg time (My mean+var):", avg_time_custom(price))

wed_df = df[df["Day"] == "Wed"]
wed_price = wed_df.iloc[:, 3].to_numpy()

wed_mean = np.mean(wed_price)

print("\nPopulation mean price:", mean_np)
print("Wednesday sample mean price:", wed_mean)
print("Difference (Population - Wednesday):", mean_np - wed_mean)

apr_df = df[df["Month"] == "Apr"]
apr_price = apr_df.iloc[:, 3].to_numpy()

apr_mean = np.mean(apr_price)

print("\nApril sample mean price:", apr_mean)
print("Difference (Population - April):", mean_np - apr_mean)

valid_chg = df["Chg_num"].dropna()

loss_days = valid_chg.apply(lambda x: x < 0).sum()
total_days = len(valid_chg)

prob_loss = loss_days / total_days
print("\nProbability of making a loss:", prob_loss)

profit_and_wed = ((df["Day"] == "Wed") & (df["Chg_num"] > 0)).sum()
prob_profit_on_wed = profit_and_wed / total_days
print("\nProbability of making profit on Wednesday:", prob_profit_on_wed)

wed_total = (df["Day"] == "Wed").sum()
wed_profit = ((df["Day"] == "Wed") & (df["Chg_num"] > 0)).sum()

cond_prob_profit_given_wed = wed_profit / wed_total
print("\nConditional Probability P(Profit | Wednesday):", cond_prob_profit_given_wed)

day_map = {"Mon": 1, "Tue": 2, "Wed": 3, "Thu": 4, "Fri": 5}
df["Day_num"] = df["Day"].map(day_map)

plt.figure(figsize=(7, 4))
plt.scatter(df["Day_num"], df["Chg_num"])
plt.xticks([1, 2, 3, 4, 5], ["Mon", "Tue", "Wed", "Thu", "Fri"])
plt.xlabel("Day of the Week")
plt.ylabel("Chg%")
plt.title("Chg% vs Day of the Week (IRCTC)")
plt.grid(True)
plt.show()
