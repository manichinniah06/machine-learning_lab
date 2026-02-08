import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_percentage_error

def regression_metrics(X, y):
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    model = LinearRegression()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    mape = mean_absolute_percentage_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    return mse, rmse, mape, r2

def main():
    data = pd.read_excel(
        r"C:\Users\manic\OneDrive\Desktop\Amrita Documents\Semester 4\Machine Learning\Lab 4\Lab Session Data.xlsx"
    )

    X = data.iloc[:, 1:4]
    y = data["Payment (Rs)"]

    mse, rmse, mape, r2 = regression_metrics(X, y)

    print(mse)
    print(rmse)
    print(mape)
    print(r2)

if __name__ == "__main__":
    main()
