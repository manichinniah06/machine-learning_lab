import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.svm import LinearSVC
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score

def load_data():
    data = pd.read_excel(
        r"C:\Users\manic\OneDrive\Desktop\Amrita Documents\Semester 4\Machine Learning\Lab 4\Lab Session Data.xlsx",
        sheet_name=2
    )

    data = data.dropna()

    X = data.iloc[:, :-1]
    y_raw = data.iloc[:, -1].astype(str).str.strip()

    class_counts = y_raw.value_counts()
    if len(class_counts) < 2:
        raise ValueError("Less than two classes in target column")

    classes = class_counts.index[:2]
    y = y_raw.map({classes[0]: 0, classes[1]: 1})

    data = X[y.notna()]
    y = y.dropna()

    X_encoded = pd.get_dummies(data)

    return X_encoded.values, y.values

def train_and_evaluate(X, y):
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    model = LinearSVC()
    model.fit(X_train, y_train)

    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)

    return (
        confusion_matrix(y_train, y_train_pred),
        classification_report(y_train, y_train_pred),
        accuracy_score(y_train, y_train_pred),
        confusion_matrix(y_test, y_test_pred),
        classification_report(y_test, y_test_pred),
        accuracy_score(y_test, y_test_pred)
    )

def fit_type(train_acc, test_acc):
    if train_acc < 0.7 and test_acc < 0.7:
        return "Underfit"
    elif train_acc > 0.9 and test_acc < 0.7:
        return "Overfit"
    else:
        return "Regular fit"

def main():
    X, y = load_data()

    print("Samples:", len(y))
    print("Classes:", set(y))

    cm_tr, cr_tr, tr_acc, cm_te, cr_te, te_acc = train_and_evaluate(X, y)

    print("Training CM")
    print(cm_tr)
    print(cr_tr)
    print("Training Accuracy:", tr_acc)

    print("Testing CM")
    print(cm_te)
    print(cr_te)
    print("Testing Accuracy:", te_acc)

    print("Model Fit:", fit_type(tr_acc, te_acc))

if __name__ == "__main__":
    main()
