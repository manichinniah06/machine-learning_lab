import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score

data = pd.read_csv(r"C:\Users\manic\OneDrive\Desktop\Amrita Documents\Semester 4\Machine Learning\Lab\f1_deviation_dataset_2022_2024.csv")

data = data.dropna()

X = data[['year','meeting_key','driver_number','grid_position']]
y = data['class_label']

le = LabelEncoder()
y = le.fit_transform(y)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)

pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('mlp', MLPClassifier(hidden_layer_sizes=(10,), max_iter=10000, random_state=1))
])

pipeline.fit(X_train, y_train)

print("Train Accuracy:", accuracy_score(y_train, pipeline.predict(X_train)))
print("Test Accuracy:", accuracy_score(y_test, pipeline.predict(X_test)))