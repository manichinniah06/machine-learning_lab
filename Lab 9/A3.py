import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
import lime
import lime.lime_tabular

data = pd.read_csv(r"C:\Users\manic\OneDrive\Desktop\Amrita Documents\Semester 4\Machine Learning\Lab\f1_deviation_dataset_2022_2024.csv")

data = data.dropna()

X = data[['year','meeting_key','driver_number','grid_position','finish_position','deviation']]
y = data['class_label']

le = LabelEncoder()
y = le.fit_transform(y)

X = X.values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)

model = MLPClassifier(hidden_layer_sizes=(10,), max_iter=10000, random_state=1)
model.fit(X_train_scaled, y_train)

explainer = lime.lime_tabular.LimeTabularExplainer(
    X_train_scaled,
    feature_names=['year','meeting_key','driver_number','grid_position','finish_position','deviation'],
    class_names=['Neutral','Outperform','Underperform'],
    mode='classification'
)

exp = explainer.explain_instance(
    scaler.transform([X_test[0]])[0],
    model.predict_proba
)

exp.save_to_file("lime_output.html")
print("LIME output saved")