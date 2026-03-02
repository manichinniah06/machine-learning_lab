from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import pandas as pd
from sklearn.metrics import mean_squared_error, r2_score
import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score
import matplotlib.pyplot as plt

path = r"C:\Users\manic\OneDrive\Desktop\Amrita Documents\Semester 4\Machine Learning\f1_deviation_dataset_2022_2024.csv"

df = pd.read_csv(path)

print("Dataset Shape:", df.shape)
print("Columns:", df.columns)


X_single = df[['grid_position']]      
y = df['deviation']                  

X_train, X_test, y_train, y_test = train_test_split(
    X_single, y, test_size=0.2, random_state=42
)

reg1 = LinearRegression()
reg1.fit(X_train, y_train)

y_train_pred = reg1.predict(X_train)

print("\nA1 - Predicted values (Single Feature):")
print(y_train_pred)


X_multi = df[['year','meeting_key','driver_number','grid_position','finish_position']]

X_train, X_test, y_train, y_test = train_test_split(
    X_multi, y, test_size=0.2, random_state=42
)

reg2 = LinearRegression()
reg2.fit(X_train, y_train)

y_train_pred = reg2.predict(X_train)
y_test_pred = reg2.predict(X_test)

mse_train = mean_squared_error(y_train, y_train_pred)
rmse_train = np.sqrt(mse_train)
mape_train = np.mean(np.abs((y_train - y_train_pred) / y_train)) * 100
r2_train = r2_score(y_train, y_train_pred)

mse_test = mean_squared_error(y_test, y_test_pred)
rmse_test = np.sqrt(mse_test)
mape_test = np.mean(np.abs((y_test - y_test_pred) / y_test)) * 100
r2_test = r2_score(y_test, y_test_pred)

print("\nA2 & A3 - Regression Metrics")
print("Training Data:")
print("MSE:", mse_train)
print("RMSE:", rmse_train)
print("MAPE:", mape_train)
print("R2:", r2_train)

print("\nTest Data:")
print("MSE:", mse_test)
print("RMSE:", rmse_test)
print("MAPE:", mape_test)
print("R2:", r2_test)

X_cluster = df[['year','meeting_key','driver_number','grid_position','finish_position','deviation']]

X_train, X_test = train_test_split(
    X_cluster, test_size=0.2, random_state=42
)

kmeans = KMeans(n_clusters=2, random_state=42, n_init="auto")
kmeans.fit(X_train)

print("\nA4 - KMeans Clustering")
print("Cluster Labels:")
print(kmeans.labels_)

print("\nCluster Centers:")
print(kmeans.cluster_centers_)

sil_score = silhouette_score(X_train, kmeans.labels_)
ch_score = calinski_harabasz_score(X_train, kmeans.labels_)
db_index = davies_bouldin_score(X_train, kmeans.labels_)

print("Silhouette Score:", sil_score)
print("Calinski-Harabasz Score:", ch_score)
print("Davies-Bouldin Index:", db_index)

k_values = range(2, 11)

sil_scores = []
ch_scores = []
db_scores = []

for k in k_values:
    kmeans = KMeans(n_clusters=k, random_state=42, n_init="auto")
    kmeans.fit(X_cluster)
    labels = kmeans.labels_
    
    sil_scores.append(silhouette_score(X_cluster, labels))
    ch_scores.append(calinski_harabasz_score(X_cluster, labels))
    db_scores.append(davies_bouldin_score(X_cluster, labels))

plt.figure()
plt.plot(k_values, sil_scores, marker='o')
plt.title("Silhouette Score vs k")
plt.show()

plt.figure()
plt.plot(k_values, ch_scores, marker='o')
plt.title("CH Score vs k")
plt.show()

plt.figure()
plt.plot(k_values, db_scores, marker='o')
plt.title("DB Index vs k")
plt.show()


distortions = []

for k in range(2, 20):
    kmeans = KMeans(n_clusters=k, random_state=42, n_init="auto")
    kmeans.fit(X_train)
    distortions.append(kmeans.inertia_)

plt.figure()
plt.plot(range(2, 20), distortions, marker='o')
plt.xlabel("Number of Clusters (k)")
plt.ylabel("Inertia")
plt.title("Elbow Plot")
plt.show()