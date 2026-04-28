from sklearn.neural_network import MLPClassifier

X = [[0,0],[0,1],[1,0],[1,1]]

y_and = [0,0,0,1]
y_xor = [0,1,1,0]

model_and = MLPClassifier(hidden_layer_sizes=(2,), activation='logistic', solver='adam', max_iter=10000, random_state=1)
model_and.fit(X, y_and)
print("AND:", model_and.predict(X))

model_xor = MLPClassifier(hidden_layer_sizes=(4,), activation='tanh', solver='adam', max_iter=10000, random_state=1)
model_xor.fit(X, y_xor)
print("XOR:", model_xor.predict(X))