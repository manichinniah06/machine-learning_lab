import numpy as np

X = np.array([
[20,6,2,386],
[16,3,6,289],
[27,6,2,393],
[19,1,2,110],
[24,4,2,280],
[22,1,5,167],
[15,4,2,271],
[18,4,2,274],
[21,1,4,148],
[16,2,4,198]
])

T = np.array([1,1,1,0,1,0,1,1,0,0])

X_bias = np.c_[np.ones(len(X)), X]
w = np.linalg.pinv(X_bias).dot(T)

print("Weights:", w)