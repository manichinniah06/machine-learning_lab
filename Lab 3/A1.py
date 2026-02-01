import numpy as np
import math as m

a = np.array([1,2,3,4])

b = np.array([2,3,4,5])

def dotproduct(a,b):
    result = 0
    if len(a) != len(b):
        return "Vectors with different dimention"
    else:
        for i in range(len(a)):
            result +=a[i]*b[i]
    return result

def euclideannorm(a):
    result = 0
    for i in a:
        result += i*i
    result_norm  = m.sqrt(result)
    return result_norm

print(np.linalg.norm(a))

print(np.linalg.norm(b))

print(np.dot(a,b))

print(euclideannorm(a))

print(euclideannorm(b))

print(dotproduct(a,b))

