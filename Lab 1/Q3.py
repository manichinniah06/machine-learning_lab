import numpy as np

n = int(input("Enter size of square matrix: "))

A = []
for i in range(n):
    row = []
    for j in range(n):
        row.append(int(input(f"Enter A[{i}][{j}] value: ")))
    A.append(row)

A = np.array(A)

m = int(input("Enter power m: "))

result = np.linalg.matrix_power(A, m)

print(f"\nMatrix A^{m}:")
print(result)
