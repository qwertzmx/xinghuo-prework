import numpy as np

X = np.array([[0, 1], [1, 0]], dtype=complex)
Z = np.array([[1, 0], [0, -1]], dtype=complex)
I = np.array([[1, 0], [0, 1]], dtype=complex)

def tensor_product(matrices):
    result = matrices[0]
    for mat in matrices[1:]:
        result = np.kron(result, mat)
    return result

def construct_H(n):
    H = np.zeros((2**n, 2**n), dtype=complex)
    for i in range(n):
        matrices = [I] * n
        matrices[i] = Z
        term = tensor_product(matrices)
        H += term
    for i in range(n - 1):
        matrices = [I] * n
        matrices[i] = X
        matrices[i + 1] = X
        term = tensor_product(matrices)
        H += term
    return H

def expectation(n):
    H = construct_H(n)
    v = np.zeros(2**n, dtype=complex)
    v[0] = 1
    expectation = np.vdot(v, H.dot(v)) 
    return np.real(expectation)

n=int(input("输入阶数"))
print(f"期望值 (n={n}):", expectation(n))