import numpy as np
from scipy.linalg import expm

sigma_x = np.array([[0, 1], [1, 0]], dtype=complex)
sigma_y = np.array([[0, -1j], [1j, 0]], dtype=complex)
sigma_z = np.array([[1, 0], [0, -1]], dtype=complex)

def verify_exponential(P, theta):
    exp_P = expm(1j * theta * P)
    analytic = np.cos(theta) * np.eye(2) + 1j * np.sin(theta) * P
    
    print("expm(1j * theta * P):\n", exp_P)
    print("cos(theta)I + i sin(theta)P:\n", analytic)
    print("是否接近:", np.allclose(exp_P, analytic))

theta = np.pi*float(input("请输入："))
print("验证 σ_x:")
verify_exponential(sigma_x, theta)
print("\n验证 σ_y:")
verify_exponential(sigma_y, theta)
print("\n验证 σ_z:")
verify_exponential(sigma_z, theta)