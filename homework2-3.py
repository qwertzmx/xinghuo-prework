import numpy as np
import matplotlib.pyplot as plt

# 定义泡利矩阵
pauli_matrices = [
    np.array([[0, 1], [1, 0]], dtype=complex),  # sigma_x
    np.array([[0, -1j], [1j, 0]], dtype=complex),  # sigma_y
    np.array([[1, 0], [0, -1]], dtype=complex)  # sigma_z
]

def expectation_value(P_idx, Q_idx, theta):
    # 获取 P 和 Q
    P = pauli_matrices[P_idx]
    Q = pauli_matrices[Q_idx]
    
    # 计算 e^{i θ/2 P}
    exp_P = np.cos(theta/2) * np.eye(2) + 1j * np.sin(theta/2) * P
    
    # 初始向量 v0 = [1, 0]
    v0 = np.array([1, 0], dtype=complex)
    
    # 计算 v(θ) = e^{i θ/2 P} v0
    v_theta = exp_P @ v0
    
    # 计算期望值 <Q> = v^† Q v
    expectation = np.vdot(v_theta, Q @ v_theta)  # v^† Q v
    
    # 由于泡利矩阵是厄米的，期望值应为实数
    return np.real(expectation)

# 测试
print(expectation_value(0, 0, np.pi/2))  # P=sigma_x, Q=sigma_x, θ=π/2

# 可视化
theta_values = np.linspace(0, 2*np.pi, 100)
P_indices = [0, 1, 2]  # sigma_x, sigma_y, sigma_z
Q_indices = [0, 1, 2]

plt.figure(figsize=(12, 8))
for P_idx in P_indices:
    for Q_idx in Q_indices:
        expectations = [expectation_value(P_idx, Q_idx, theta) for theta in theta_values]
        plt.plot(theta_values, expectations, 
                label=f'P={["σx", "σy", "σz"][P_idx]}, Q={["σx", "σy", "σz"][Q_idx]}')

plt.xlabel('θ')
plt.ylabel('Expectation value ⟨Q⟩')
plt.title('Expectation value ⟨Q⟩ vs θ for different P and Q')
plt.legend()
plt.grid()
plt.show()