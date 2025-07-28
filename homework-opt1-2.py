import numpy as np
from math import pi, sqrt
import tensorcircuit as tc

tc.set_backend("numpy")  # 使用numpy作为后端

# 问题参数
n_qubits = 6  # 3x2格子需要6个量子位
total_states = 2**n_qubits
solutions = ["011001", "100110"]  # 所有有效解
M = len(solutions)  # 解的数量

# 将解转换为索引
solution_indices = [int(s, 2) for s in solutions]

# 计算最优迭代次数
optimal_iterations = round((pi/4) * sqrt(total_states/M))
print(f"理论最优迭代次数: {optimal_iterations}")

# 初始化状态向量
def initialize_state():
    circ = tc.Circuit(n_qubits)
    for i in range(n_qubits):
        circ.h(i)  # 应用Hadamard门创建均匀叠加态
    return circ.state()

# Oracle操作 - 标记解
def oracle(state):
    new_state = state.copy()
    for idx in solution_indices:
        new_state[idx] *= -1  # 翻转解的相位
    return new_state

# 扩散算子
def diffusion_operator(state):
    # 创建电路来应用扩散算子
    circ = tc.Circuit(n_qubits)
    
    # 第一步：应用H门到所有量子位
    for i in range(n_qubits):
        circ.h(i)
    
    # 第二步：应用条件相位翻转（不使用多比特控制，用if实现）
    # 计算 |0...0> 状态的索引
    zero_state = 0
    state_vector = circ.state()
    for i in range(len(state_vector)):
        if i == zero_state:
            state_vector[i] *= -1
    
    # 第三步：再次应用H门到所有量子位
    for i in range(n_qubits):
        circ.h(i)
    
    # 应用这个变换到输入状态
    # 由于tensorcircuit的限制，我们直接计算扩散算子的矩阵形式
    mean_amp = np.mean(state)
    diffused_state = 2 * mean_amp - state
    return diffused_state

# 完整的Grover迭代
def grover_iteration(state, iterations):
    for _ in range(iterations):
        state = oracle(state)  # 应用Oracle
        state = diffusion_operator(state)  # 应用扩散算子
    return state

# 计算成功概率
def success_probability(state):
    prob = np.abs(state)**2
    return np.sum(prob[solution_indices])

# 模拟不同迭代次数
max_iterations = 10
for iterations in range(max_iterations + 1):
    state = initialize_state()
    state = grover_iteration(state, iterations)
    prob = success_probability(state)
    print(f"迭代次数 {iterations}: 成功概率 = {prob:.4f}")