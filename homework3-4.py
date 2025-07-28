import jax
import jax.numpy as jnp
from jax import grad
from typing import Callable

def variational_solver(
    f, 
    initial_guess: float, 
    error_threshold: float=0.000001,
    initial_learning_rate: float = 0.1,
    max_iterations: int = 2000,
    decay_factor: float = 0.5,
    patience: int = 5
) -> float:
    """
    带学习率自适应衰减的变分优化求解器
    
    参数:
        f: 目标函数
        initial_guess: 初始猜测值
        error_threshold: 可接受的误差阈值
        initial_learning_rate: 初始学习率
        max_iterations: 最大迭代次数
        decay_factor: 学习率衰减因子
        patience: 允许的震荡次数
    
    返回:
        近似解 x
    """
    x = initial_guess
    lr = initial_learning_rate
    best_x = x
    best_loss = float("inf")
    oscillating_count = 0
    
    # 定义损失函数
    def loss(x):
        return f(x)
    
    # 使用JAX自动微分
    grad_loss = grad(loss)
    
    for i in range(max_iterations):
        current_loss = loss(x)
        current_grad = grad_loss(x)
        
        # 更新最佳解
        if current_loss < best_loss:
            best_loss = current_loss
            best_x = x
            oscillating_count = 0
        else:
            oscillating_count += 1
            
        # 如果震荡次数过多，衰减学习率
        if oscillating_count > patience:
            lr *= decay_factor
            oscillating_count = 0
            
        # 梯度下降更新
        delta=lr * current_grad
        if delta>0.1:
            delta=0.1
        if delta<-0.1:
            delta=-0.1
        x = x-delta
        print(f"迭代次数：{i}")
        print(f"近似解x:{x:.8f}")
        print(f"f(x) ={current_loss:.8f}")

        # 检查停止条件
        if abs(delta) < error_threshold:
            break
        
    
    # 返回最佳解
    solution = float(best_x)
    return solution


# 使用JAX实现的Pauli矩阵和初始态
sigma_x = jnp.array([[0, 1], [1, 0]], dtype=jnp.complex64)
sigma_y = jnp.array([[0, -1j], [1j, 0]], dtype=jnp.complex64)
sigma_z = jnp.array([[1, 0], [0, -1]], dtype=jnp.complex64)
pauli = [sigma_x, sigma_y, sigma_z]
ket0 = jnp.array([1, 0], dtype=jnp.complex64)

ij = input('请输入矩阵，0,1,2对应X,Y,Z').split(',')
P1 = pauli[int(ij[0])]
P2 = pauli[int(ij[1])]

def f(theta):
    U = jnp.cos(theta/2) * jnp.eye(2) + 1j * jnp.sin(theta/2) * P1
    evolved_state = U @ ket0
    P2_evolved = P2 @ evolved_state
    return jnp.vdot(evolved_state, P2_evolved).real

solution = variational_solver(
    f, 
    initial_guess=0.2,
)

print(f'最小值：{f(solution):.8f}')