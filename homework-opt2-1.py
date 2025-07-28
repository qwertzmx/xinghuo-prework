#普遍的一元函数f
import jax
import jax.numpy as jnp
from jax import grad
from typing import Callable

def variational_solver(
    f, 
    initial_guess: float, 
    error_threshold: float,
    initial_learning_rate: float = 0.001,
    max_iterations: int = 100000,
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
        return f(x)**2
    
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
        
        # 检查停止条件
        if abs(current_loss) < error_threshold**2:
            break
            
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
        print(f"|f(x)| ={current_loss:.8f}")
        
    
    # 返回最佳解
    solution = float(best_x)
    return solution

# 示例使用

# 输入处理
coefficients = list(map(float, input("请输入多项式系数:").split(",")))
guess = float(input("请输入初始猜测值: "))
    
# 定义目标函数
def target_function(x):
    return coefficients[0]*x**3 + coefficients[1]*x**2 + coefficients[2]*x + coefficients[3]
    
    # 求解
solution = variational_solver(
    target_function, 
    initial_guess=guess,
    error_threshold=0.0001,
)
    
    # 计算最终误差
final_error = abs(target_function(solution))
    
print(f"\n近似解: x ≈ {solution:.8f}")
print(f"最终 |f(x)| = {final_error:.8f}")