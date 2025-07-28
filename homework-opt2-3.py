#普遍的二元函数f
import jax
import jax.numpy as jnp
from jax import grad
from typing import Callable, List, Tuple
import math

def variational_solver_2d(
    f: Callable[[jnp.ndarray], float],
    initial_guess: List[float], 
    error_threshold: float,
    initial_learning_rate: float =0.8,
    max_iterations: int = 10000,
    decay_factor: float = 0.9,
    patience: int = 10
) -> Tuple[float, float]:
    """
    二元函数的变分优化求解器（向量输入版）
    
    参数:
        f: 目标函数，接受jnp数组[x,y]并返回标量
        initial_guess: 初始猜测值[x,y]
        error_threshold: 可接受的误差阈值
        initial_learning_rate: 初始学习率
        max_iterations: 最大迭代次数
        decay_factor: 学习率衰减因子
        patience: 允许的震荡次数
    
    返回:
        近似解 (x, y)
    """
    # 初始化变量
    params = jnp.array(initial_guess, dtype=jnp.float32)
    lr = initial_learning_rate
    best_params = params
    best_loss = float("inf")
    oscillating_count = 0
    
    # 定义损失函数
    def loss(p):
        return f(p)
    
    # 使用JAX自动微分
    grad_loss = grad(loss)
    
    for i in range(max_iterations):
        current_loss = loss(params)
        current_grad = grad_loss(params)
        
        # 如果震荡次数过多，衰减学习率
        if oscillating_count > patience:
            lr *= decay_factor
            oscillating_count = 0

        # 更新最佳解
        if current_loss < best_loss:
            best_loss = current_loss
            best_params = params
            oscillating_count = 0
        else:
            oscillating_count += 1
            
        # 梯度下降更新（带裁剪）
        delta = lr * current_grad
        delta = jnp.clip(delta, -0.1, 0.1)  # 裁剪到[-0.1, 0.1]范围
        params = params - delta
        
        # 打印进度
        print(f"迭代次数：{i}")
        print(f"变化值：({delta[0]:.8f}, {delta[1]:.8f})")
        print(f"近似解 (k,b): ({params[0]:.8f}, {params[1]:.8f})")
        print(f"f(k,b) = {current_loss:.8f}")
        print(f"最优值={best_loss:.8f}")

        # 检查停止条件
        if jnp.linalg.norm(delta) < error_threshold:
            break

        
    
    # 返回最佳解
    solution = (float(best_params[0]), float(best_params[1]))
    return solution

# 输入处理
num = int(input("请输入组数:"))
poi = []
for i in range(num):
    coords = list(map(float, input(f"请输入第{i}组坐标(x,y): ").split(',')))
    poi.extend(coords)  # 展平坐标列表

# 定义目标函数（计算到所有点的距离总和）
def f(line: jnp.ndarray) -> float:
    total_distance = 0.0
    for i in range(0, len(poi), 2):
        dis0=(line[0]*poi[i]+line[1]-poi[i+1])
        total_distance+=jnp.abs(dis0)/jnp.sqrt(line[0]**2+1)
    return total_distance

# 求解
initial_guess = list(map(float, input("请输入初始猜测值(k,b): ").split(',')))
solution = variational_solver_2d(
    f,
    initial_guess=initial_guess,
    error_threshold=0.00000001
)

print(f"\n最优解: (k,b) ≈ ({solution[0]:.8f}, {solution[1]:.8f})")
print(f"最小总距离: {f(jnp.array(solution)):.8f}")