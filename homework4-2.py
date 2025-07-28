import tensorcircuit as tc
import numpy as np

# 设置随机种子以保证可重复性
tc.set_backend("numpy")
np.random.seed(42)

# 创建电路并施加门
c = tc.Circuit(2)
c.h(0)
c.cx(0, 1)

# 采样测量（运行多次）
n_samples = 1000  # 采样次数
samples = c.sample(allow_state=True, batch=n_samples)  # 返回 shape=(n_samples, 2)


expval_sum = 0.0
for i in range(n_samples):
    b=samples[i]
    bb,bbb=b
    b0=bb[0] 
    b1=bb[1] # 获取第 i 次测量的比特串
    z0z1 = (-1) ** (b0 + b1)  # 计算本次测量的 Z0Z1 值（+1 或 -1）
    expval_sum += z0z1
expval_estimate = expval_sum / n_samples

print("\nEstimated <Z0Z1> from samples (loop summation):", expval_estimate)
