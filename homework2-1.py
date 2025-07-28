import numpy as np
import matplotlib.pyplot as plt

def rotation_matrix(theta):
    return np.array([
        [np.cos(theta), np.sin(theta)],
        [-np.sin(theta), np.cos(theta)]
    ])


v = np.array([1, 0])
theta = np.pi / 4
v_prime = rotation_matrix(theta) @ v


plt.figure(figsize=(8, 8))
plt.quiver(0, 0, v[0], v[1], angles='xy', scale_units='xy', scale=1, color='r', label='Original vector (1, 0)')
plt.quiver(0, 0, v_prime[0], v_prime[1], angles='xy', scale_units='xy', scale=1, color='b', label=f'Rotated vector (θ={np.degrees(theta):.0f}°)')


plt.xlim(-1.5, 1.5)
plt.ylim(-1.5, 1.5)
plt.axhline(0, color='black', linewidth=0.5)
plt.axvline(0, color='black', linewidth=0.5)
plt.grid(color='gray', linestyle='--', linewidth=0.5)
plt.title('Rotation of Vector by R(θ)')
plt.legend()
plt.show()