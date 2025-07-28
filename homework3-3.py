import numpy as np
sigma_x = np.array([[0, 1], [1, 0]], dtype=complex)
sigma_y = np.array([[0, -1j], [1j, 0]], dtype=complex)
sigma_z = np.array([[1, 0], [0, -1]], dtype=complex)
ket0 = np.array([1, 0], dtype=complex)

def f(theta, P1, P2):
    U = np.cos(theta/2) * np.eye(2) + 1j * np.sin(theta/2) * P1
    evolved_state = U @ ket0
    P2_evolved = P2 @ evolved_state
    return np.vdot(evolved_state, P2_evolved).real

def numerical_derivative(theta, P1, P2, delta=1e-5):
    return (f(theta + delta, P1, P2) - f(theta - delta, P1, P2)) / (2 * delta)