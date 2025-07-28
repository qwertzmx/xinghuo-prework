import numpy as np
def numerical_gradient(f, x, delta=1e-5):
    grad=[0*len(x)]
    for i in range(len(x)):
        xo = x[i]
        x[i] = xo + delta
        f_plus = f(x)
        x[i] = xo - delta
        f_minus = f(x)
        grad[i] = (f_plus - f_minus) / (2 * delta)
        x[i] = xo
    return grad