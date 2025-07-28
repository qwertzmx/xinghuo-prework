import tensorcircuit as tc
import numpy as np

c = tc.Circuit(2)
c.h(0)       
c.cx(0, 1)

expval = c.expectation([tc.gates.z(), [0]], [tc.gates.z(), [1]])

state = c.state()
z0 = tc.gates.z().tensor
z1 = tc.gates.z().tensor
z0z1 = np.kron(z0, z1)
expval_manual = np.vdot(state, z0z1 @ state)

print("Expectation value (TensorCircuit):", expval)
print("Expectation value (Manual):", expval_manual)