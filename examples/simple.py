import numpy as np
from qitesse import Circuit, Gate

circuit = Circuit([
    Gate.h(0),
    Gate.cp(0, 1, np.pi / 2),
    Gate.ry(1, np.pi / 3),
    Gate.cnot(0, 1),
])

print(circuit.run(2))
