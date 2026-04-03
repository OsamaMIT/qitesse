# qitesse
[![PyPI Version](https://img.shields.io/pypi/v/qitesse.svg)](https://pypi.org/project/qitesse/)
[![License](https://img.shields.io/badge/license-MIT-blue.svg)](https://github.com/OsamaMIT/qitesse/blob/main/LICENSE)
[![Python Versions](https://img.shields.io/pypi/pyversions/qitesse.svg)](https://pypi.org/project/qitesse/)

**qitesse** is an open-source python API for qitesse-sim, the performant Rust quantum simulator.

qitesse is built upon qitesse-sim, the **high-performance CPU-based state-vector simulator** for quantum circuits, fully built in Rust.

This PyPI module provides a high-level python interface for the purpose of production, research, and development.

## Features

- Performant CPU based simulation
- State-vector execution for common single-, two-, and multi-qubit gates
- Mid-circuit `measure`, `reset`, and `barrier` operations
- Custom unitary operations with `Gate.unitary(...)`
- Controlled custom unitaries with `Gate.controlled_unitary(...)`

## Installation

qitesse requires **Python 3.8+**. Install it via pip:

```bash
pip install qitesse
```

Or install from source:

```bash
git clone https://github.com/OsamaMIT/qitesse.git

pip install maturin

maturin develop --release
```
To run examples:

`python examples/h_example.py`

`python examples/qft._example.py`

`python examples/custom_unitary.py`

## Documentation
_**Avaliable soon!**_

## Supported Gates

Single-qubit gates:

- `i`
- `x`
- `y`
- `z`
- `h`
- `s`
- `sdg`
- `t`
- `tdg`
- `rx`
- `ry`
- `rz`
- `p` / `phase`
- `u`

Two-qubit gates:

- `cnot` / `cx`
- `cy`
- `cz`
- `ch`
- `swap`
- `iswap`
- `crx`
- `cry`
- `crz`
- `cp` / `cphase`
- `cu`

Three-qubit and larger:

- `ccx` / `toffoli`
- `cswap` / `fredkin`
- `mcx`
- `mcz`
- `mcp` / `mcphase`
- `controlled_unitary`

Circuit operations:

- `measure`
- `reset`
- `barrier`

Custom unitaries:

```python
import numpy as np
import qitesse

hadamard = np.array([[1, 1], [1, -1]], dtype=np.complex64) / np.sqrt(2)

circuit = qitesse.Circuit([
    qitesse.Gate.unitary([0], hadamard),
    qitesse.Gate.controlled_unitary([0], [1], hadamard),
])

state = circuit.run(2)
```

Use `run_with_measurements(num_qubits)` if the circuit contains measurement gates and you want the observed bit values back.

## Planned Features
- Differentiable gradients
- Additional simulation backends

## Contributing
Contributions are welcome! To contribute:

1. Fork the repository
2. Create a new branch (feature-branch)
3. Commit your changes and open a pull request

## License
This project is licensed under the MIT License. See the LICENSE file for details.
