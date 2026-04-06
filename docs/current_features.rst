Current Features
================

qitesse currently combines a general-purpose statevector simulator with a compiled execution path for repeated parameterized workloads.

Execution Modes
---------------

qitesse currently exposes two main execution models:

- :code:`Gate` + :code:`Circuit` for one-off simulation
- :code:`CircuitSpec` + :code:`CompiledCircuit` for repeated execution of fixed circuit structure

General Simulation API
----------------------

The one-off simulation path currently includes:

- gate-by-gate circuit construction with :code:`Gate`
- execution through :code:`Circuit.run(...)`
- statevector output as :code:`numpy.complex64`
- optional :code:`run_with_measurements(...)`
- sampling through :code:`sample(...)`
- mid-circuit :code:`measure`, :code:`reset`, and :code:`barrier`

Compiled Execution API
----------------------

The compiled path currently includes:

- symbolic parameters through :code:`Parameter`
- reusable circuit structure through :code:`CircuitSpec`
- compiled execution plans through :code:`CompiledCircuit`
- reusable zero-copy parameter buffers through :code:`ParamBuffer` and :code:`ParamBatchBuffer`
- scalar expectation evaluation
- batched expectation evaluation
- scalar gradients via parameter-shift
- batched gradients via parameter-shift
- combined value-and-gradient calls
- reusable scalar execution buffers through :code:`ExecutionContext`
- optional statevector inspection

Observables
-----------

qitesse currently supports:

- single-qubit Pauli observables through :code:`Observable.pauli_x`, :code:`Observable.pauli_y`, and :code:`Observable.pauli_z`
- Pauli strings through :code:`Observable.pauli_string(...)`
- Hamiltonians assembled from observable terms through :code:`Observable.hamiltonian(...)`

Supported Gates
---------------

Single-qubit gates:

- :code:`i`
- :code:`x`
- :code:`y`
- :code:`z`
- :code:`h`
- :code:`s`
- :code:`sdg`
- :code:`t`
- :code:`tdg`
- :code:`rx`
- :code:`ry`
- :code:`rz`
- :code:`p` / :code:`phase`
- :code:`u`

Two-qubit gates:

- :code:`cnot` / :code:`cx`
- :code:`cy`
- :code:`cz`
- :code:`ch`
- :code:`swap`
- :code:`iswap`
- :code:`crx`
- :code:`cry`
- :code:`crz`
- :code:`cp` / :code:`cphase`
- :code:`cu`

Three-qubit and larger:

- :code:`ccx` / :code:`toffoli`
- :code:`cswap` / :code:`fredkin`
- :code:`mcx`
- :code:`mcz`
- :code:`mcp` / :code:`mcphase`

Custom Operations
-----------------

qitesse currently supports:

- arbitrary custom unitary gates through :code:`Gate.unitary(...)`
- controlled custom unitaries through :code:`Gate.controlled_unitary(...)`

Performance-Oriented Features
-----------------------------

The current backend-oriented features include:

- low Python overhead for repeated compiled execution
- Rust-side parameter binding
- reusable parameter buffers that avoid per-call Python-to-Rust parameter copies
- batch execution APIs
- GIL-released compiled hot paths
- reusable execution buffers for sequential scalar evaluation

For usage patterns and examples built around these capabilities, continue to :doc:`quickstart` and :doc:`guides/compiled_execution`.
