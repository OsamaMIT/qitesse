Compiled Execution
==================

qitesse is optimized for repeated evaluation of the same circuit structure on CPU.

The compiled path is the one to use for hybrid quantum algorithms, variational circuits, parameter sweeps, and framework integrations:

1. Build a :code:`CircuitSpec`
2. Register parameters once
3. Compile once
4. Reuse the resulting :code:`CompiledCircuit` across many calls

Choose The Right Execution Path
-------------------------------

Use :code:`Circuit([...]).run(...)` when:

- you need a quick one-off simulation
- you are debugging gate-level behavior
- you need mid-circuit measurement or reset semantics

Use :code:`CircuitSpec(...).compile()` when:

- the circuit structure is fixed and only parameter values change
- you are inside an optimizer loop
- you want expectation values or gradients
- you want to amortize Python overhead across many evaluations

Scalar Evaluation
-----------------

.. code-block:: python

   import numpy as np
   import qitesse

   spec = qitesse.CircuitSpec(1)
   theta = spec.param("theta")

   spec.ry(0, theta)

   compiled = spec.compile()
   observable = qitesse.Observable.pauli_z(0)

   params = np.array([0.37], dtype=np.float32)

   value = compiled.expectation(params, observable)
   gradient = compiled.gradient(params, observable)
   value_again, grad_again = compiled.value_and_gradient(params, observable)

:code:`value_and_gradient(...)` is the better default when the caller needs both values and derivatives in the same optimization step.

Batch Evaluation
----------------

.. code-block:: python

   import numpy as np

   params_batch = np.array(
       [[0.1], [0.2], [0.3], [0.4]],
       dtype=np.float32,
   )

   values = compiled.batch_expectation(params_batch, observable)
   gradients = compiled.batch_gradient(params_batch, observable)

Batch APIs are the preferred path for:

- parameter sweeps
- minibatch-style training loops
- backend integrations that already accumulate many parameter vectors per call

They avoid a Python loop over scalar :code:`expectation(...)` or :code:`gradient(...)` calls.

Zero-Copy Parameter Buffers
---------------------------

For repeated workloads, qitesse also provides reusable parameter buffers backed by NumPy arrays.

.. code-block:: python

   buffer = compiled.param_buffer()
   buffer.numpy()[:] = [0.37]

   value, gradient = compiled.value_and_gradient_buffer(buffer, observable)

   batch_buffer = compiled.param_batch_buffer(batch_size=128)
   batch_buffer.numpy()[:] = params_batch
   values = compiled.batch_expectation_buffer(batch_buffer, observable)

These buffer-based APIs avoid the per-call :code:`numpy -> Vec<f32>` copies used by the convenience path.

They are the preferred interface for:

- tight optimizer loops
- framework adapters that repeatedly overwrite the same parameter storage
- workloads where Python overhead matters as much as kernel time

Reusing Execution Contexts
--------------------------

:code:`ExecutionContext` keeps the statevector allocation alive across repeated scalar calls.

.. code-block:: python

   context = compiled.execution_context()

   for step_params in optimizer_steps:
       value, gradient = context.value_and_gradient(compiled, step_params, observable)

This is useful when:

- the caller naturally evaluates one parameter vector at a time
- the circuit is too small for batching to dominate
- you want deterministic reuse of internal buffers

Input Contracts
---------------

For best throughput:

- pass contiguous :code:`numpy.float32` parameter arrays
- use shape :code:`(n_params,)` for scalar execution
- use shape :code:`(batch_size, n_params)` for batch execution
- keep observables prebuilt instead of recreating them every iteration

qitesse will coerce compatible NumPy inputs, but pre-normalizing inputs on the caller side reduces avoidable overhead.

Integration Pattern
-------------------

Higher-level libraries should treat qitesse as a compiled backend:

1. translate framework circuit structure into a :code:`CircuitSpec`
2. compile once when the structure changes
3. keep parameter binding in NumPy
4. call :code:`expectation`, :code:`batch_expectation`, :code:`gradient`, or :code:`value_and_gradient`

Good adapter layers are thin. They should avoid:

- rebuilding the circuit per step
- converting parameters gate-by-gate in Python
- materializing full statevectors unless the caller explicitly asks for them

Current Gradient Scope
----------------------

Gradients on the compiled path currently use parameter-shift evaluation for supported parameterized gates.

That is a good fit for:

- VQE
- QAOA
- variational classifiers
- small-to-medium CPU training loops

Future work can add lower-overhead derivative kernels without changing the compiled-circuit workflow.

Production Checklist
--------------------

- compile the circuit once per structure
- keep observables cached
- prefer :code:`value_and_gradient(...)` over separate value and gradient calls when both are needed
- prefer batch APIs over Python loops when you already have many parameter vectors
- reuse :code:`ExecutionContext` for sequential scalar calls
- keep parameters in contiguous :code:`float32` NumPy arrays
