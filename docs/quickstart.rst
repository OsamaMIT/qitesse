Quickstart
==========

Compiled circuits are the primary qitesse interface for repeated execution.

.. code-block:: python

   import numpy as np
   import qitesse

   spec = qitesse.CircuitSpec(2)
   theta = spec.param("theta")

   spec.ry(0, theta)
   spec.cx(0, 1)

   compiled = spec.compile()
   observable = qitesse.Observable.pauli_z(1)

   params = np.array([0.4], dtype=np.float32)
   value = compiled.expectation(params, observable)
   gradient = compiled.gradient(params, observable)

   params_batch = np.array([[0.1], [0.2], [0.3]], dtype=np.float32)
   values = compiled.batch_expectation(params_batch, observable)

For sequential scalar execution, reuse an :code:`ExecutionContext`:

.. code-block:: python

   context = compiled.execution_context()
   value, grad = context.value_and_gradient(compiled, params, observable)

For the detailed execution model and integration guidance, continue to :doc:`guides/compiled_execution`.
