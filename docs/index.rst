qitesse
=======

qitesse is a high-throughput CPU backend for repeated evaluation of parameterized quantum circuits.

The documentation is organized around the compiled execution path first, because that is the product's core interface for hybrid quantum algorithms, optimizer loops, and backend integrations.

Current qitesse includes:

- one-off circuit execution with :code:`Gate` and :code:`Circuit`
- compiled parameterized execution with :code:`CircuitSpec` and :code:`CompiledCircuit`
- expectation, batch expectation, gradient, and batch gradient APIs
- reusable :code:`ExecutionContext` buffers
- statevector output, observables, Hamiltonians, custom unitaries, and measurement/reset operations

.. toctree::
   :maxdepth: 2
   :caption: Guides

   installation
   current_features
   quickstart
   guides/compiled_execution

.. toctree::
   :maxdepth: 2
   :caption: API Reference

   api/index
