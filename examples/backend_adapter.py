import numpy as np
import qitesse


class CompiledExpectationBackend:
    """Thin backend wrapper for repeated compiled-circuit execution."""

    def __init__(self, compiled, observable):
        self._compiled = compiled
        self._observable = observable
        self._context = compiled.execution_context()

    def expectation(self, params):
        params = np.ascontiguousarray(params, dtype=np.float32)
        return self._context.expectation(self._compiled, params, self._observable)

    def gradient(self, params):
        params = np.ascontiguousarray(params, dtype=np.float32)
        return self._context.gradient(self._compiled, params, self._observable)

    def value_and_gradient(self, params):
        params = np.ascontiguousarray(params, dtype=np.float32)
        return self._context.value_and_gradient(self._compiled, params, self._observable)

    def batch_expectation(self, params_batch):
        params_batch = np.ascontiguousarray(params_batch, dtype=np.float32)
        return self._compiled.batch_expectation(params_batch, self._observable)

    def batch_gradient(self, params_batch):
        params_batch = np.ascontiguousarray(params_batch, dtype=np.float32)
        return self._compiled.batch_gradient(params_batch, self._observable)


def build_backend():
    spec = qitesse.CircuitSpec(2)
    theta = spec.param("theta")
    phi = spec.param("phi")

    spec.ry(0, theta)
    spec.cx(0, 1)
    spec.rz(1, phi)

    compiled = spec.compile()
    observable = qitesse.Observable.hamiltonian(
        [
            qitesse.Observable.pauli_z(0, coefficient=0.5),
            qitesse.Observable.pauli_z(1, coefficient=0.5),
        ]
    )
    return CompiledExpectationBackend(compiled, observable)


def main():
    backend = build_backend()

    params = np.array([0.2, 0.5], dtype=np.float32)
    value, gradient = backend.value_and_gradient(params)

    params_batch = np.array(
        [
            [0.1, 0.2],
            [0.2, 0.4],
            [0.3, 0.6],
        ],
        dtype=np.float32,
    )

    print("value:", value)
    print("gradient:", gradient)
    print("batch values:", backend.batch_expectation(params_batch))
    print("batch gradients:", backend.batch_gradient(params_batch))


if __name__ == "__main__":
    main()
