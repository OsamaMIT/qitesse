import numpy as np
import qitesse


class QitesseDevicePoc:
    """Proof-of-concept interface for a PennyLane-style execution backend."""

    def __init__(self, compiled, observable):
        self._compiled = compiled
        self._observable = observable
        self._context = compiled.execution_context()

    def execute(self, params_batch):
        params_batch = np.ascontiguousarray(params_batch, dtype=np.float32)
        if params_batch.ndim == 1:
            return self._context.expectation(self._compiled, params_batch, self._observable)
        return self._compiled.batch_expectation(params_batch, self._observable)

    def compute_derivatives(self, params_batch):
        params_batch = np.ascontiguousarray(params_batch, dtype=np.float32)
        if params_batch.ndim == 1:
            return self._context.gradient(self._compiled, params_batch, self._observable)
        return self._compiled.batch_gradient(params_batch, self._observable)

    def execute_and_compute_derivatives(self, params_batch):
        params_batch = np.ascontiguousarray(params_batch, dtype=np.float32)
        if params_batch.ndim == 1:
            return self._context.value_and_gradient(
                self._compiled,
                params_batch,
                self._observable,
            )
        return self._compiled.batch_value_and_gradient(params_batch, self._observable)


def build_device():
    spec = qitesse.CircuitSpec(2)
    gamma = spec.param("gamma")
    beta = spec.param("beta")

    spec.ry(0, gamma)
    spec.cx(0, 1)
    spec.rx(1, beta)

    compiled = spec.compile()
    observable = qitesse.Observable.pauli_string([("Z", 0), ("Z", 1)])
    return QitesseDevicePoc(compiled, observable)


def main():
    device = build_device()

    single = np.array([0.4, 0.7], dtype=np.float32)
    batch = np.array(
        [
            [0.1, 0.2],
            [0.2, 0.4],
            [0.3, 0.6],
        ],
        dtype=np.float32,
    )

    print("single value:", device.execute(single))
    print("single gradient:", device.compute_derivatives(single))
    print("single value+grad:", device.execute_and_compute_derivatives(single))
    print("batch values:", device.execute(batch))
    print("batch gradients:", device.compute_derivatives(batch))


if __name__ == "__main__":
    main()
