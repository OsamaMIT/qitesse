import numpy as np
import qitesse


def build_problem():
    spec = qitesse.CircuitSpec(2)
    theta = spec.param("theta")
    phi = spec.param("phi")

    spec.ry(0, theta)
    spec.cx(0, 1)
    spec.rz(1, phi)

    compiled = spec.compile()
    hamiltonian = qitesse.Observable.hamiltonian(
        [
            qitesse.Observable.pauli_z(0, coefficient=0.7),
            qitesse.Observable.pauli_z(1, coefficient=-0.2),
            qitesse.Observable.pauli_string([("Z", 0), ("Z", 1)], coefficient=0.5),
        ]
    )
    return compiled, hamiltonian


def main():
    compiled, hamiltonian = build_problem()
    context = compiled.execution_context()

    params = np.array([0.6, 0.2], dtype=np.float32)
    learning_rate = np.float32(0.2)

    for step in range(8):
        value, gradient = context.value_and_gradient(compiled, params, hamiltonian)
        params = np.ascontiguousarray(params - learning_rate * gradient, dtype=np.float32)
        print(f"step={step:02d} value={value:.6f} params={params}")


if __name__ == "__main__":
    main()
