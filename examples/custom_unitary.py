import numpy as np
import qitesse


def main():
    hadamard = np.array([[1, 1], [1, -1]], dtype=np.complex64) / np.sqrt(2)

    circuit = qitesse.Circuit([
        qitesse.Gate.unitary([0], hadamard),
        qitesse.Gate.controlled_unitary([0], [1], hadamard),
        qitesse.Gate.measure(0),
        qitesse.Gate.reset(1),
        qitesse.Gate.barrier(),
    ])

    state, measurements = circuit.run_with_measurements(2)

    print("Measurements:", measurements)
    print("State:", state)


if __name__ == "__main__":
    main()
