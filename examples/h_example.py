import qitesse
import numpy as np
import time

def main():
    # Define a quantum circuit with 3 qubits
    gates = [
        qitesse.Gate.h(0),  # Apply Hadamard gate to qubit 0
    ]

    # Build the circuit
    circuit = qitesse.Circuit(gates)

    # Run the circuit
    t0 = time.time()
    amplitudes = circuit.run(1)
    t1 = time.time()

    # Print the results
    print(f"Circuit execution took {t1 - t0:.3f}s")
    print("Amplitudes:", amplitudes)

if __name__ == "__main__":
    main()
