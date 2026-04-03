import math
import time
import qitesse
import numpy as np

def qft(n: int, threads: int = None) -> "qitesse.Circuit":
    """
    Build a gate-by-gate QFT circuit for n qubits.

    Gate order:
      for i in 0..n-1:
        H(i)
        for j in i+1..n-1:
          CP(control=i, target=j, theta = pi / 2^(j-i))
      swap qubits via SWAP for i in 0..n/2-1

    Parameters:
        n (int): Number of qubits.
        threads (int, optional): Number of CPU threads to use. Defaults to None.

    Returns:
        qitesse.Circuit: Constructed QFT circuit.
    """
    if n <= 0:
        raise ValueError("n must be positive")

    if threads is not None:
        if threads <= 0:
            raise ValueError("Number of threads must be positive")
        qitesse.set_num_threads(threads)

    gates = []

    # Main QFT body: Hadamard + controlled phase rotations
    for i in range(n):
        gates.append(qitesse.Gate.h(i))
        for j in range(i + 1, n):
            theta = math.pi / float(1 << (j - i))  # pi / 2^(j-i)
            gates.append(qitesse.Gate.cp(i, j, theta))

    # Bit-reversal swaps
    for i in range(n // 2):
        a = i
        b = n - 1 - i
        gates.append(qitesse.Gate.swap(a, b))

    return qitesse.Circuit(gates)


def main():
    n = 10  # change to desired qubit count
    threads = None  # change to desired number of threads, or None for all
    circuit = qft(n, threads)

    t0 = time.time()
    amplitudes = circuit.run(n)
    t1 = time.time()

    print(f"QFT-{n} circuit execution took {t1 - t0:.6f}s")
    print("Number of amplitudes:", amplitudes.shape)
    # Print a few amplitudes (complex64)
    print("First 8 amplitudes:", amplitudes[:8])

    # Optional: verify normalization
    probs = np.abs(amplitudes)**2
    print("Sum of probabilities (should be 1):", np.sum(probs))

if __name__ == "__main__":
    main()
