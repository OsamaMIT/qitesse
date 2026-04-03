import math
import subprocess
import sys
import textwrap
import unittest

import numpy as np

try:
    import qitesse
except ModuleNotFoundError as exc:  # pragma: no cover - depends on local extension install
    raise unittest.SkipTest(
        "qitesse is not installed in this environment. Run `maturin develop` before executing the test suite."
    ) from exc


SQRT2_INV = np.float32(1.0 / math.sqrt(2.0))


def phase(theta: float) -> np.complex64:
    return np.complex64(np.exp(1j * theta))


def basis_state(num_qubits: int, index: int, amplitude: complex = 1.0 + 0.0j) -> np.ndarray:
    state = np.zeros(1 << num_qubits, dtype=np.complex64)
    state[index] = np.complex64(amplitude)
    return state


def run_state(gates, num_qubits: int) -> np.ndarray:
    circuit = qitesse.Circuit(list(gates))
    state = circuit.run(num_qubits)
    if not isinstance(state, np.ndarray):
        raise AssertionError(f"Expected numpy.ndarray from Circuit.run, got {type(state)!r}")
    if state.dtype != np.complex64:
        raise AssertionError(f"Expected complex64 amplitudes, got {state.dtype!r}")
    return state


class PythonApiTests(unittest.TestCase):
    def assert_state_close(self, actual, expected):
        np.testing.assert_allclose(actual, np.asarray(expected, dtype=np.complex64), atol=1e-5, rtol=1e-5)

    def assert_alias_gate_state(self, primary, alias, num_qubits: int):
        self.assert_state_close(run_state(primary, num_qubits), run_state(alias, num_qubits))

    def test_single_qubit_gate_constructors(self):
        theta = math.pi / 3

        cases = [
            ("i", [qitesse.Gate.i(0)], basis_state(1, 0)),
            ("x", [qitesse.Gate.x(0)], basis_state(1, 1)),
            ("y", [qitesse.Gate.y(0)], basis_state(1, 1, 1j)),
            ("z", [qitesse.Gate.x(0), qitesse.Gate.z(0)], basis_state(1, 1, -1.0)),
            ("h", [qitesse.Gate.h(0)], np.array([SQRT2_INV, SQRT2_INV], dtype=np.complex64)),
            ("s", [qitesse.Gate.x(0), qitesse.Gate.s(0)], basis_state(1, 1, 1j)),
            ("sdg", [qitesse.Gate.x(0), qitesse.Gate.sdg(0)], basis_state(1, 1, -1j)),
            ("t", [qitesse.Gate.x(0), qitesse.Gate.t(0)], basis_state(1, 1, phase(math.pi / 4))),
            ("tdg", [qitesse.Gate.x(0), qitesse.Gate.tdg(0)], basis_state(1, 1, phase(-math.pi / 4))),
            ("rx", [qitesse.Gate.rx(0, math.pi)], basis_state(1, 1, -1j)),
            ("ry", [qitesse.Gate.ry(0, math.pi)], basis_state(1, 1)),
            ("rz", [qitesse.Gate.rz(0, math.pi)], basis_state(1, 0, -1j)),
            ("p", [qitesse.Gate.x(0), qitesse.Gate.p(0, theta)], basis_state(1, 1, phase(theta))),
            ("u", [qitesse.Gate.u(0, math.pi, 0.0, math.pi)], basis_state(1, 1)),
        ]

        for name, gates, expected in cases:
            with self.subTest(gate=name):
                self.assert_state_close(run_state(gates, 1), expected)

    def test_single_qubit_aliases(self):
        self.assert_alias_gate_state(
            [qitesse.Gate.x(0), qitesse.Gate.p(0, 0.37)],
            [qitesse.Gate.x(0), qitesse.Gate.phase(0, 0.37)],
            1,
        )

    def test_controlled_and_two_qubit_gate_constructors(self):
        s = SQRT2_INV
        cases = [
            (
                "cnot",
                [qitesse.Gate.h(0), qitesse.Gate.cnot(0, 1)],
                np.array([s, 0.0, 0.0, s], dtype=np.complex64),
            ),
            (
                "cy",
                [qitesse.Gate.h(0), qitesse.Gate.cy(0, 1)],
                np.array([s, 0.0, 0.0, 1j * s], dtype=np.complex64),
            ),
            (
                "cz",
                [qitesse.Gate.x(0), qitesse.Gate.x(1), qitesse.Gate.cz(0, 1)],
                basis_state(2, 3, -1.0),
            ),
            (
                "ch",
                [qitesse.Gate.x(0), qitesse.Gate.ch(0, 1)],
                np.array([0.0, s, 0.0, s], dtype=np.complex64),
            ),
            (
                "swap",
                [qitesse.Gate.x(0), qitesse.Gate.swap(0, 1)],
                basis_state(2, 2),
            ),
            (
                "iswap",
                [qitesse.Gate.x(0), qitesse.Gate.iswap(0, 1)],
                basis_state(2, 2, 1j),
            ),
            (
                "crx",
                [qitesse.Gate.x(0), qitesse.Gate.crx(0, 1, math.pi)],
                basis_state(2, 3, -1j),
            ),
            (
                "cry",
                [qitesse.Gate.x(0), qitesse.Gate.cry(0, 1, math.pi)],
                basis_state(2, 3),
            ),
            (
                "crz",
                [qitesse.Gate.x(0), qitesse.Gate.h(1), qitesse.Gate.crz(0, 1, math.pi)],
                np.array([0.0, -1j * s, 0.0, 1j * s], dtype=np.complex64),
            ),
            (
                "cp",
                [qitesse.Gate.x(0), qitesse.Gate.h(1), qitesse.Gate.cp(0, 1, math.pi / 2)],
                np.array([0.0, s, 0.0, 1j * s], dtype=np.complex64),
            ),
            (
                "cu",
                [qitesse.Gate.h(0), qitesse.Gate.cu(0, 1, math.pi, 0.0, math.pi)],
                np.array([s, 0.0, 0.0, s], dtype=np.complex64),
            ),
        ]

        for name, gates, expected in cases:
            with self.subTest(gate=name):
                self.assert_state_close(run_state(gates, 2), expected)

    def test_controlled_gate_aliases(self):
        self.assert_alias_gate_state(
            [qitesse.Gate.h(0), qitesse.Gate.cnot(0, 1)],
            [qitesse.Gate.h(0), qitesse.Gate.cx(0, 1)],
            2,
        )
        self.assert_alias_gate_state(
            [qitesse.Gate.x(0), qitesse.Gate.h(1), qitesse.Gate.cp(0, 1, 0.23)],
            [qitesse.Gate.x(0), qitesse.Gate.h(1), qitesse.Gate.cphase(0, 1, 0.23)],
            2,
        )

    def test_multi_qubit_gate_constructors(self):
        s = SQRT2_INV
        cases = [
            (
                "ccx",
                [qitesse.Gate.x(0), qitesse.Gate.x(1), qitesse.Gate.ccx(0, 1, 2)],
                basis_state(3, 7),
            ),
            (
                "cswap",
                [qitesse.Gate.x(0), qitesse.Gate.x(1), qitesse.Gate.cswap(0, 1, 2)],
                basis_state(3, 5),
            ),
            (
                "mcx",
                [qitesse.Gate.x(0), qitesse.Gate.x(1), qitesse.Gate.mcx([0, 1], 2)],
                basis_state(3, 7),
            ),
            (
                "mcz",
                [qitesse.Gate.x(0), qitesse.Gate.x(1), qitesse.Gate.x(2), qitesse.Gate.mcz([0, 1], 2)],
                basis_state(3, 7, -1.0),
            ),
            (
                "mcp",
                [qitesse.Gate.x(0), qitesse.Gate.x(1), qitesse.Gate.h(2), qitesse.Gate.mcp([0, 1], 2, math.pi / 2)],
                np.array([0.0, 0.0, 0.0, s, 0.0, 0.0, 0.0, 1j * s], dtype=np.complex64),
            ),
        ]

        for name, gates, expected in cases:
            with self.subTest(gate=name):
                self.assert_state_close(run_state(gates, 3), expected)

    def test_multi_qubit_aliases(self):
        self.assert_alias_gate_state(
            [qitesse.Gate.x(0), qitesse.Gate.x(1), qitesse.Gate.ccx(0, 1, 2)],
            [qitesse.Gate.x(0), qitesse.Gate.x(1), qitesse.Gate.toffoli(0, 1, 2)],
            3,
        )
        self.assert_alias_gate_state(
            [qitesse.Gate.x(0), qitesse.Gate.x(1), qitesse.Gate.cswap(0, 1, 2)],
            [qitesse.Gate.x(0), qitesse.Gate.x(1), qitesse.Gate.fredkin(0, 1, 2)],
            3,
        )
        self.assert_alias_gate_state(
            [qitesse.Gate.x(0), qitesse.Gate.x(1), qitesse.Gate.h(2), qitesse.Gate.mcp([0, 1], 2, 0.11)],
            [qitesse.Gate.x(0), qitesse.Gate.x(1), qitesse.Gate.h(2), qitesse.Gate.mcphase([0, 1], 2, 0.11)],
            3,
        )

    def test_custom_unitary_gate(self):
        hadamard = np.array([[1.0, 1.0], [1.0, -1.0]], dtype=np.complex64) / np.sqrt(2.0)
        self.assert_state_close(run_state([qitesse.Gate.unitary([0], hadamard)], 1), run_state([qitesse.Gate.h(0)], 1))

    def test_controlled_custom_unitary_gate(self):
        pauli_x = np.array([[0.0, 1.0], [1.0, 0.0]], dtype=np.complex64)
        self.assert_state_close(
            run_state([qitesse.Gate.h(0), qitesse.Gate.controlled_unitary([0], [1], pauli_x)], 2),
            run_state([qitesse.Gate.h(0), qitesse.Gate.cnot(0, 1)], 2),
        )

    def test_custom_unitary_validation_errors(self):
        with self.assertRaises(ValueError):
            qitesse.Gate.unitary([0], np.eye(3, dtype=np.complex64))

        with self.assertRaises(ValueError):
            qitesse.Gate.unitary([0], np.array([[1.0, 1.0], [0.0, 1.0]], dtype=np.complex64))

        with self.assertRaises(ValueError):
            qitesse.Gate.controlled_unitary([], [0], np.eye(2, dtype=np.complex64))

    def test_circuit_new_rejects_non_gates(self):
        with self.assertRaises(TypeError):
            qitesse.Circuit([object()])

    def test_circuit_run_method(self):
        state = qitesse.Circuit([qitesse.Gate.h(0)]).run(1)
        self.assertIsInstance(state, np.ndarray)
        self.assertEqual(state.dtype, np.complex64)
        self.assert_state_close(state, np.array([SQRT2_INV, SQRT2_INV], dtype=np.complex64))

    def test_run_with_measurements_and_barrier_and_reset(self):
        circuit = qitesse.Circuit([
            qitesse.Gate.x(0),
            qitesse.Gate.measure(0),
            qitesse.Gate.barrier(),
            qitesse.Gate.reset(0),
        ])

        state, measurements = circuit.run_with_measurements(1)
        self.assert_state_close(state, basis_state(1, 0))
        self.assertEqual(measurements, [(0, 1)])

    def test_sample_method(self):
        samples = qitesse.Circuit([qitesse.Gate.x(0)]).sample(1, 8)
        self.assertEqual(samples, [1] * 8)

    def test_set_num_threads_rejects_zero(self):
        with self.assertRaises(ValueError):
            qitesse.set_num_threads(0)

    def test_set_num_threads_accepts_positive_value_in_fresh_process(self):
        code = textwrap.dedent(
            """
            import qitesse
            qitesse.set_num_threads(1)
            print("ok")
            """
        )
        result = subprocess.run(
            [sys.executable, "-c", code],
            capture_output=True,
            text=True,
            check=False,
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(result.stdout.strip(), "ok")


if __name__ == "__main__":
    unittest.main()
