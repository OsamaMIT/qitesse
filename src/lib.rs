#![allow(non_local_definitions)]

use num_complex::Complex32;
use numpy::{IntoPyArray, PyArray1, PyArray2};
use pyo3::prelude::*;
use pyo3::types::{PyAny, PyList};
use rayon::ThreadPoolBuilder;
use std::f32::consts::PI;

mod sim;
use sim::{
    make_controlled_unitary_gate, make_unitary_gate, phase_matrix, with_global_phase, Circuit, Gate,
    StateVector,
};

fn gate_error(message: String) -> PyErr {
    PyErr::new::<pyo3::exceptions::PyValueError, _>(message)
}

fn matrix_from_python(py: Python<'_>, matrix: &PyAny) -> PyResult<Vec<Complex32>> {
    let numpy = py.import("numpy")?;
    let complex64 = numpy.getattr("complex64")?;
    let contiguous = numpy
        .getattr("ascontiguousarray")?
        .call1((matrix, complex64))?;

    let matrix = contiguous.downcast::<PyArray2<Complex32>>().map_err(|_| {
        PyErr::new::<pyo3::exceptions::PyTypeError, _>(
            "matrix must be convertible to a 2D numpy complex64 array",
        )
    })?;

    let shape = matrix.shape();
    if shape[0] != shape[1] {
        return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
            "matrix must be square",
        ));
    }

    let data = unsafe { matrix.as_slice() }
        .map_err(|_| {
            PyErr::new::<pyo3::exceptions::PyValueError, _>(
                "matrix must be stored contiguously in memory",
            )
        })?
        .to_vec();

    Ok(data)
}

/// Python wrapper for the Gate enum
#[pyclass(name = "Gate")]
#[derive(Clone)]
pub struct PyGate {
    gate: Gate,
}

impl PyGate {
    fn wrap(result: Result<Gate, String>) -> PyResult<Self> {
        result.map(|gate| Self { gate }).map_err(gate_error)
    }
}

#[pymethods]
impl PyGate {
    #[staticmethod]
    pub fn i(target: usize) -> PyResult<Self> {
        Self::wrap(make_unitary_gate(vec![target], sim::i_matrix()))
    }

    #[staticmethod]
    pub fn x(target: usize) -> PyResult<Self> {
        Self::wrap(make_unitary_gate(vec![target], sim::x_matrix()))
    }

    #[staticmethod]
    pub fn y(target: usize) -> PyResult<Self> {
        Self::wrap(make_unitary_gate(vec![target], sim::y_matrix()))
    }

    #[staticmethod]
    pub fn z(target: usize) -> PyResult<Self> {
        Self::wrap(make_unitary_gate(vec![target], sim::z_matrix()))
    }

    #[staticmethod]
    pub fn h(target: usize) -> PyResult<Self> {
        Self::wrap(make_unitary_gate(vec![target], sim::h_matrix()))
    }

    #[staticmethod]
    pub fn s(target: usize) -> PyResult<Self> {
        Self::wrap(make_unitary_gate(vec![target], phase_matrix(PI / 2.0)))
    }

    #[staticmethod]
    pub fn sdg(target: usize) -> PyResult<Self> {
        Self::wrap(make_unitary_gate(vec![target], phase_matrix(-PI / 2.0)))
    }

    #[staticmethod]
    pub fn t(target: usize) -> PyResult<Self> {
        Self::wrap(make_unitary_gate(vec![target], phase_matrix(PI / 4.0)))
    }

    #[staticmethod]
    pub fn tdg(target: usize) -> PyResult<Self> {
        Self::wrap(make_unitary_gate(vec![target], phase_matrix(-PI / 4.0)))
    }

    #[staticmethod]
    pub fn rx(target: usize, theta: f32) -> PyResult<Self> {
        Self::wrap(make_unitary_gate(vec![target], sim::rx_matrix(theta)))
    }

    #[staticmethod]
    pub fn ry(target: usize, theta: f32) -> PyResult<Self> {
        Self::wrap(make_unitary_gate(vec![target], sim::ry_matrix(theta)))
    }

    #[staticmethod]
    pub fn rz(target: usize, theta: f32) -> PyResult<Self> {
        Self::wrap(make_unitary_gate(vec![target], sim::rz_matrix(theta)))
    }

    #[staticmethod]
    pub fn p(target: usize, theta: f32) -> PyResult<Self> {
        Self::wrap(make_unitary_gate(vec![target], phase_matrix(theta)))
    }

    #[staticmethod]
    pub fn phase(target: usize, theta: f32) -> PyResult<Self> {
        Self::p(target, theta)
    }

    #[staticmethod]
    pub fn u(target: usize, theta: f32, phi: f32, lambda: f32) -> PyResult<Self> {
        Self::wrap(make_unitary_gate(vec![target], sim::u_matrix(theta, phi, lambda)))
    }

    #[staticmethod]
    pub fn cnot(control: usize, target: usize) -> PyResult<Self> {
        Self::wrap(make_controlled_unitary_gate(
            vec![control],
            vec![target],
            sim::x_matrix(),
        ))
    }

    #[staticmethod]
    pub fn cx(control: usize, target: usize) -> PyResult<Self> {
        Self::cnot(control, target)
    }

    #[staticmethod]
    pub fn cy(control: usize, target: usize) -> PyResult<Self> {
        Self::wrap(make_controlled_unitary_gate(
            vec![control],
            vec![target],
            sim::y_matrix(),
        ))
    }

    #[staticmethod]
    pub fn cz(control: usize, target: usize) -> PyResult<Self> {
        Self::wrap(make_controlled_unitary_gate(
            vec![control],
            vec![target],
            sim::z_matrix(),
        ))
    }

    #[staticmethod]
    pub fn ch(control: usize, target: usize) -> PyResult<Self> {
        Self::wrap(make_controlled_unitary_gate(
            vec![control],
            vec![target],
            sim::h_matrix(),
        ))
    }

    #[staticmethod]
    pub fn swap(a: usize, b: usize) -> PyResult<Self> {
        Self::wrap(make_unitary_gate(vec![a, b], sim::swap_matrix()))
    }

    #[staticmethod]
    pub fn iswap(a: usize, b: usize) -> PyResult<Self> {
        Self::wrap(make_unitary_gate(vec![a, b], sim::iswap_matrix()))
    }

    #[staticmethod]
    pub fn crx(control: usize, target: usize, theta: f32) -> PyResult<Self> {
        Self::wrap(make_controlled_unitary_gate(
            vec![control],
            vec![target],
            sim::rx_matrix(theta),
        ))
    }

    #[staticmethod]
    pub fn cry(control: usize, target: usize, theta: f32) -> PyResult<Self> {
        Self::wrap(make_controlled_unitary_gate(
            vec![control],
            vec![target],
            sim::ry_matrix(theta),
        ))
    }

    #[staticmethod]
    pub fn crz(control: usize, target: usize, theta: f32) -> PyResult<Self> {
        Self::wrap(make_controlled_unitary_gate(
            vec![control],
            vec![target],
            sim::rz_matrix(theta),
        ))
    }

    #[staticmethod]
    pub fn cp(control: usize, target: usize, theta: f32) -> PyResult<Self> {
        Self::wrap(make_controlled_unitary_gate(
            vec![control],
            vec![target],
            phase_matrix(theta),
        ))
    }

    #[staticmethod]
    pub fn cphase(control: usize, target: usize, theta: f32) -> PyResult<Self> {
        Self::cp(control, target, theta)
    }

    #[staticmethod]
    #[pyo3(signature = (control, target, theta, phi, lambda, gamma=None))]
    pub fn cu(
        control: usize,
        target: usize,
        theta: f32,
        phi: f32,
        lambda: f32,
        gamma: Option<f32>,
    ) -> PyResult<Self> {
        let base = sim::u_matrix(theta, phi, lambda);
        let matrix = with_global_phase(&base, gamma.unwrap_or(0.0));
        Self::wrap(make_controlled_unitary_gate(vec![control], vec![target], matrix))
    }

    #[staticmethod]
    pub fn ccx(control_a: usize, control_b: usize, target: usize) -> PyResult<Self> {
        Self::wrap(make_controlled_unitary_gate(
            vec![control_a, control_b],
            vec![target],
            sim::x_matrix(),
        ))
    }

    #[staticmethod]
    pub fn toffoli(control_a: usize, control_b: usize, target: usize) -> PyResult<Self> {
        Self::ccx(control_a, control_b, target)
    }

    #[staticmethod]
    pub fn cswap(control: usize, a: usize, b: usize) -> PyResult<Self> {
        Self::wrap(make_controlled_unitary_gate(
            vec![control],
            vec![a, b],
            sim::swap_matrix(),
        ))
    }

    #[staticmethod]
    pub fn fredkin(control: usize, a: usize, b: usize) -> PyResult<Self> {
        Self::cswap(control, a, b)
    }

    #[staticmethod]
    pub fn mcx(controls: Vec<usize>, target: usize) -> PyResult<Self> {
        Self::wrap(make_controlled_unitary_gate(
            controls,
            vec![target],
            sim::x_matrix(),
        ))
    }

    #[staticmethod]
    pub fn mcz(controls: Vec<usize>, target: usize) -> PyResult<Self> {
        Self::wrap(make_controlled_unitary_gate(
            controls,
            vec![target],
            sim::z_matrix(),
        ))
    }

    #[staticmethod]
    pub fn mcp(controls: Vec<usize>, target: usize, theta: f32) -> PyResult<Self> {
        Self::wrap(make_controlled_unitary_gate(
            controls,
            vec![target],
            phase_matrix(theta),
        ))
    }

    #[staticmethod]
    pub fn mcphase(controls: Vec<usize>, target: usize, theta: f32) -> PyResult<Self> {
        Self::mcp(controls, target, theta)
    }

    #[staticmethod]
    pub fn unitary(py: Python<'_>, targets: Vec<usize>, matrix: &PyAny) -> PyResult<Self> {
        let matrix = matrix_from_python(py, matrix)?;
        Self::wrap(make_unitary_gate(targets, matrix))
    }

    #[staticmethod]
    pub fn controlled_unitary(
        py: Python<'_>,
        controls: Vec<usize>,
        targets: Vec<usize>,
        matrix: &PyAny,
    ) -> PyResult<Self> {
        let matrix = matrix_from_python(py, matrix)?;
        Self::wrap(make_controlled_unitary_gate(controls, targets, matrix))
    }

    #[staticmethod]
    pub fn measure(target: usize) -> Self {
        Self {
            gate: Gate::Measure(target),
        }
    }

    #[staticmethod]
    pub fn reset(target: usize) -> Self {
        Self {
            gate: Gate::Reset(target),
        }
    }

    #[staticmethod]
    pub fn barrier() -> Self {
        Self {
            gate: Gate::Barrier,
        }
    }
}

/// Python wrapper for the Circuit struct
#[pyclass(name = "Circuit")]
pub struct PyCircuit {
    circuit: Circuit,
}

#[pymethods]
impl PyCircuit {
    #[new]
    pub fn new(gates: &PyList) -> PyResult<Self> {
        let rust_gates: PyResult<Vec<Gate>> = gates
            .iter()
            .map(|item| {
                item.extract::<PyRef<PyGate>>()
                    .map(|gate| gate.gate.clone())
                    .map_err(|_| {
                        PyErr::new::<pyo3::exceptions::PyTypeError, _>(
                            "all circuit items must be qitesse.Gate instances",
                        )
                    })
            })
            .collect();

        Ok(Self {
            circuit: Circuit::new(rust_gates?),
        })
    }

    pub fn run(&self, py: Python<'_>, num_qubits: usize) -> PyResult<Py<PyArray1<Complex32>>> {
        let mut state_vector = StateVector::new(num_qubits);
        self.circuit.run(&mut state_vector).map_err(gate_error)?;
        Ok(state_vector.amps.into_pyarray(py).to_owned())
    }

    pub fn run_with_measurements(
        &self,
        py: Python<'_>,
        num_qubits: usize,
    ) -> PyResult<(Py<PyArray1<Complex32>>, Vec<(usize, u8)>)> {
        let mut state_vector = StateVector::new(num_qubits);
        let measurements = self.circuit.run(&mut state_vector).map_err(gate_error)?;
        Ok((state_vector.amps.into_pyarray(py).to_owned(), measurements))
    }

    pub fn sample(&self, num_qubits: usize, shots: usize) -> PyResult<Vec<usize>> {
        let mut state_vector = StateVector::new(num_qubits);
        self.circuit.run(&mut state_vector).map_err(gate_error)?;
        Ok(state_vector.measure(shots))
    }
}

/// Set the number of threads for Rayon
#[pyfunction]
fn set_num_threads(num_threads: usize) -> PyResult<()> {
    if num_threads == 0 {
        return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
            "number of threads must be greater than 0",
        ));
    }

    ThreadPoolBuilder::new()
        .num_threads(num_threads)
        .build_global()
        .map_err(|error| {
            PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(format!(
                "failed to set number of threads: {}",
                error
            ))
        })?;

    Ok(())
}

#[pymodule]
fn qitesse(_py: Python<'_>, m: &PyModule) -> PyResult<()> {
    m.add_class::<PyGate>()?;
    m.add_class::<PyCircuit>()?;
    m.add_function(wrap_pyfunction!(set_num_threads, m)?)?;
    Ok(())
}
