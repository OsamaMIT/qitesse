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
    ExecutionContext, Observable, ParamGateKind, ParameterizedCircuitSpec, Pauli, PauliTerm,
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

fn params_from_python(py: Python<'_>, params: &PyAny) -> PyResult<Vec<f32>> {
    let numpy = py.import("numpy")?;
    let float32 = numpy.getattr("float32")?;
    let contiguous = numpy
        .getattr("ascontiguousarray")?
        .call1((params, float32))?;

    let params = contiguous.downcast::<PyArray1<f32>>().map_err(|_| {
        PyErr::new::<pyo3::exceptions::PyTypeError, _>(
            "params must be convertible to a 1D numpy float32 array",
        )
    })?;

    let data = unsafe { params.as_slice() }
        .map_err(|_| {
            PyErr::new::<pyo3::exceptions::PyValueError, _>(
                "params must be stored contiguously in memory",
            )
        })?
        .to_vec();

    Ok(data)
}

fn params_batch_from_python(py: Python<'_>, params: &PyAny) -> PyResult<(Vec<f32>, usize, usize)> {
    let numpy = py.import("numpy")?;
    let float32 = numpy.getattr("float32")?;
    let contiguous = numpy
        .getattr("ascontiguousarray")?
        .call1((params, float32))?;

    let params = contiguous.downcast::<PyArray2<f32>>().map_err(|_| {
        PyErr::new::<pyo3::exceptions::PyTypeError, _>(
            "params_batch must be convertible to a 2D numpy float32 array",
        )
    })?;

    let shape = params.shape();
    let data = unsafe { params.as_slice() }
        .map_err(|_| {
            PyErr::new::<pyo3::exceptions::PyValueError, _>(
                "params_batch must be stored contiguously in memory",
            )
        })?
        .to_vec();

    Ok((data, shape[0], shape[1]))
}

fn validate_param_width(width: usize, expected: usize, label: &str) -> PyResult<()> {
    if width != expected {
        return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(format!(
            "{} has width {}, expected {}",
            label, width, expected
        )));
    }
    Ok(())
}

fn gradients_into_pyarray(
    py: Python<'_>,
    gradients: Vec<Vec<f32>>,
    rows: usize,
    cols: usize,
) -> PyResult<Py<PyArray2<f32>>> {
    if rows == 0 || cols == 0 {
        return Ok(PyArray2::<f32>::zeros(py, [rows, cols], false).to_owned());
    }

    PyArray2::from_vec2(py, &gradients)
        .map(|array| array.to_owned())
        .map_err(|err| {
            PyErr::new::<pyo3::exceptions::PyValueError, _>(format!(
                "failed to build gradient array: {}",
                err
            ))
        })
}

fn param_buffer_slice<'py>(
    py: Python<'py>,
    buffer: &'py PyParamBuffer,
    expected: usize,
) -> PyResult<&'py [f32]> {
    if buffer.size != expected {
        return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(format!(
            "param_buffer has length {}, expected {}",
            buffer.size, expected
        )));
    }

    let array = buffer.array.as_ref(py);
    unsafe { array.as_slice() }.map_err(|_| {
        PyErr::new::<pyo3::exceptions::PyValueError, _>(
            "param_buffer must be stored contiguously in memory",
        )
    })
}

fn param_batch_buffer_slice<'py>(
    py: Python<'py>,
    buffer: &'py PyParamBatchBuffer,
    expected_width: usize,
) -> PyResult<&'py [f32]> {
    if buffer.parameter_count != expected_width {
        return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(format!(
            "param_batch_buffer has width {}, expected {}",
            buffer.parameter_count, expected_width
        )));
    }

    let array = buffer.array.as_ref(py);
    unsafe { array.as_slice() }.map_err(|_| {
        PyErr::new::<pyo3::exceptions::PyValueError, _>(
            "param_batch_buffer must be stored contiguously in memory",
        )
    })
}

/// Gate constructors for one-off circuit building and custom unitary definition.
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

/// Symbolic parameter handle used when building a compiled circuit specification.
#[pyclass(name = "Parameter")]
#[derive(Clone)]
pub struct PyParameter {
    index: usize,
    name: Option<String>,
}

#[pymethods]
impl PyParameter {
    #[getter]
    /// Stable zero-based slot used when binding parameter vectors.
    pub fn index(&self) -> usize {
        self.index
    }

    #[getter]
    /// Optional user-provided parameter name.
    pub fn name(&self) -> Option<String> {
        self.name.clone()
    }

    pub fn __repr__(&self) -> String {
        match &self.name {
            Some(name) => format!("Parameter(index={}, name='{}')", self.index, name),
            None => format!("Parameter(index={})", self.index),
        }
    }
}

/// Pauli observables and Hamiltonians for expectation-value workflows.
#[pyclass(name = "Observable")]
#[derive(Clone)]
pub struct PyObservable {
    observable: Observable,
}

impl PyObservable {
    fn wrap(result: Result<Observable, String>) -> PyResult<Self> {
        result
            .map(|observable| Self { observable })
            .map_err(gate_error)
    }

    fn single(pauli: Pauli, qubit: usize, coefficient: f32) -> PyResult<Self> {
        Self::wrap(Observable::single(pauli, qubit, coefficient))
    }
}

#[pymethods]
impl PyObservable {
    #[staticmethod]
    #[pyo3(signature = (qubit, coefficient=1.0))]
    pub fn pauli_x(qubit: usize, coefficient: f32) -> PyResult<Self> {
        Self::single(Pauli::X, qubit, coefficient)
    }

    #[staticmethod]
    #[pyo3(signature = (qubit, coefficient=1.0))]
    pub fn pauli_y(qubit: usize, coefficient: f32) -> PyResult<Self> {
        Self::single(Pauli::Y, qubit, coefficient)
    }

    #[staticmethod]
    #[pyo3(signature = (qubit, coefficient=1.0))]
    pub fn pauli_z(qubit: usize, coefficient: f32) -> PyResult<Self> {
        Self::single(Pauli::Z, qubit, coefficient)
    }

    #[staticmethod]
    #[pyo3(signature = (ops, coefficient=1.0))]
    pub fn pauli_string(ops: Vec<(String, usize)>, coefficient: f32) -> PyResult<Self> {
        let mut parsed = Vec::with_capacity(ops.len());
        for (label, qubit) in ops {
            let pauli = match label.as_str() {
                "X" | "x" => Pauli::X,
                "Y" | "y" => Pauli::Y,
                "Z" | "z" => Pauli::Z,
                _ => {
                    return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(format!(
                        "unsupported Pauli label '{}'; expected X, Y, or Z",
                        label
                    )))
                }
            };
            parsed.push((qubit, pauli));
        }

        let term = PauliTerm::new(coefficient, parsed).map_err(gate_error)?;
        Self::wrap(Observable::new(vec![term]))
    }

    #[staticmethod]
    pub fn hamiltonian(terms: &PyList) -> PyResult<Self> {
        let mut flattened = Vec::new();
        for item in terms.iter() {
            let observable = item.extract::<PyRef<PyObservable>>().map_err(|_| {
                PyErr::new::<pyo3::exceptions::PyTypeError, _>(
                    "hamiltonian terms must all be qitesse.Observable instances",
                )
            })?;
            flattened.extend(observable.observable.terms.clone());
        }

        Self::wrap(Observable::new(flattened))
    }
}

/// Reusable compiled execution plan for repeated parameterized circuit evaluation.
#[pyclass(name = "CompiledCircuit")]
pub struct PyCompiledCircuit {
    circuit: sim::CompiledCircuit,
}

/// Reusable parameter buffer for zero-copy compiled scalar execution.
#[pyclass(name = "ParamBuffer")]
pub struct PyParamBuffer {
    array: Py<PyArray1<f32>>,
    size: usize,
}

/// Reusable parameter batch buffer for zero-copy compiled batch execution.
#[pyclass(name = "ParamBatchBuffer")]
pub struct PyParamBatchBuffer {
    array: Py<PyArray2<f32>>,
    batch_size: usize,
    parameter_count: usize,
}

/// Reusable execution buffer for repeated scalar compiled-circuit calls.
#[pyclass(name = "ExecutionContext")]
pub struct PyExecutionContext {
    context: ExecutionContext,
}

#[pymethods]
impl PyParamBuffer {
    #[getter]
    /// Number of parameters stored in the buffer.
    pub fn size(&self) -> usize {
        self.size
    }

    /// Return the writable NumPy array backing this buffer.
    pub fn numpy(&self, py: Python<'_>) -> Py<PyArray1<f32>> {
        self.array.clone_ref(py)
    }
}

#[pymethods]
impl PyParamBatchBuffer {
    #[getter]
    /// Number of parameter vectors stored in the batch buffer.
    pub fn batch_size(&self) -> usize {
        self.batch_size
    }

    #[getter]
    /// Number of parameters in each row of the batch buffer.
    pub fn parameter_count(&self) -> usize {
        self.parameter_count
    }

    /// Return the writable NumPy array backing this batch buffer.
    pub fn numpy(&self, py: Python<'_>) -> Py<PyArray2<f32>> {
        self.array.clone_ref(py)
    }
}

#[pymethods]
impl PyCompiledCircuit {
    #[getter]
    /// Number of qubits in the compiled circuit.
    pub fn num_qubits(&self) -> usize {
        self.circuit.num_qubits
    }

    #[getter]
    /// Number of parameters expected in each bound parameter vector.
    pub fn parameter_count(&self) -> usize {
        self.circuit.num_params
    }

    pub fn statevector(
        &self,
        py: Python<'_>,
        params: &PyAny,
    ) -> PyResult<Py<PyArray1<Complex32>>> {
        let params = params_from_python(py, params)?;
        let amplitudes = py
            .allow_threads(|| self.circuit.run_statevector(&params).map(|state| state.amps))
            .map_err(gate_error)?;
        Ok(amplitudes.into_pyarray(py).to_owned())
    }

    pub fn expectation(
        &self,
        py: Python<'_>,
        params: &PyAny,
        observable: PyRef<'_, PyObservable>,
    ) -> PyResult<f32> {
        let params = params_from_python(py, params)?;
        let observable = observable.observable.clone();
        py.allow_threads(|| self.circuit.expectation(&params, &observable))
            .map_err(gate_error)
    }

    pub fn batch_expectation(
        &self,
        py: Python<'_>,
        params_batch: &PyAny,
        observable: PyRef<'_, PyObservable>,
    ) -> PyResult<Py<PyArray1<f32>>> {
        let (params_batch, batch_size, width) = params_batch_from_python(py, params_batch)?;
        validate_param_width(width, self.circuit.num_params, "params_batch")?;
        let observable = observable.observable.clone();
        let values = py
            .allow_threads(|| self.circuit.batch_expectation(&params_batch, batch_size, &observable))
            .map_err(gate_error)?;
        Ok(values.into_pyarray(py).to_owned())
    }

    /// Allocate a reusable parameter buffer backed by a NumPy array.
    pub fn param_buffer(&self, py: Python<'_>) -> PyParamBuffer {
        PyParamBuffer {
            array: PyArray1::<f32>::zeros(py, self.circuit.num_params, false).to_owned(),
            size: self.circuit.num_params,
        }
    }

    /// Allocate a reusable parameter batch buffer backed by a NumPy array.
    pub fn param_batch_buffer(&self, py: Python<'_>, batch_size: usize) -> PyParamBatchBuffer {
        PyParamBatchBuffer {
            array: PyArray2::<f32>::zeros(py, [batch_size, self.circuit.num_params], false)
                .to_owned(),
            batch_size,
            parameter_count: self.circuit.num_params,
        }
    }

    pub fn statevector_buffer(
        &self,
        py: Python<'_>,
        params: PyRef<'_, PyParamBuffer>,
    ) -> PyResult<Py<PyArray1<Complex32>>> {
        let params = param_buffer_slice(py, &params, self.circuit.num_params)?;
        let state = self.circuit.run_statevector(params).map_err(gate_error)?;
        Ok(state.amps.into_pyarray(py).to_owned())
    }

    pub fn expectation_buffer(
        &self,
        py: Python<'_>,
        params: PyRef<'_, PyParamBuffer>,
        observable: PyRef<'_, PyObservable>,
    ) -> PyResult<f32> {
        let params = param_buffer_slice(py, &params, self.circuit.num_params)?;
        let observable = observable.observable.clone();
        self.circuit.expectation(params, &observable).map_err(gate_error)
    }

    pub fn gradient(
        &self,
        py: Python<'_>,
        params: &PyAny,
        observable: PyRef<'_, PyObservable>,
    ) -> PyResult<Py<PyArray1<f32>>> {
        let params = params_from_python(py, params)?;
        let observable = observable.observable.clone();
        let gradient = py
            .allow_threads(|| self.circuit.gradient(&params, &observable))
            .map_err(gate_error)?;
        Ok(gradient.into_pyarray(py).to_owned())
    }

    pub fn gradient_buffer(
        &self,
        py: Python<'_>,
        params: PyRef<'_, PyParamBuffer>,
        observable: PyRef<'_, PyObservable>,
    ) -> PyResult<Py<PyArray1<f32>>> {
        let params = param_buffer_slice(py, &params, self.circuit.num_params)?;
        let observable = observable.observable.clone();
        let gradient = self.circuit.gradient(params, &observable).map_err(gate_error)?;
        Ok(gradient.into_pyarray(py).to_owned())
    }

    pub fn value_and_gradient(
        &self,
        py: Python<'_>,
        params: &PyAny,
        observable: PyRef<'_, PyObservable>,
    ) -> PyResult<(f32, Py<PyArray1<f32>>)> {
        let params = params_from_python(py, params)?;
        let observable = observable.observable.clone();
        let (value, gradient) = py
            .allow_threads(|| self.circuit.value_and_gradient(&params, &observable))
            .map_err(gate_error)?;
        Ok((value, gradient.into_pyarray(py).to_owned()))
    }

    pub fn value_and_gradient_buffer(
        &self,
        py: Python<'_>,
        params: PyRef<'_, PyParamBuffer>,
        observable: PyRef<'_, PyObservable>,
    ) -> PyResult<(f32, Py<PyArray1<f32>>)> {
        let params = param_buffer_slice(py, &params, self.circuit.num_params)?;
        let observable = observable.observable.clone();
        let (value, gradient) = self
            .circuit
            .value_and_gradient(params, &observable)
            .map_err(gate_error)?;
        Ok((value, gradient.into_pyarray(py).to_owned()))
    }

    pub fn batch_gradient(
        &self,
        py: Python<'_>,
        params_batch: &PyAny,
        observable: PyRef<'_, PyObservable>,
    ) -> PyResult<Py<PyArray2<f32>>> {
        let (params_batch, batch_size, width) = params_batch_from_python(py, params_batch)?;
        validate_param_width(width, self.circuit.num_params, "params_batch")?;
        let observable = observable.observable.clone();
        let gradients = py
            .allow_threads(|| self.circuit.batch_gradient(&params_batch, batch_size, &observable))
            .map_err(gate_error)?;
        gradients_into_pyarray(py, gradients, batch_size, self.circuit.num_params)
    }

    pub fn batch_expectation_buffer(
        &self,
        py: Python<'_>,
        params_batch: PyRef<'_, PyParamBatchBuffer>,
        observable: PyRef<'_, PyObservable>,
    ) -> PyResult<Py<PyArray1<f32>>> {
        let params = param_batch_buffer_slice(py, &params_batch, self.circuit.num_params)?;
        let observable = observable.observable.clone();
        let values = self
            .circuit
            .batch_expectation(params, params_batch.batch_size, &observable)
            .map_err(gate_error)?;
        Ok(values.into_pyarray(py).to_owned())
    }

    pub fn batch_gradient_buffer(
        &self,
        py: Python<'_>,
        params_batch: PyRef<'_, PyParamBatchBuffer>,
        observable: PyRef<'_, PyObservable>,
    ) -> PyResult<Py<PyArray2<f32>>> {
        let params = param_batch_buffer_slice(py, &params_batch, self.circuit.num_params)?;
        let observable = observable.observable.clone();
        let gradients = self
            .circuit
            .batch_gradient(params, params_batch.batch_size, &observable)
            .map_err(gate_error)?;
        gradients_into_pyarray(py, gradients, params_batch.batch_size, self.circuit.num_params)
    }

    pub fn batch_value_and_gradient(
        &self,
        py: Python<'_>,
        params_batch: &PyAny,
        observable: PyRef<'_, PyObservable>,
    ) -> PyResult<(Py<PyArray1<f32>>, Py<PyArray2<f32>>)> {
        let (params_batch, batch_size, width) = params_batch_from_python(py, params_batch)?;
        validate_param_width(width, self.circuit.num_params, "params_batch")?;
        let observable = observable.observable.clone();
        let (values, gradients) = py
            .allow_threads(|| self.circuit.batch_value_and_gradient(&params_batch, batch_size, &observable))
            .map_err(gate_error)?;
        let values = values.into_pyarray(py).to_owned();
        let gradients = gradients_into_pyarray(py, gradients, batch_size, self.circuit.num_params)?;
        Ok((values, gradients))
    }

    pub fn batch_value_and_gradient_buffer(
        &self,
        py: Python<'_>,
        params_batch: PyRef<'_, PyParamBatchBuffer>,
        observable: PyRef<'_, PyObservable>,
    ) -> PyResult<(Py<PyArray1<f32>>, Py<PyArray2<f32>>)> {
        let params = param_batch_buffer_slice(py, &params_batch, self.circuit.num_params)?;
        let observable = observable.observable.clone();
        let (values, gradients) = self
            .circuit
            .batch_value_and_gradient(params, params_batch.batch_size, &observable)
            .map_err(gate_error)?;
        let values = values.into_pyarray(py).to_owned();
        let gradients =
            gradients_into_pyarray(py, gradients, params_batch.batch_size, self.circuit.num_params)?;
        Ok((values, gradients))
    }

    pub fn execution_context(&self) -> PyExecutionContext {
        PyExecutionContext {
            context: self.circuit.execution_context(),
        }
    }
}

#[pymethods]
impl PyExecutionContext {
    pub fn statevector_buffer(
        &mut self,
        py: Python<'_>,
        circuit: PyRef<'_, PyCompiledCircuit>,
        params: PyRef<'_, PyParamBuffer>,
    ) -> PyResult<Py<PyArray1<Complex32>>> {
        let params = param_buffer_slice(py, &params, circuit.circuit.num_params)?;
        let compiled = circuit.circuit.clone();
        let state = compiled
            .run_statevector_with_context(params, &mut self.context)
            .map_err(gate_error)?;
        Ok(state.amps.clone().into_pyarray(py).to_owned())
    }

    pub fn statevector(
        &mut self,
        py: Python<'_>,
        circuit: PyRef<'_, PyCompiledCircuit>,
        params: &PyAny,
    ) -> PyResult<Py<PyArray1<Complex32>>> {
        let params = params_from_python(py, params)?;
        let compiled = circuit.circuit.clone();
        let amplitudes = py
            .allow_threads(|| {
                compiled
                    .run_statevector_with_context(&params, &mut self.context)
                    .map(|state| state.amps.clone())
            })
            .map_err(gate_error)?;
        Ok(amplitudes.into_pyarray(py).to_owned())
    }

    pub fn expectation(
        &mut self,
        py: Python<'_>,
        circuit: PyRef<'_, PyCompiledCircuit>,
        params: &PyAny,
        observable: PyRef<'_, PyObservable>,
    ) -> PyResult<f32> {
        let params = params_from_python(py, params)?;
        let compiled = circuit.circuit.clone();
        let observable = observable.observable.clone();
        py.allow_threads(|| compiled.expectation_with_context(&params, &observable, &mut self.context))
        .map_err(gate_error)
    }

    pub fn expectation_buffer(
        &mut self,
        py: Python<'_>,
        circuit: PyRef<'_, PyCompiledCircuit>,
        params: PyRef<'_, PyParamBuffer>,
        observable: PyRef<'_, PyObservable>,
    ) -> PyResult<f32> {
        let params = param_buffer_slice(py, &params, circuit.circuit.num_params)?;
        let compiled = circuit.circuit.clone();
        let observable = observable.observable.clone();
        compiled
            .expectation_with_context(params, &observable, &mut self.context)
            .map_err(gate_error)
    }

    pub fn gradient(
        &mut self,
        py: Python<'_>,
        circuit: PyRef<'_, PyCompiledCircuit>,
        params: &PyAny,
        observable: PyRef<'_, PyObservable>,
    ) -> PyResult<Py<PyArray1<f32>>> {
        let params = params_from_python(py, params)?;
        let compiled = circuit.circuit.clone();
        let observable = observable.observable.clone();
        let gradient = py
            .allow_threads(|| compiled.gradient_with_context(&params, &observable, &mut self.context))
            .map_err(gate_error)?;
        Ok(gradient.into_pyarray(py).to_owned())
    }

    pub fn gradient_buffer(
        &mut self,
        py: Python<'_>,
        circuit: PyRef<'_, PyCompiledCircuit>,
        params: PyRef<'_, PyParamBuffer>,
        observable: PyRef<'_, PyObservable>,
    ) -> PyResult<Py<PyArray1<f32>>> {
        let params = param_buffer_slice(py, &params, circuit.circuit.num_params)?;
        let compiled = circuit.circuit.clone();
        let observable = observable.observable.clone();
        let gradient = compiled
            .gradient_with_context(params, &observable, &mut self.context)
            .map_err(gate_error)?;
        Ok(gradient.into_pyarray(py).to_owned())
    }

    pub fn value_and_gradient(
        &mut self,
        py: Python<'_>,
        circuit: PyRef<'_, PyCompiledCircuit>,
        params: &PyAny,
        observable: PyRef<'_, PyObservable>,
    ) -> PyResult<(f32, Py<PyArray1<f32>>)> {
        let params = params_from_python(py, params)?;
        let compiled = circuit.circuit.clone();
        let observable = observable.observable.clone();
        let (value, gradient) = py
            .allow_threads(|| {
                compiled.value_and_gradient_with_context(&params, &observable, &mut self.context)
            })
            .map_err(gate_error)?;
        Ok((value, gradient.into_pyarray(py).to_owned()))
    }

    pub fn value_and_gradient_buffer(
        &mut self,
        py: Python<'_>,
        circuit: PyRef<'_, PyCompiledCircuit>,
        params: PyRef<'_, PyParamBuffer>,
        observable: PyRef<'_, PyObservable>,
    ) -> PyResult<(f32, Py<PyArray1<f32>>)> {
        let params = param_buffer_slice(py, &params, circuit.circuit.num_params)?;
        let compiled = circuit.circuit.clone();
        let observable = observable.observable.clone();
        let (value, gradient) = compiled
            .value_and_gradient_with_context(params, &observable, &mut self.context)
            .map_err(gate_error)?;
        Ok((value, gradient.into_pyarray(py).to_owned()))
    }
}

/// Builder for reusable parameterized circuit structure.
#[pyclass(name = "CircuitSpec")]
pub struct PyCircuitSpec {
    spec: ParameterizedCircuitSpec,
}

impl PyCircuitSpec {
    fn add_fixed_gate(&mut self, gate: Result<Gate, String>) -> PyResult<()> {
        self.spec
            .add_fixed_gate(gate.map_err(gate_error)?)
            .map_err(gate_error)
    }

    fn add_param_gate(
        &mut self,
        kind: ParamGateKind,
        target: usize,
        param: PyRef<'_, PyParameter>,
    ) -> PyResult<()> {
        self.spec
            .add_param_single(kind, target, param.index)
            .map_err(gate_error)
    }
}

#[pymethods]
impl PyCircuitSpec {
    #[new]
    pub fn new(num_qubits: usize) -> PyResult<Self> {
        Ok(Self {
            spec: ParameterizedCircuitSpec::new(num_qubits).map_err(gate_error)?,
        })
    }

    #[getter]
    /// Number of qubits declared for the circuit specification.
    pub fn num_qubits(&self) -> usize {
        self.spec.num_qubits
    }

    #[getter]
    /// Number of symbolic parameters currently registered on the specification.
    pub fn parameter_count(&self) -> usize {
        self.spec.param_names.len()
    }

    /// Register a new symbolic parameter and return its handle.
    #[pyo3(signature = (name=None))]
    pub fn param(&mut self, name: Option<String>) -> PyParameter {
        let index = self.spec.add_parameter(name.clone());
        PyParameter { index, name }
    }

    pub fn x(&mut self, target: usize) -> PyResult<()> {
        self.add_fixed_gate(make_unitary_gate(vec![target], sim::x_matrix()))
    }

    pub fn h(&mut self, target: usize) -> PyResult<()> {
        self.add_fixed_gate(make_unitary_gate(vec![target], sim::h_matrix()))
    }

    pub fn z(&mut self, target: usize) -> PyResult<()> {
        self.add_fixed_gate(make_unitary_gate(vec![target], sim::z_matrix()))
    }

    pub fn cnot(&mut self, control: usize, target: usize) -> PyResult<()> {
        self.add_fixed_gate(make_controlled_unitary_gate(
            vec![control],
            vec![target],
            sim::x_matrix(),
        ))
    }

    pub fn cx(&mut self, control: usize, target: usize) -> PyResult<()> {
        self.cnot(control, target)
    }

    pub fn cz(&mut self, control: usize, target: usize) -> PyResult<()> {
        self.add_fixed_gate(make_controlled_unitary_gate(
            vec![control],
            vec![target],
            sim::z_matrix(),
        ))
    }

    pub fn rx(&mut self, target: usize, param: PyRef<'_, PyParameter>) -> PyResult<()> {
        self.add_param_gate(ParamGateKind::Rx, target, param)
    }

    pub fn ry(&mut self, target: usize, param: PyRef<'_, PyParameter>) -> PyResult<()> {
        self.add_param_gate(ParamGateKind::Ry, target, param)
    }

    pub fn rz(&mut self, target: usize, param: PyRef<'_, PyParameter>) -> PyResult<()> {
        self.add_param_gate(ParamGateKind::Rz, target, param)
    }

    pub fn p(&mut self, target: usize, param: PyRef<'_, PyParameter>) -> PyResult<()> {
        self.add_param_gate(ParamGateKind::Phase, target, param)
    }

    /// Compile the circuit specification into a reusable execution plan.
    pub fn compile(&self) -> PyResult<PyCompiledCircuit> {
        Ok(PyCompiledCircuit {
            circuit: self.spec.compile().map_err(gate_error)?,
        })
    }
}

/// One-off gate-by-gate circuit execution API.
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
    m.add_class::<PyParameter>()?;
    m.add_class::<PyObservable>()?;
    m.add_class::<PyCompiledCircuit>()?;
    m.add_class::<PyParamBuffer>()?;
    m.add_class::<PyParamBatchBuffer>()?;
    m.add_class::<PyExecutionContext>()?;
    m.add_class::<PyCircuitSpec>()?;
    m.add_class::<PyCircuit>()?;
    m.add_function(wrap_pyfunction!(set_num_threads, m)?)?;
    Ok(())
}
