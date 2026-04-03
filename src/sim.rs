use num_complex::Complex;
use rand::distributions::WeightedIndex;
use rand::prelude::*;
use rayon::prelude::*;

pub type C32 = Complex<f32>;

fn c32(re: f32, im: f32) -> C32 {
    Complex::new(re, im)
}

fn exp_i(theta: f32) -> C32 {
    Complex::from_polar(1.0f32, theta)
}

fn matrix_dim(num_qubits: usize) -> Result<usize, String> {
    1usize
        .checked_shl(num_qubits as u32)
        .ok_or_else(|| format!("too many qubits in operation: {}", num_qubits))
}

fn ensure_unique_qubits(qubits: &[usize]) -> Result<(), String> {
    for (i, &lhs) in qubits.iter().enumerate() {
        for &rhs in &qubits[(i + 1)..] {
            if lhs == rhs {
                return Err(format!("duplicate qubit index: {}", lhs));
            }
        }
    }
    Ok(())
}

fn is_unitary(matrix: &[C32], dim: usize, tolerance: f32) -> bool {
    for row in 0..dim {
        for col in 0..dim {
            let mut sum = c32(0.0, 0.0);
            for k in 0..dim {
                sum += matrix[row * dim + k] * matrix[col * dim + k].conj();
            }
            let expected = if row == col { c32(1.0, 0.0) } else { c32(0.0, 0.0) };
            if (sum - expected).norm() > tolerance {
                return false;
            }
        }
    }
    true
}

fn identity_matrix(dim: usize) -> Vec<C32> {
    let mut matrix = vec![c32(0.0, 0.0); dim * dim];
    for i in 0..dim {
        matrix[i * dim + i] = c32(1.0, 0.0);
    }
    matrix
}

fn clear_target_bits(mut index: usize, targets: &[usize]) -> usize {
    for &target in targets {
        index &= !(1usize << target);
    }
    index
}

fn extract_sub_index(index: usize, targets: &[usize]) -> usize {
    let width = targets.len();
    let mut sub_index = 0usize;
    for (position, &target) in targets.iter().enumerate() {
        let bit = (index >> target) & 1usize;
        sub_index |= bit << (width - 1 - position);
    }
    sub_index
}

fn compose_index(base: usize, targets: &[usize], sub_index: usize) -> usize {
    let width = targets.len();
    let mut index = base;
    for (position, &target) in targets.iter().enumerate() {
        let bit = (sub_index >> (width - 1 - position)) & 1usize;
        if bit == 1 {
            index |= 1usize << target;
        } else {
            index &= !(1usize << target);
        }
    }
    index
}

#[derive(Copy, Clone)]
pub struct Mat2 {
    pub m00: C32,
    pub m01: C32,
    pub m10: C32,
    pub m11: C32,
}

impl Mat2 {
    pub fn from_slice(matrix: &[C32]) -> Option<Self> {
        if matrix.len() != 4 {
            return None;
        }
        Some(Self {
            m00: matrix[0],
            m01: matrix[1],
            m10: matrix[2],
            m11: matrix[3],
        })
    }

    /// Matrix multiply: A * B
    pub fn mul(&self, other: &Mat2) -> Mat2 {
        Mat2 {
            m00: self.m00 * other.m00 + self.m01 * other.m10,
            m01: self.m00 * other.m01 + self.m01 * other.m11,
            m10: self.m10 * other.m00 + self.m11 * other.m10,
            m11: self.m10 * other.m01 + self.m11 * other.m11,
        }
    }

    pub fn into_vec(self) -> Vec<C32> {
        vec![self.m00, self.m01, self.m10, self.m11]
    }
}

#[derive(Clone)]
pub struct Operation {
    pub targets: Vec<usize>,
    pub matrix: Vec<C32>,
}

impl Operation {
    pub fn new(targets: Vec<usize>, matrix: Vec<C32>) -> Result<Self, String> {
        if targets.is_empty() {
            return Err("unitary operation must target at least one qubit".to_string());
        }
        ensure_unique_qubits(&targets)?;
        let dim = matrix_dim(targets.len())?;
        if matrix.len() != dim * dim {
            return Err(format!(
                "matrix shape mismatch: expected {} complex values for {} targets, got {}",
                dim * dim,
                targets.len(),
                matrix.len()
            ));
        }
        if !is_unitary(&matrix, dim, 1e-3) {
            return Err("matrix must be unitary within tolerance 1e-3".to_string());
        }
        Ok(Self { targets, matrix })
    }

    pub fn single_qubit(target: usize, matrix: Vec<C32>) -> Result<Self, String> {
        Self::new(vec![target], matrix)
    }

    pub fn is_single_qubit(&self) -> bool {
        self.targets.len() == 1 && self.matrix.len() == 4
    }
}

#[derive(Clone)]
pub enum Gate {
    Operation(Operation),
    Measure(usize),
    Reset(usize),
    Barrier,
}

pub fn make_unitary_gate(targets: Vec<usize>, matrix: Vec<C32>) -> Result<Gate, String> {
    Ok(Gate::Operation(Operation::new(targets, matrix)?))
}

pub fn make_controlled_unitary_gate(
    controls: Vec<usize>,
    targets: Vec<usize>,
    matrix: Vec<C32>,
) -> Result<Gate, String> {
    if controls.is_empty() {
        return Err("controlled unitary requires at least one control qubit".to_string());
    }
    if targets.is_empty() {
        return Err("controlled unitary requires at least one target qubit".to_string());
    }

    let target_dim = matrix_dim(targets.len())?;
    if matrix.len() != target_dim * target_dim {
        return Err(format!(
            "matrix shape mismatch: expected {} complex values for {} target qubits, got {}",
            target_dim * target_dim,
            targets.len(),
            matrix.len()
        ));
    }
    if !is_unitary(&matrix, target_dim, 1e-3) {
        return Err("matrix must be unitary within tolerance 1e-3".to_string());
    }

    let full_matrix = controlled_matrix(controls.len(), targets.len(), &matrix)?;
    let mut full_targets = controls;
    full_targets.extend(targets);
    make_unitary_gate(full_targets, full_matrix)
}

pub fn i_matrix() -> Vec<C32> {
    vec![c32(1.0, 0.0), c32(0.0, 0.0), c32(0.0, 0.0), c32(1.0, 0.0)]
}

pub fn x_matrix() -> Vec<C32> {
    vec![c32(0.0, 0.0), c32(1.0, 0.0), c32(1.0, 0.0), c32(0.0, 0.0)]
}

pub fn y_matrix() -> Vec<C32> {
    vec![c32(0.0, 0.0), c32(0.0, -1.0), c32(0.0, 1.0), c32(0.0, 0.0)]
}

pub fn z_matrix() -> Vec<C32> {
    vec![c32(1.0, 0.0), c32(0.0, 0.0), c32(0.0, 0.0), c32(-1.0, 0.0)]
}

pub fn h_matrix() -> Vec<C32> {
    let v = 1.0f32 / std::f32::consts::SQRT_2;
    vec![c32(v, 0.0), c32(v, 0.0), c32(v, 0.0), c32(-v, 0.0)]
}

pub fn phase_matrix(theta: f32) -> Vec<C32> {
    vec![c32(1.0, 0.0), c32(0.0, 0.0), c32(0.0, 0.0), exp_i(theta)]
}

pub fn rz_matrix(theta: f32) -> Vec<C32> {
    vec![
        exp_i(-theta / 2.0),
        c32(0.0, 0.0),
        c32(0.0, 0.0),
        exp_i(theta / 2.0),
    ]
}

pub fn rx_matrix(theta: f32) -> Vec<C32> {
    let cos = (theta / 2.0).cos();
    let sin = (theta / 2.0).sin();
    vec![
        c32(cos, 0.0),
        c32(0.0, -sin),
        c32(0.0, -sin),
        c32(cos, 0.0),
    ]
}

pub fn ry_matrix(theta: f32) -> Vec<C32> {
    let cos = (theta / 2.0).cos();
    let sin = (theta / 2.0).sin();
    vec![
        c32(cos, 0.0),
        c32(-sin, 0.0),
        c32(sin, 0.0),
        c32(cos, 0.0),
    ]
}

pub fn u_matrix(theta: f32, phi: f32, lambda: f32) -> Vec<C32> {
    let cos = (theta / 2.0).cos();
    let sin = (theta / 2.0).sin();
    let e_phi = exp_i(phi);
    let e_lambda = exp_i(lambda);
    let e_phi_lambda = exp_i(phi + lambda);
    vec![
        c32(cos, 0.0),
        -e_lambda * sin,
        e_phi * sin,
        e_phi_lambda * cos,
    ]
}

pub fn with_global_phase(matrix: &[C32], gamma: f32) -> Vec<C32> {
    let phase = exp_i(gamma);
    matrix.iter().map(|value| *value * phase).collect()
}

pub fn swap_matrix() -> Vec<C32> {
    vec![
        c32(1.0, 0.0),
        c32(0.0, 0.0),
        c32(0.0, 0.0),
        c32(0.0, 0.0),
        c32(0.0, 0.0),
        c32(0.0, 0.0),
        c32(1.0, 0.0),
        c32(0.0, 0.0),
        c32(0.0, 0.0),
        c32(1.0, 0.0),
        c32(0.0, 0.0),
        c32(0.0, 0.0),
        c32(0.0, 0.0),
        c32(0.0, 0.0),
        c32(0.0, 0.0),
        c32(1.0, 0.0),
    ]
}

pub fn iswap_matrix() -> Vec<C32> {
    vec![
        c32(1.0, 0.0),
        c32(0.0, 0.0),
        c32(0.0, 0.0),
        c32(0.0, 0.0),
        c32(0.0, 0.0),
        c32(0.0, 0.0),
        c32(0.0, 1.0),
        c32(0.0, 0.0),
        c32(0.0, 0.0),
        c32(0.0, 1.0),
        c32(0.0, 0.0),
        c32(0.0, 0.0),
        c32(0.0, 0.0),
        c32(0.0, 0.0),
        c32(0.0, 0.0),
        c32(1.0, 0.0),
    ]
}

pub fn controlled_matrix(
    control_count: usize,
    target_qubit_count: usize,
    target_matrix: &[C32],
) -> Result<Vec<C32>, String> {
    if control_count == 0 {
        return Err("controlled operation requires at least one control qubit".to_string());
    }

    let target_dim = matrix_dim(target_qubit_count)?;
    if target_matrix.len() != target_dim * target_dim {
        return Err("controlled operation target matrix has an invalid shape".to_string());
    }

    let dim = matrix_dim(control_count + target_qubit_count)?;
    let mut matrix = identity_matrix(dim);
    let offset = (matrix_dim(control_count)? - 1) * target_dim;

    for row in 0..target_dim {
        for col in 0..target_dim {
            matrix[(offset + row) * dim + offset + col] = target_matrix[row * target_dim + col];
        }
    }

    Ok(matrix)
}

pub fn fuse_gates(gates: &[Gate]) -> Vec<Gate> {
    let mut fused = Vec::with_capacity(gates.len());
    let mut i = 0usize;

    while i < gates.len() {
        match &gates[i] {
            Gate::Operation(op) if op.is_single_qubit() => {
                let target = op.targets[0];
                let mut matrix = Mat2::from_slice(&op.matrix).unwrap();
                i += 1;

                while i < gates.len() {
                    match &gates[i] {
                        Gate::Operation(next) if next.is_single_qubit() && next.targets[0] == target => {
                            let next_matrix = Mat2::from_slice(&next.matrix).unwrap();
                            matrix = next_matrix.mul(&matrix);
                            i += 1;
                        }
                        _ => break,
                    }
                }

                fused.push(Gate::Operation(
                    Operation::single_qubit(target, matrix.into_vec()).unwrap(),
                ));
            }
            gate => {
                fused.push(gate.clone());
                i += 1;
            }
        }
    }

    fused
}

pub struct StateVector {
    pub num_qubits: usize,
    pub amps: Vec<C32>,
}

impl StateVector {
    pub fn new(num_qubits: usize) -> Self {
        let dim = 1usize
            .checked_shl(num_qubits as u32)
            .expect("too many qubits in state vector");
        let mut amps = vec![c32(0.0, 0.0); dim];
        amps[0] = c32(1.0, 0.0);
        Self { num_qubits, amps }
    }

    fn validate_targets(&self, targets: &[usize]) -> Result<(), String> {
        ensure_unique_qubits(targets)?;
        for &target in targets {
            if target >= self.num_qubits {
                return Err(format!(
                    "qubit index {} is out of range for {} qubits",
                    target, self.num_qubits
                ));
            }
        }
        Ok(())
    }

    pub fn apply_mat2(&mut self, target: usize, matrix: &Mat2) {
        let stride = 1usize << target;
        let jump = stride << 1;

        self.amps.par_chunks_mut(jump).for_each(|chunk| {
            for i in 0..stride {
                let a = chunk[i];
                let b = chunk[i + stride];
                chunk[i] = matrix.m00 * a + matrix.m01 * b;
                chunk[i + stride] = matrix.m10 * a + matrix.m11 * b;
            }
        });
    }

    pub fn apply_operation(&mut self, operation: &Operation) -> Result<(), String> {
        self.validate_targets(&operation.targets)?;

        if operation.is_single_qubit() {
            let matrix = Mat2::from_slice(&operation.matrix).unwrap();
            self.apply_mat2(operation.targets[0], &matrix);
            return Ok(());
        }

        let dim = matrix_dim(operation.targets.len())?;
        let old = self.amps.clone();
        let targets = &operation.targets;
        let matrix = &operation.matrix;

        let mut new_amps = vec![c32(0.0, 0.0); old.len()];
        new_amps.par_iter_mut().enumerate().for_each(|(out_index, slot)| {
            let row = extract_sub_index(out_index, targets);
            let base = clear_target_bits(out_index, targets);
            let mut value = c32(0.0, 0.0);

            for col in 0..dim {
                let in_index = compose_index(base, targets, col);
                value += matrix[row * dim + col] * old[in_index];
            }

            *slot = value;
        });

        self.amps = new_amps;
        Ok(())
    }

    pub fn measure_qubit<R: Rng + ?Sized>(
        &mut self,
        qubit: usize,
        rng: &mut R,
    ) -> Result<u8, String> {
        self.validate_targets(&[qubit])?;

        let stride = 1usize << qubit;
        let jump = stride << 1;
        let mut p1 = 0.0f32;

        for chunk in self.amps.chunks(jump) {
            for i in 0..stride {
                p1 += chunk[i + stride].norm_sqr();
            }
        }

        let measured_one = rng.gen::<f32>() < p1;
        let norm = if measured_one {
            p1.sqrt()
        } else {
            (1.0 - p1).sqrt()
        };

        for chunk in self.amps.chunks_mut(jump) {
            for i in 0..stride {
                if measured_one {
                    chunk[i] = c32(0.0, 0.0);
                    chunk[i + stride] = if norm > 0.0 {
                        chunk[i + stride] / norm
                    } else {
                        c32(0.0, 0.0)
                    };
                } else {
                    chunk[i] = if norm > 0.0 { chunk[i] / norm } else { c32(0.0, 0.0) };
                    chunk[i + stride] = c32(0.0, 0.0);
                }
            }
        }

        Ok(u8::from(measured_one))
    }

    pub fn reset_qubit<R: Rng + ?Sized>(
        &mut self,
        qubit: usize,
        rng: &mut R,
    ) -> Result<(), String> {
        let measured = self.measure_qubit(qubit, rng)?;
        if measured == 1 {
            let x = Mat2::from_slice(&x_matrix()).unwrap();
            self.apply_mat2(qubit, &x);
        }
        Ok(())
    }

    pub fn measure(&self, shots: usize) -> Vec<usize> {
        let probabilities: Vec<f64> = self.amps.iter().map(|value| value.norm_sqr() as f64).collect();
        let distribution = WeightedIndex::new(&probabilities).unwrap();
        let mut rng = thread_rng();
        (0..shots).map(|_| distribution.sample(&mut rng)).collect()
    }
}

pub struct Circuit {
    pub gates: Vec<Gate>,
}

impl Circuit {
    pub fn new(raw: Vec<Gate>) -> Self {
        let single_qubit_count = raw
            .iter()
            .filter(|gate| matches!(gate, Gate::Operation(operation) if operation.is_single_qubit()))
            .count();

        let gates = if single_qubit_count >= 16 {
            fuse_gates(&raw)
        } else {
            raw
        };

        Self { gates }
    }

    pub fn run(&self, state_vector: &mut StateVector) -> Result<Vec<(usize, u8)>, String> {
        let mut rng = thread_rng();
        let mut measurements = Vec::new();

        for gate in &self.gates {
            match gate {
                Gate::Operation(operation) => state_vector.apply_operation(operation)?,
                Gate::Measure(qubit) => {
                    let value = state_vector.measure_qubit(*qubit, &mut rng)?;
                    measurements.push((*qubit, value));
                }
                Gate::Reset(qubit) => state_vector.reset_qubit(*qubit, &mut rng)?,
                Gate::Barrier => {}
            }
        }

        Ok(measurements)
    }
}

#[cfg_attr(not(test), allow(dead_code))]
pub fn qft_circuit(num_qubits: usize) -> Circuit {
    let mut gates = Vec::new();

    for i in 0..num_qubits {
        gates.push(make_unitary_gate(vec![i], h_matrix()).unwrap());
        for j in (i + 1)..num_qubits {
            let theta = std::f32::consts::PI / (1usize << (j - i)) as f32;
            gates.push(make_controlled_unitary_gate(
                vec![i],
                vec![j],
                phase_matrix(theta),
            ).unwrap());
        }
    }

    for i in 0..(num_qubits / 2) {
        gates.push(make_unitary_gate(vec![i, num_qubits - 1 - i], swap_matrix()).unwrap());
    }

    Circuit::new(gates)
}

#[cfg_attr(not(test), allow(dead_code))]
pub fn build_circuit<G>(num_qubits: usize, gate: G) -> Circuit
where
    G: Fn(usize) -> Gate + Copy,
{
    let mut gates = Vec::with_capacity(num_qubits);
    for qubit in 0..num_qubits {
        gates.push(gate(qubit));
    }
    Circuit::new(gates)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn assert_state_close(actual: &[C32], expected: &[C32]) {
        assert_eq!(actual.len(), expected.len());
        for (lhs, rhs) in actual.iter().zip(expected.iter()) {
            assert!(
                (*lhs - *rhs).norm() < 1e-4,
                "state mismatch: left={:?}, right={:?}",
                lhs,
                rhs
            );
        }
    }

    #[test]
    fn x_gate_flips_zero_to_one() {
        let circuit = Circuit::new(vec![make_unitary_gate(vec![0], x_matrix()).unwrap()]);
        let mut state = StateVector::new(1);
        circuit.run(&mut state).unwrap();
        assert_state_close(&state.amps, &[c32(0.0, 0.0), c32(1.0, 0.0)]);
    }

    #[test]
    fn cnot_creates_bell_pair() {
        let circuit = Circuit::new(vec![
            make_unitary_gate(vec![0], h_matrix()).unwrap(),
            make_controlled_unitary_gate(vec![0], vec![1], x_matrix()).unwrap(),
        ]);

        let mut state = StateVector::new(2);
        circuit.run(&mut state).unwrap();
        let v = 1.0f32 / std::f32::consts::SQRT_2;
        assert_state_close(
            &state.amps,
            &[c32(v, 0.0), c32(0.0, 0.0), c32(0.0, 0.0), c32(v, 0.0)],
        );
    }

    #[test]
    fn swap_moves_excitation() {
        let circuit = Circuit::new(vec![
            make_unitary_gate(vec![0], x_matrix()).unwrap(),
            make_unitary_gate(vec![0, 1], swap_matrix()).unwrap(),
        ]);

        let mut state = StateVector::new(2);
        circuit.run(&mut state).unwrap();
        assert_state_close(
            &state.amps,
            &[c32(0.0, 0.0), c32(0.0, 0.0), c32(1.0, 0.0), c32(0.0, 0.0)],
        );
    }

    #[test]
    fn toffoli_flips_target() {
        let circuit = Circuit::new(vec![
            make_unitary_gate(vec![0], x_matrix()).unwrap(),
            make_unitary_gate(vec![1], x_matrix()).unwrap(),
            make_controlled_unitary_gate(vec![0, 1], vec![2], x_matrix()).unwrap(),
        ]);

        let mut state = StateVector::new(3);
        circuit.run(&mut state).unwrap();

        let mut expected = vec![c32(0.0, 0.0); 8];
        expected[7] = c32(1.0, 0.0);
        assert_state_close(&state.amps, &expected);
    }

    #[test]
    fn custom_unitary_matches_pauli_y() {
        let circuit = Circuit::new(vec![make_unitary_gate(vec![0], y_matrix()).unwrap()]);
        let mut state = StateVector::new(1);
        circuit.run(&mut state).unwrap();
        assert_state_close(&state.amps, &[c32(0.0, 0.0), c32(0.0, 1.0)]);
    }

    #[test]
    fn reset_returns_qubit_to_zero() {
        let circuit = Circuit::new(vec![
            make_unitary_gate(vec![0], x_matrix()).unwrap(),
            Gate::Reset(0),
        ]);

        let mut state = StateVector::new(1);
        circuit.run(&mut state).unwrap();
        assert_state_close(&state.amps, &[c32(1.0, 0.0), c32(0.0, 0.0)]);
    }

    #[test]
    fn measurement_records_observed_bit() {
        let circuit = Circuit::new(vec![
            make_unitary_gate(vec![0], x_matrix()).unwrap(),
            Gate::Measure(0),
        ]);

        let mut state = StateVector::new(1);
        let measurements = circuit.run(&mut state).unwrap();
        assert_eq!(measurements, vec![(0, 1)]);
        assert_state_close(&state.amps, &[c32(0.0, 0.0), c32(1.0, 0.0)]);
    }

    #[test]
    fn qft_circuit_on_zero_state_is_uniform() {
        let circuit = qft_circuit(2);
        let mut state = StateVector::new(2);
        circuit.run(&mut state).unwrap();

        let expected = vec![
            c32(0.5, 0.0),
            c32(0.5, 0.0),
            c32(0.5, 0.0),
            c32(0.5, 0.0),
        ];
        assert_state_close(&state.amps, &expected);
    }

    #[test]
    fn build_circuit_applies_gate_to_each_qubit() {
        let circuit = build_circuit(3, |qubit| make_unitary_gate(vec![qubit], x_matrix()).unwrap());
        let mut state = StateVector::new(3);
        circuit.run(&mut state).unwrap();

        let mut expected = vec![c32(0.0, 0.0); 8];
        expected[7] = c32(1.0, 0.0);
        assert_state_close(&state.amps, &expected);
    }
}
