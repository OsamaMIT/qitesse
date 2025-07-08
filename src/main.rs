use num_complex::Complex;
use rand::distributions::WeightedIndex;
use rand::prelude::*;
use rayon::prelude::*;
use std::time::Instant;

#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::*;

/// 2×2 complex matrix for single‐qubit fusion
#[derive(Copy, Clone)]
pub struct Mat2 {
    pub m00: Complex<f32>,
    pub m01: Complex<f32>,
    pub m10: Complex<f32>,
    pub m11: Complex<f32>,
}

impl Mat2 {
    /// Matrix multiply: A·B
    pub fn mul(&self, other: &Mat2) -> Mat2 {
        Mat2 {
            m00: self.m00 * other.m00 + self.m01 * other.m10,
            m01: self.m00 * other.m01 + self.m01 * other.m11,
            m10: self.m10 * other.m00 + self.m11 * other.m10,
            m11: self.m10 * other.m01 + self.m11 * other.m11,
        }
    }

    /// Convert a primitive Gate into its Mat2 (only X, H, RZ supported)
    pub fn from_gate(g: &Gate) -> Option<Mat2> {
        match *g {
            Gate::X(_) => Some(Mat2 {
                m00: Complex::new(0.0, 0.0),
                m01: Complex::new(1.0, 0.0),
                m10: Complex::new(1.0, 0.0),
                m11: Complex::new(0.0, 0.0),
            }),
            Gate::H(_) => {
                let v = 1.0f32 / std::f32::consts::SQRT_2;
                Some(Mat2 {
                    m00: Complex::new(v, 0.0),
                    m01: Complex::new(v, 0.0),
                    m10: Complex::new(v, 0.0),
                    m11: Complex::new(-v, 0.0),
                })
            }
            Gate::RZ(_, theta) => {
                let ph = Complex::from_polar(1.0f32, theta);
                Some(Mat2 {
                    m00: Complex::new(1.0, 0.0),
                    m01: Complex::new(0.0, 0.0),
                    m10: Complex::new(0.0, 0.0),
                    m11: ph,
                })
            }
            _ => None,
        }
    }
}

/// Extended gate enum
#[derive(Clone)]
pub enum Gate {
    X(usize),
    H(usize),
    RZ(usize, f32),
    CRZ(usize, usize, f32),
    CNOT(usize, usize),
    /// A fused single‐qubit unitary
    Unitary(usize, Mat2),
}

/// Fuse any adjacent single‐qubit gates on the same wire into one `Unitary`
pub fn fuse_gates(gates: &[Gate]) -> Vec<Gate> {
    let mut out = Vec::with_capacity(gates.len());
    let mut i = 0;
    while i < gates.len() {
        if let Some(q) = match gates[i] {
            Gate::X(q) | Gate::H(q) | Gate::RZ(q, _) => Some(q),
            _ => None,
        } {
            // start fusion
            let mut mat = Mat2::from_gate(&gates[i]).unwrap();
            i += 1;
            while i < gates.len() {
                if let Some(q2) = match gates[i] {
                    Gate::X(q2) | Gate::H(q2) | Gate::RZ(q2, _) => Some(q2),
                    _ => None,
                } {
                    if q2 == q {
                        let nxt = Mat2::from_gate(&gates[i]).unwrap();
                        mat = nxt.mul(&mat);
                        i += 1;
                        continue;
                    }
                }
                break;
            }
            out.push(Gate::Unitary(q, mat));
        } else {
            out.push(gates[i].clone());
            i += 1;
        }
    }
    out
}

/// State vector for n qubits: 2^n amplitudes, Complex<f32>
#[repr(align(32))]
pub struct StateVector {
    pub num_qubits: usize,
    pub amps: Vec<Complex<f32>>,
}

impl StateVector {
    pub fn new(num_qubits: usize) -> Self {
        let dim = 1 << num_qubits;
        let mut amps = vec![Complex::new(0.0, 0.0); dim];
        amps[0] = Complex::new(1.0, 0.0);
        StateVector { num_qubits, amps }
    }

    /// SIMD+Rayon apply of any 2×2 matrix
    pub fn apply_mat2(&mut self, target: usize, m: &Mat2) {
        let stride = 1 << target;
        let jump   = stride << 1;
        let (m00, m01, m10, m11) = (m.m00, m.m01, m.m10, m.m11);
        self.amps.par_chunks_mut(jump).for_each(|chunk| unsafe {
            for i in 0..stride {
                let a = chunk[i];
                let b = chunk[i + stride];
                chunk[i]          = m00 * a + m01 * b;
                chunk[i + stride] = m10 * a + m11 * b;
            }
        });
    }

    /// Controlled‑RZ and CNOT unchanged
    pub fn apply_crz(&mut self, control: usize, target: usize, theta: f32) {
        let ctrl_stride = 1 << control;
        let tgt_stride  = 1 << target;
        let block       = 1 << (control + 1);
        let (pr, pi)    = (theta.cos(), theta.sin());
        self.amps.par_chunks_mut(block).for_each(|chunk| unsafe {
            let ptr = chunk.as_mut_ptr();
            for offset in 0..ctrl_stride {
                let mut idx = ctrl_stride + offset;
                while idx < block {
                    if idx & tgt_stride != 0 {
                        let c = &mut *ptr.add(idx);
                        let (re, im) = (c.re * pr - c.im * pi, c.re * pi + c.im * pr);
                        c.re = re; c.im = im;
                    }
                    idx += tgt_stride << 1;
                }
            }
        });
    }
    pub fn apply_cnot(&mut self, control: usize, target: usize) {
        let stride   = 1 << target;
        let jump     = stride << 1;
        let ctrl_mask = 1 << control;
        self.amps.par_chunks_mut(jump).enumerate().for_each(|(ci,chunk)| {
            if (ci * jump) & ctrl_mask != 0 {
                for i in 0..stride { chunk.swap(i, i + stride); }
            }
        });
    }

    /// Born‑rule measurement
    pub fn measure(&self, shots: usize) -> Vec<usize> {
        let probs: Vec<f64> = self.amps.iter().map(|c| c.norm_sqr() as f64).collect();
        let dist = WeightedIndex::new(&probs).unwrap();
        let mut rng = thread_rng();
        (0..shots).map(|_| dist.sample(&mut rng)).collect()
    }
}

/// Circuit runner with adaptive fusion
pub struct Circuit {
    pub gates: Vec<Gate>,
}

impl Circuit {
    pub fn new(raw: Vec<Gate>) -> Self {
        // count single-qubit gates
        let count_sq: usize = raw.iter().filter(|g| matches!(g, Gate::X(_) | Gate::H(_) | Gate::RZ(_, _))).count();
        // threshold: only fuse if enough single-qubit gates
        let fused = if count_sq >= 16 { // uses gate fusion only if more than 16 gates are used
            fuse_gates(&raw)
        } else {
            raw
        };
        Circuit { gates: fused }
    }

    pub fn run(&self, sv: &mut StateVector) {
        for gate in &self.gates {
            match *gate {
                Gate::Unitary(t, mat) => sv.apply_mat2(t, &mat),
                Gate::X(t)            => sv.apply_mat2(t, &Mat2::from_gate(&Gate::X(t)).unwrap()),
                Gate::H(t)            => sv.apply_mat2(t, &Mat2::from_gate(&Gate::H(t)).unwrap()),
                Gate::RZ(t,th)        => sv.apply_mat2(t, &Mat2::from_gate(&Gate::RZ(t,th)).unwrap()),
                Gate::CRZ(c,t,th)     => sv.apply_crz(c, t, th),
                Gate::CNOT(c,t)       => sv.apply_cnot(c, t),
            }
        }
    }
}

/// Build QFT with primitive gates (fusion checked in `Circuit::new`)
pub fn qft_circuit(n: usize) -> Circuit {
    let mut gates = Vec::new();

    for i in 0..n {
        gates.push(Gate::H(i));
        for j in (i+1)..n {
            // Note: control is i, target is j for CRZ in QFT
            let theta = std::f32::consts::PI / (1 << (j - i)) as f32;
            gates.push(Gate::CRZ(i, j, theta));
        }
    }

    // Bit reversal with swaps (via 3 CNOTs)
    for i in 0..(n/2) {
        gates.push(Gate::CNOT(i, n - 1 - i));
        gates.push(Gate::CNOT(n - 1 - i, i));
        gates.push(Gate::CNOT(i, n - 1 - i));
    }

    Circuit::new(gates)
}

/// Build a circuit with n number of `gate`
pub fn build_circuit<G>(n: usize, gate: G) -> Circuit
where
    G: Fn(usize) -> Gate + Copy,
{
    let mut gates = Vec::new();
    for i in 0..n {
        gates.push(gate(0));
    }
    Circuit::new(gates)
}

fn main() {
    rayon::ThreadPoolBuilder::new().num_threads(1).build_global().unwrap(); // set to 1 thread for benchmarking
    println!("Using {} threads", rayon::current_num_threads());

    let n = 30; // change this to the desired number of qubits
    let mut sv = StateVector::new(n);

    let circuit = qft_circuit(n); // QFT circuit for n qubits (for algorithm benchmarking)
    // let circuit = build_circuit(1, |i| Gate::X(i)); // given frequency of a given gate on n qubits (for gate benchmarking)

    let start = Instant::now();
    circuit.run(&mut sv);
    println!("Circuit with {} qubits took {:?}", n, start.elapsed());
}

// QFT-29 single threaded took 166.0865099s 