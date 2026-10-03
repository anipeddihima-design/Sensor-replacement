import time
import sys
import numpy as np
import qiskit
from qiskit_aer import AerSimulator
from qiskit.circuit import Parameter
from scipy.optimize import minimize
from data_manager import SENSOR_CANDIDATES
from quantum_qaoa import build_sensor_qubo_matrix, qubo_to_ising, evaluate_bitstring_qubo

n_qubits = 24
k_budget = 12

print("Starting 24-Qubit Vectorized QAOA Benchmark...", flush=True)
t0 = time.time()

# 1. Build QUBO Matrix
Q = build_sensor_qubo_matrix(SENSOR_CANDIDATES, k_budget=k_budget)
h, J, offset = qubo_to_ising(Q)

# 2. Fast precomputation of 2^24 diagonal energy vector D
t_pre = time.time()
N = 1 << n_qubits # 16,777,216
# Calculate QUBO cost for all 2^24 states using vectorized bit operations
D = np.zeros(N, dtype=np.float64)
for i in range(n_qubits):
    mask_i = (np.arange(N, dtype=np.uint32) >> i) & 1
    D += Q[i, i] * mask_i
    for j in range(i + 1, n_qubits):
        coupling = Q[i, j] + Q[j, i]
        if abs(coupling) > 1e-8:
            mask_j = (np.arange(N, dtype=np.uint32) >> j) & 1
            D += coupling * (mask_i & mask_j)

print(f"Precomputed 2^24 QUBO Energy Vector D in {time.time() - t_pre:.3f}s", flush=True)

# 3. Build parameterized circuit
gamma_param = Parameter('gamma')
beta_param = Parameter('beta')

qc = qiskit.QuantumCircuit(n_qubits)
qc.h(range(n_qubits))
for i in range(n_qubits):
    if abs(h[i]) > 1e-8:
        qc.rz(2.0 * gamma_param * h[i], i)
for i in range(n_qubits):
    for j in range(i + 1, n_qubits):
        coupling = J[i, j]
        if abs(coupling) > 1e-8:
            qc.cx(i, j)
            qc.rz(2.0 * gamma_param * coupling, j)
            qc.cx(i, j)
for i in range(n_qubits):
    qc.rx(2.0 * beta_param, i)

qc.save_statevector()
sim = AerSimulator(method='statevector')
compiled_qc = qiskit.transpile(qc, sim, optimization_level=0)

# 4. COBYLA Optimization
iter_count = [0]
t_opt_start = time.time()

def objective_fn(params):
    gamma, beta = params[0], params[1]
    bound_circ = compiled_qc.assign_parameters({gamma_param: gamma, beta_param: beta})
    res = sim.run(bound_circ).result()
    sv = res.get_statevector()
    probs = np.abs(sv.data) ** 2
    avg_energy = float(np.dot(probs, D))
    iter_count[0] += 1
    print(f"Iter {iter_count[0]} | Elapsed: {time.time() - t_opt_start:.2f}s | Energy: {avg_energy:.4f}", flush=True)
    return avg_energy

init_params = [0.4, 0.3]
opt_res = minimize(objective_fn, init_params, method='COBYLA', options={'maxiter': 5})

print("="*50, flush=True)
print(f"Optimization Completed in {time.time() - t_opt_start:.3f}s!", flush=True)
print(f"Total Elapsed Time: {time.time() - t0:.3f}s", flush=True)
print(f"Optimal Parameters: gamma={opt_res.x[0]:.4f}, beta={opt_res.x[1]:.4f}", flush=True)
print(f"Optimal Energy: {opt_res.fun:.4f}", flush=True)
