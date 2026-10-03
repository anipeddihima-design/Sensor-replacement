import time
import sys
from quantum_qaoa import run_qaoa_local_simulation

def cb(i, max_i, elapsed, energy):
    print(f"Progress: Step {i}/{max_i} | Elapsed: {elapsed}s | Energy: {energy}", flush=True)

t0 = time.time()
res = run_qaoa_local_simulation(
    k_budget=12,
    p_layers=1,
    shots=128,
    maxiter=3,
    sim_method='matrix_product_state',
    progress_callback=cb
)
t_elapsed = time.time() - t0

print('='*50, flush=True)
print('BENCHMARK QUICK TEST RESULT:', flush=True)
print('Total Execution Time:', round(t_elapsed, 3), 'seconds', flush=True)
print('Simulator Engine:', res['backend'], flush=True)
print('Qubits Count:', res['n_qubits'], 'qubits', flush=True)
print('Sensors Selected:', len(res['selected_sensors']), 'nodes', flush=True)
print('Optimal QUBO Energy:', res['optimal_qubo_energy'], flush=True)
print('Best Bitstring:', res['best_bitstring'], flush=True)
print('SUCCESS!', flush=True)
