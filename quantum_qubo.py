"""
Quantum Sensor Placement - QUBO (Quadratic Unconstrained Binary Optimization) Engine
Formulates the 24-sensor placement problem as a QUBO matrix and maps it to an Ising Hamiltonian.
"""

import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from typing import Dict, List, Tuple, Any

from data_manager import SENSOR_CANDIDATES
from quantum_qaoa import (
    haversine_distance,
    build_sensor_qubo_matrix,
    qubo_to_ising,
    evaluate_bitstring_qubo
)


def create_qubo_heatmap_figure(Q: np.ndarray, candidate_names: List[str] = None) -> go.Figure:
    """Generates an interactive Plotly heatmap visualization of the 24x24 QUBO Matrix Q."""
    n = Q.shape[0]
    if candidate_names is None or len(candidate_names) != n:
        candidate_names = [f"Q{i} ({SENSOR_CANDIDATES[i]['name'][:10]})" if i < len(SENSOR_CANDIDATES) else f"Q{i}" for i in range(n)]
        
    fig = px.imshow(
        Q,
        labels=dict(x="Qubit Node j", y="Qubit Node i", color="Q(i,j) Coeff"),
        x=candidate_names,
        y=candidate_names,
        color_continuous_scale="Viridis",
        title=f"24x24 Sensor Placement QUBO Interaction Matrix Q"
    )
    fig.update_layout(
        template="plotly_dark",
        margin=dict(l=20, r=20, t=50, b=20),
        height=550,
        paper_bgcolor="rgba(15,23,42,0.6)",
        plot_bgcolor="rgba(15,23,42,0.6)"
    )
    return fig


def get_qubo_summary(Q: np.ndarray) -> Dict[str, Any]:
    """Returns summary statistics of the QUBO matrix for UI display."""
    diag_terms = np.diag(Q)
    off_diag_mask = ~np.eye(Q.shape[0], dtype=bool)
    off_diag_terms = Q[off_diag_mask]
    
    return {
        "num_qubits": Q.shape[0],
        "matrix_shape": f"{Q.shape[0]}x{Q.shape[1]}",
        "min_linear_coeff": float(np.min(diag_terms)),
        "max_linear_coeff": float(np.max(diag_terms)),
        "mean_linear_coeff": float(np.mean(diag_terms)),
        "min_quadratic_interaction": float(np.min(off_diag_terms)),
        "max_quadratic_interaction": float(np.max(off_diag_terms)),
        "sparsity_pct": float(np.sum(Q == 0) / Q.size * 100)
    }
