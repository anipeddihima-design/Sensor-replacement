"""
Quantum Sensor Placement - Benchmark Comparison Module
Provides classical vs 24-qubit QAOA quantum benchmark comparison and interactive Plotly visualization components.
"""

import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
from typing import Dict, List, Any

from quantum_utils import (
    run_classical_baseline_sensor_placement,
    compute_quantum_placement_metrics,
    compare_classical_vs_quantum
)


def create_benchmark_comparison_chart(comparison_data: Dict[str, Any]) -> go.Figure:
    """Generates a side-by-side bar chart comparing Classical vs Quantum performance metrics."""
    classical = comparison_data["classical"]
    quantum_m = comparison_data["quantum_metrics"]
    
    categories = ["Coverage (%)", "Efficiency (%)", "Redundancy (%)", "Dead Zones"]
    classical_vals = [
        classical["coverage_pct"],
        classical["efficiency"] * 100,
        classical["redundancy_pct"],
        classical["dead_zones"]
    ]
    quantum_vals = [
        quantum_m["coverage_pct"],
        quantum_m["efficiency"] * 100,
        quantum_m["redundancy_pct"],
        quantum_m["dead_zones"]
    ]
    
    fig = go.Figure(data=[
        go.Bar(name='Classical Baseline (Greedy/Sequential)', x=categories, y=classical_vals, marker_color='#EF4444'),
        go.Bar(name='24-Qubit QAOA Quantum Engine', x=categories, y=quantum_vals, marker_color='#10B981')
    ])
    
    fig.update_layout(
        barmode='group',
        title="Side-by-Side Performance Comparison: Classical vs 24-Qubit Quantum",
        template="plotly_dark",
        height=450,
        paper_bgcolor="rgba(15,23,42,0.6)",
        plot_bgcolor="rgba(15,23,42,0.6)",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )
    return fig


def create_energy_convergence_chart(energy_history: List[float]) -> go.Figure:
    """Generates QAOA ground energy convergence curve across optimization iterations."""
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=list(range(1, len(energy_history) + 1)),
        y=energy_history,
        mode='lines+markers',
        name='QAOA Energy Expectation <H_C>',
        line=dict(color='#F59E0B', width=3),
        marker=dict(size=6, color='#F59E0B')
    ))
    
    fig.update_layout(
        title="QAOA COBYLA Optimizer Energy Convergence (<H_C> vs Iteration)",
        xaxis_title="Optimization Step / Iteration",
        yaxis_title="Ground Energy Expectation <H_C>",
        template="plotly_dark",
        height=400,
        paper_bgcolor="rgba(15,23,42,0.6)",
        plot_bgcolor="rgba(15,23,42,0.6)"
    )
    return fig
