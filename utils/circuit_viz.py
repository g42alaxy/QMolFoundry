""" Quantum circuit visualization."""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")  # headless: required on a Space / in CI
import matplotlib.pyplot as plt
import numpy as np
import pennylane as qml

qml.drawer.use_style("black_white")

_BG = "#f8fafc"


def _vvrq_circuit(weights, n_qubits, n_layers, z):
    """The VVRQ ansatz as a PennyLane qfunc (drawing only — random weights)."""
    for qubit in range(n_qubits):
        qml.RY(z[qubit], wires=[qubit])
    for layer in range(n_layers):
        for j in range(n_qubits):
            qml.Rot(*weights[layer][j], wires=j)
        for j in range(len(z) - 1):
            qml.CNOT(wires=[j, j + 1])
    return qml.probs(wires=list(range(n_qubits)))


def _efq_circuit(weights, n_qubits, n_layers, z):
    """The EFQ ansatz as a PennyLane qfunc — mirrors QuantumExponentialCircut."""
    for layer in range(n_layers):
        for qubit in range(n_qubits):
            qml.RY((2**layer) * z[qubit], wires=[qubit])
        # RotCNOTGate: Rot, CNOT ladder, Rot
        for j in range(n_qubits):
            qml.Rot(*weights[layer][0][j], wires=j)
        for j in range(n_qubits - 1):
            qml.CNOT(wires=[j, j + 1])
        for j in range(n_qubits):
            qml.Rot(*weights[layer][1][j], wires=j)
    for j in range(n_qubits - 1):
        qml.CNOT(wires=[j, j + 1])
    return qml.probs(wires=list(range(n_qubits)))


def circuit_title(arch: str = "VVRQ", n_qubits: int = 8) -> str:
    """The gr.Plot label for the given architecture."""
    if arch == "EFQ":
        return f"Quantum circuit — EFQ · {n_qubits} qubits · depth 2"
    return f"Quantum circuit — VVRQ · {n_qubits} qubits · depth 3"


def draw_vqc(
    arch: str = "VVRQ",
    n_qubits: int = 8,
    n_ancilla: int = 2,
    n_layers: int | None = None,
    title: str | None = None,
):
    """Draw the selected quantum ansatz (VVRQ or EFQ) and return a Figure."""
    rng = np.random.default_rng(42)
    z = rng.standard_normal(n_qubits)

    if arch == "EFQ":
        n_layers = n_layers or 2
        # weights[layer][0|1][qubit] -> 3-vector (two Rot banks per layer)
        weights = [
            [[rng.standard_normal(3) for _ in range(n_qubits)] for _ in range(2)]
            for _ in range(n_layers)
        ]
        qfunc = _efq_circuit
        default_title = (
            f"EFQ (data re-upload) — {n_qubits} qubits ({n_ancilla} ancilla), depth {n_layers}"
        )
    else:
        n_layers = n_layers or 3
        weights = [[rng.standard_normal(3) for _ in range(n_qubits)] for _ in range(n_layers)]
        qfunc = _vvrq_circuit
        default_title = f"VVRQ — {n_qubits} qubits ({n_ancilla} ancilla), depth {n_layers}"

    fig, ax = qml.draw_mpl(qfunc)(weights=weights, n_qubits=n_qubits, n_layers=n_layers, z=z)
    ax.set_title(title or default_title, fontsize=13, color="#1e293b", pad=10)
    fig.patch.set_facecolor(_BG)
    ax.set_facecolor(_BG)
    fig.subplots_adjust(left=0.04, right=0.98, top=0.88, bottom=0.06)
    return fig


def empty_circuit_figure(message: str = "Classical model — no quantum circuit"):
    """Placeholder figure for non-quantum models."""
    _BG = "#f8fafc"
    fig, ax = plt.subplots(figsize=(8, 2))
    fig.patch.set_facecolor(_BG)
    ax.set_facecolor(_BG)
    ax.text(0.5, 0.5, message, ha="center", va="center", fontsize=13, color="#94a3b8")
    ax.axis("off")
    return fig
