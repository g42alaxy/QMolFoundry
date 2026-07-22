import numpy as np
import pytest
from qiskit.quantum_info import Statevector

from models.backends import (
    IdealBackend,
    _build_qiskit_efq,
    _build_qiskit_vvrq,
    _counts_to_probs,
)


@pytest.mark.parametrize(
    ("ansatz", "builder", "weight_shape"),
    [
        ("vvrq", _build_qiskit_vvrq, (3, 8, 3)),
        ("efq", _build_qiskit_efq, (2, 2, 8, 3)),
    ],
)
def test_qiskit_circuit_matches_pennylane(ansatz, builder, weight_shape):
    rng = np.random.default_rng(2026)
    z = rng.normal(size=8)
    weights = rng.normal(size=weight_shape)

    expected = IdealBackend(ansatz).run_pqc(z, weights)
    circuit = builder(z, weights)
    circuit.remove_final_measurements(inplace=True)
    qiskit_order = Statevector.from_instruction(circuit).probabilities()
    penny_order = np.array([qiskit_order[int(f"{index:08b}"[::-1], 2)] for index in range(2**8)])

    np.testing.assert_allclose(penny_order, expected, atol=1e-12)


def test_counts_are_converted_to_pennylane_wire_order():
    probs = _counts_to_probs({"01": 3, "10": 1}, n_qubits=2)

    np.testing.assert_allclose(probs, [0.0, 0.25, 0.75, 0.0])
