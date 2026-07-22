from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Literal, Optional

import numpy as np

N_QUBITS = 8
N_LAYERS = 3  # VVRQ depth (alias)
N_LAYERS_VVRQ = 3
N_LAYERS_EFQ = 2
N_ANCILLAS = 2
DEFAULT_SHOTS = 8192  # live default; demo can offer 1024..200000
PAPER_SHOTS = 200_000  # the setting used in the paper's device runs

Ansatz = Literal["vvrq", "efq"]


@dataclass(frozen=True)
class BackendInfo:
    key: str  # internal id, e.g. "fake_brisbane"
    label: str  # UI label, e.g. "IBM Brisbane (noisy sim)"
    is_quantum_hw_model: bool  # True for the fake-device sims
    device_qubits: Optional[int] = None
    ansatz: str = "vvrq"


class QuantumBackend(ABC):
    info: BackendInfo

    @abstractmethod
    def run_pqc(self, z: np.ndarray, weights: np.ndarray, shots: int = DEFAULT_SHOTS) -> np.ndarray:
        # z: (N_QUBITS,) encoding angles. weights: VVRQ (N_LAYERS, N_QUBITS, 3)
        # or EFQ (N_LAYERS, 2, N_QUBITS, 3). Returns a (2**N_QUBITS,) prob vector.
        raise NotImplementedError


class IdealBackend(QuantumBackend):
    """Noiseless PennyLane statevector."""

    def __init__(self, ansatz: Ansatz = "vvrq", n_qubits: int = N_QUBITS):
        import pennylane as qml

        from .quantum_model import QuantumCircut, QuantumExponentialCircut

        self.ansatz = ansatz
        self.n_qubits = n_qubits
        self.n_layers = N_LAYERS_VVRQ if ansatz == "vvrq" else N_LAYERS_EFQ
        self._dev = qml.device("default.qubit", wires=n_qubits)
        fn = QuantumCircut if ansatz == "vvrq" else QuantumExponentialCircut

        @qml.qnode(self._dev, interface=None)
        def circuit(z, weights):
            return fn(weights, n_qubits, self.n_layers, z)

        self._circuit = circuit
        self.info = BackendInfo(
            "ideal",
            "Ideal simulator (noiseless)",
            False,
            ansatz=ansatz,
        )

    def run_pqc(self, z, weights, shots=DEFAULT_SHOTS):
        return np.asarray(self._circuit(z, weights), dtype=np.float64)


def _apply_rot(qc, q, params):
    # Qiskit gate sequence equivalent to PennyLane's Rot.
    a, b, c = (float(x) for x in params)
    qc.rz(a, q)
    qc.ry(b, q)
    qc.rz(c, q)


def _build_qiskit_vvrq(z, weights, n_qubits=N_QUBITS, n_layers=N_LAYERS_VVRQ):
    # Rebuild the VVRQ ansatz in Qiskit from PennyLane-trained weights.
    from qiskit import QuantumCircuit

    qc = QuantumCircuit(n_qubits)
    for q in range(n_qubits):
        qc.ry(float(z[q]), q)
    for layer in range(n_layers):
        for q in range(n_qubits):
            _apply_rot(qc, q, weights[layer][q])
        for q in range(n_qubits - 1):
            qc.cx(q, q + 1)
    qc.measure_all()
    return qc


def _build_qiskit_efq(z, weights, n_qubits=N_QUBITS, n_layers=N_LAYERS_EFQ):
    # Rebuild the EFQ ansatz in Qiskit from PennyLane-trained weights.
    from qiskit import QuantumCircuit

    qc = QuantumCircuit(n_qubits)
    for layer in range(n_layers):
        for q in range(n_qubits):
            qc.ry(float(2**layer * z[q]), q)
        for q in range(n_qubits):
            _apply_rot(qc, q, weights[layer][0][q])
        for q in range(n_qubits - 1):
            qc.cx(q, q + 1)
        for q in range(n_qubits):
            _apply_rot(qc, q, weights[layer][1][q])
    for q in range(n_qubits - 1):
        qc.cx(q, q + 1)
    qc.measure_all()
    return qc


def _counts_to_probs(counts: dict, n_qubits: int = N_QUBITS) -> np.ndarray:
    # Qiskit orders classical bits most-significant first; PennyLane indexes
    # wire 0 first. Reversing the bitstring aligns noisy and ideal backends.
    probs = np.zeros(2**n_qubits, dtype=np.float64)
    total = sum(counts.values())
    for bitstring, c in counts.items():
        bits = bitstring.replace(" ", "")
        idx = int(bits[::-1], 2)
        probs[idx] += c
    if total:
        probs /= total
    return probs


class FakeDeviceBackend(QuantumBackend):
    """Noisy simulation of a named 127-qubit IBM Eagle device."""

    _DEVICE_CLASSES = {
        "fake_brisbane": ("FakeBrisbane", "IBM Brisbane (noisy sim)"),
        "fake_sherbrooke": ("FakeSherbrooke", "IBM Sherbrooke (noisy sim)"),
        "fake_osaka": ("FakeOsaka", "IBM Osaka (noisy sim)"),
    }

    def __init__(
        self,
        key: str,
        ansatz: Ansatz = "vvrq",
        n_qubits: int = N_QUBITS,
        optimization_level: int = 1,
    ):
        if key not in self._DEVICE_CLASSES:
            raise ValueError(f"unknown device backend: {key}")
        from qiskit_aer import AerSimulator
        from qiskit_ibm_runtime import fake_provider as fp

        cls_name, label = self._DEVICE_CLASSES[key]
        fake = getattr(fp, cls_name)()
        self._fake = fake
        self._sim = AerSimulator.from_backend(fake)
        self._opt = optimization_level
        self.ansatz = ansatz
        self.n_qubits = n_qubits
        self.n_layers = N_LAYERS_VVRQ if ansatz == "vvrq" else N_LAYERS_EFQ
        self._build = _build_qiskit_vvrq if ansatz == "vvrq" else _build_qiskit_efq
        self.info = BackendInfo(
            key,
            label,
            True,
            device_qubits=fake.num_qubits,
            ansatz=ansatz,
        )

    def run_pqc(self, z, weights, shots=DEFAULT_SHOTS):
        from qiskit import transpile

        qc = self._build(z, weights, self.n_qubits, self.n_layers)
        tqc = transpile(qc, self._sim, optimization_level=self._opt)
        result = self._sim.run(tqc, shots=shots).result()
        counts = result.get_counts()
        return _counts_to_probs(counts, self.n_qubits)


# Registry
BACKEND_KEYS = ["ideal", "fake_brisbane", "fake_sherbrooke", "fake_osaka"]

BACKEND_LABELS = {
    "ideal": "Ideal simulator (noiseless)",
    "fake_brisbane": "IBM Brisbane (noisy sim)",
    "fake_sherbrooke": "IBM Sherbrooke (noisy sim)",
    "fake_osaka": "IBM Osaka (noisy sim)",
}

# Reverse: UI label -> key
LABEL_TO_KEY = {v: k for k, v in BACKEND_LABELS.items()}


def build_backend(key: str, ansatz: Ansatz = "vvrq") -> QuantumBackend:
    if key == "ideal":
        return IdealBackend(ansatz=ansatz)
    return FakeDeviceBackend(key, ansatz=ansatz)


# Cache instances — building a FakeDevice sim is non-trivial (loads noise model).
_BACKEND_CACHE: dict[str, QuantumBackend] = {}


def get_backend(key: str, ansatz: Ansatz = "vvrq") -> QuantumBackend:
    cache_key = f"{key}:{ansatz}"
    if cache_key not in _BACKEND_CACHE:
        _BACKEND_CACHE[cache_key] = build_backend(key, ansatz=ansatz)
    return _BACKEND_CACHE[cache_key]
