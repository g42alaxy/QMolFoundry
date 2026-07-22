"""Generation result and molecular metric helpers."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np

LATENT_DIM = 8


@dataclass
class GenerationResult:
    """Everything the UI needs to render one generation request."""

    smiles: list[str]
    qed: list[float] = field(default_factory=list)
    sa: list[float] = field(default_factory=list)
    logp: list[float] = field(default_factory=list)
    # Means over *all* valid draws (kept when smiles/qed are subsampled for display).
    mean_qed: float = float("nan")
    mean_sa: float = float("nan")
    mean_logp: float = float("nan")
    validity: float = 0.0
    uniqueness: float = 0.0
    sample_count: int = 0
    valid_count: int = 0
    z_used: Optional[np.ndarray] = None
    notes: str = ""


class MoleculeGenerator:
    """Minimal shared state for generators exposed by the UI."""

    def __init__(self, dataset: str = "QM9"):
        self.dataset = dataset
        self._loaded = False
        self.pqc_weights: Optional[np.ndarray] = None


def _nanmean(values: list[float]) -> float:
    finite = [v for v in values if v == v]
    return sum(finite) / len(finite) if finite else float("nan")


def compute_metrics(smiles: list[str]) -> GenerationResult:
    """Compute RDKit metrics for a list of SMILES.

    Lives here (not in a subclass) so both the app and CI can call it. Uses
    real RDKit; invalid SMILES contribute to the validity denominator but not
    to per-molecule metric lists.
    """
    import os
    import sys

    from rdkit import Chem
    from rdkit.Chem import QED, Crippen, RDConfig

    # SA scorer ships as a contrib script in RDKit.
    sys.path.append(os.path.join(RDConfig.RDContribDir, "SA_Score"))
    try:
        import sascorer  # type: ignore

        have_sa = True
    except Exception:
        have_sa = False

    qed_vals, sa_vals, logp_vals, valid_smiles = [], [], [], []
    n_total = len(smiles)

    for smi in smiles:
        mol = Chem.MolFromSmiles(smi) if smi else None
        if mol is None:
            continue
        valid_smiles.append(Chem.MolToSmiles(mol))
        try:
            qed_vals.append(float(QED.qed(mol)))
        except Exception:
            qed_vals.append(float("nan"))
        logp_vals.append(float(Crippen.MolLogP(mol)))
        if have_sa:
            try:
                sa_vals.append(float(sascorer.calculateScore(mol)))
            except Exception:
                sa_vals.append(float("nan"))

    n_valid = len(valid_smiles)
    n_unique = len(set(valid_smiles))
    validity = n_valid / n_total if n_total else 0.0
    uniqueness = n_unique / n_valid if n_valid else 0.0

    return GenerationResult(
        smiles=valid_smiles,
        qed=qed_vals,
        sa=sa_vals,
        logp=logp_vals,
        mean_qed=_nanmean(qed_vals),
        mean_sa=_nanmean(sa_vals),
        mean_logp=_nanmean(logp_vals),
        validity=validity,
        uniqueness=uniqueness,
        sample_count=n_total,
        valid_count=n_valid,
    )
