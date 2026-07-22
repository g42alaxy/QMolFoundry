from pathlib import Path

import numpy as np
import pytest
import torch
from rdkit import Chem

from models import qm9_meta as meta
from models.base import GenerationResult, compute_metrics
from models.generators import build_registry
from models.molecule_db import insert_molecules, query_molecules
from models.quantum_model import PostProcess, QuantumGenerator


def test_matrices2mol_strips_padding_atoms():
    # nodes: pad, C, pad, O  → ethanol fragment C-O after remap
    nodes = np.array([0, 1, 0, 3])
    edges = np.zeros((4, 4), dtype=int)
    edges[3, 1] = edges[1, 3] = 1  # single bond between C and O

    mol = meta.matrices2mol(nodes, edges, strict=True)

    assert mol is not None
    assert mol.GetNumAtoms() == 2
    smiles = Chem.MolToSmiles(mol)
    assert "*" not in smiles
    assert smiles in {"CO", "OC"}


def test_metrics_keep_raw_draw_counts():
    result = compute_metrics(["CCO", "not-smiles", "", "CCO", "CCN"])

    assert result.sample_count == 5
    assert result.valid_count == 3
    assert result.validity == pytest.approx(3 / 5)
    assert result.uniqueness == pytest.approx(2 / 3)
    assert len(result.smiles) == 3
    assert result.mean_qed == pytest.approx(sum(result.qed) / len(result.qed))


def test_display_sample_preserves_full_draw_means():
    from models.generators import _sample_valid_for_display

    full = compute_metrics(["CCO", "CCN", "CCC", "C", "CC"])
    shown = _sample_valid_for_display(
        GenerationResult(
            smiles=list(full.smiles),
            qed=list(full.qed),
            sa=list(full.sa),
            logp=list(full.logp),
            mean_qed=full.mean_qed,
            mean_sa=full.mean_sa,
            mean_logp=full.mean_logp,
            validity=full.validity,
            uniqueness=full.uniqueness,
            sample_count=full.sample_count,
            valid_count=full.valid_count,
        ),
        n=2,
        rng=np.random.default_rng(0),
    )

    assert len(shown.smiles) == 2
    assert shown.mean_qed == pytest.approx(full.mean_qed)
    assert shown.valid_count == full.valid_count


def test_postprocess_rejects_zero_probability_mass():
    with pytest.raises(ValueError, match="probability mass"):
        PostProcess(torch.zeros(256), n_qubits=8, n_ancillas=2)


def test_quantum_generator_rejects_unknown_mode():
    model = QuantumGenerator([64, 128, 256], 8, 9, 5, 5, 0.0, torch.device("cpu"))

    with pytest.raises(ValueError, match="unsupported quantum generator mode"):
        model(torch.zeros((1, 256)), mode="typo")


def test_all_both_dataset_models_are_usable():
    available = []
    for name, model in build_registry("Both").items():
        assert model.available, name
        result = model.generate(2, mode="curated", seed=7)
        assert result.smiles, (name, result.notes)
        assert model.ckpt_path.exists()
        assert "_both-G.ckpt" in model.ckpt_path.name
        available.append(name)

    assert len(available) == 9


def test_unknown_seed_mode_is_rejected():
    model = build_registry()["MolGAN"]

    with pytest.raises(ValueError, match="unsupported generation mode"):
        model.generate(3, mode="ibm_hardware", seed=0)


def test_database_ordering_is_whitelisted(tmp_path: Path):
    path = tmp_path / "molecules.db"
    rows = [
        {
            "smiles": "CCO",
            "qed": 0.4,
            "sa": 2.0,
            "logp": -0.1,
            "heavy_atoms": 3,
        },
        {
            "smiles": "CCN",
            "qed": 0.7,
            "sa": 2.2,
            "logp": -0.2,
            "heavy_atoms": 3,
        },
    ]
    insert_molecules(
        rows,
        model_slug="test",
        dataset="both",
        backend="ideal",
        shots=0,
        source="test",
        n_generated=2,
        path=path,
    )

    result = query_molecules("test", "both", "ideal", path=path)
    assert [row["smiles"] for row in result] == ["CCN", "CCO"]
    with pytest.raises(ValueError, match="unsupported molecule ordering"):
        query_molecules(
            "test",
            "both",
            "ideal",
            order_by="qed DESC; DROP TABLE molecules",
            path=path,
        )


def test_curated_noisy_backend_smoke():
    model = build_registry()["HQ-MolGAN (VVRQ)"]
    result = model.generate(2, mode="curated", seed=1, backend_key="fake_brisbane")

    assert result.smiles
    assert "Precomputed" in result.notes or "DB:" in result.notes
    assert result.sample_count >= result.valid_count >= 1
    assert np.isfinite(result.validity)
