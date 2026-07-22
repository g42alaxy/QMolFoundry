"""Build a curated bank for HQ-MolGAN (VVRQ)."""

from __future__ import annotations

import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from rdkit import Chem, RDLogger
from rdkit.Chem import QED, Crippen, RDConfig

RDLogger.DisableLog("rdApp.*")
sys.path.append(os.path.join(RDConfig.RDContribDir, "SA_Score"))
import sascorer  # noqa: E402

from models import qm9_meta as meta  # noqa: E402
from models.quantum_model import QuantumGenerator  # noqa: E402

_DATA = Path(__file__).parent.parent / "data"
_WEIGHTS = Path(__file__).parent / "weights" / "vvrq_none_qm9-G.ckpt"


def _build_model():
    device = torch.device("cpu")
    m = QuantumGenerator(
        [64, 128, 256],
        8,
        meta.VERTEXES,
        meta.BOND_NUM_TYPES,
        meta.ATOM_NUM_TYPES,
        0.0,
        device,
    ).to(device)
    m.load_state_dict(torch.load(str(_WEIGHTS), map_location="cpu"))
    m.eval()
    return m


def _largest_fragment(smiles: str, min_heavy: int):
    """Return (canonical_smiles, mol) of the largest connected fragment.

    Zero-shot outputs are mostly multi-fragment (e.g. 'C.C.C=C(CC)NCC'); the
    chemically meaningful molecule is the largest connected component. Trivial
    fragments below `min_heavy` heavy atoms are rejected.
    """
    frags = [f for f in smiles.split(".") if f]
    mols = [Chem.MolFromSmiles(f) for f in frags]
    mols = [m for m in mols if m is not None]
    if not mols:
        return None
    best = max(mols, key=lambda m: m.GetNumHeavyAtoms())
    if best.GetNumHeavyAtoms() < min_heavy:
        return None
    return Chem.MolToSmiles(best), best


def _decode_batch(el, nl, seed, min_heavy):
    """Notebook-exact hard-Gumbel decode of a full logits batch.

    Returns the largest connected fragment of each valid output.
    """
    torch.manual_seed(seed)
    e_flat = F.gumbel_softmax(el.contiguous().view(-1, el.size(-1)), hard=True)
    n_flat = F.gumbel_softmax(nl.contiguous().view(-1, nl.size(-1)), hard=True)
    eh = torch.max(e_flat.view(el.size()), -1)[1]
    nh = torch.max(n_flat.view(nl.size()), -1)[1]
    out = []
    for idx, (e_, n_) in enumerate(zip(eh, nh)):
        mol = meta.matrices2mol(n_.numpy(), e_.numpy(), strict=True)
        if mol is None:
            continue
        try:
            s = Chem.MolToSmiles(mol)
        except Exception:
            continue
        if not s:
            continue
        lf = _largest_fragment(s, min_heavy)
        if lf is not None:
            out.append((lf[0], idx, lf[1]))
    return out


def build(n_seeds: int = 64, top_k: int = 64, min_heavy: int = 4) -> Path:
    model = _build_model()
    probs = np.load(str(_DATA / "deprecated" / "forward_pass_ideal.npy"))
    x = torch.tensor(probs, dtype=torch.float32)
    with torch.no_grad():
        el, nl = model(x, "IBM")

    bank: dict[str, dict] = {}
    t0 = time.time()
    for seed in range(n_seeds):
        for s, idx, mol in _decode_batch(el, nl, seed, min_heavy):
            if s in bank:
                continue
            try:
                bank[s] = dict(
                    qed=float(QED.qed(mol)),
                    sa=float(sascorer.calculateScore(mol)),
                    logp=float(Crippen.MolLogP(mol)),
                    heavy=int(mol.GetNumHeavyAtoms()),
                    seed=int(seed),
                    idx=int(idx),
                )
            except Exception:
                continue
        if (seed + 1) % 10 == 0:
            print(
                f"  seed {seed + 1}/{n_seeds}: {len(bank)} distinct so far "
                f"({time.time() - t0:.0f}s)"
            )

    # Rank by QED, keep top_k
    ranked = sorted(bank.items(), key=lambda kv: kv[1]["qed"], reverse=True)[:top_k]
    smiles = np.array([s for s, _ in ranked])
    qed = np.array([d["qed"] for _, d in ranked], dtype=np.float64)
    sa = np.array([d["sa"] for _, d in ranked], dtype=np.float64)
    logp = np.array([d["logp"] for _, d in ranked], dtype=np.float64)
    heavy = np.array([d["heavy"] for _, d in ranked], dtype=np.int64)
    seed_arr = np.array([d["seed"] for _, d in ranked], dtype=np.int64)
    idx_arr = np.array([d["idx"] for _, d in ranked], dtype=np.int64)

    out = _DATA / "deprecated" / "bank_vvrq_none_qm9.npz"
    np.savez(
        out,
        smiles=smiles,
        qed=qed,
        sa=sa,
        logp=logp,
        heavy=heavy,
        batch_seed=seed_arr,
        batch_idx=idx_arr,
    )
    print(
        f"\nWrote {out} — {len(smiles)} molecules (from {len(bank)} distinct over {n_seeds} seeds)"
    )
    print(f"QED range {qed.min():.3f}..{qed.max():.3f}, mean {qed.mean():.3f}")
    return out


if __name__ == "__main__":
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 64
    build(n_seeds=n)
