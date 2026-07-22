"""
Shared graph→SMILES decoding for all MolGAN generator variants.

Every generator (classic / VVRQ / EFQ) emits the same (edges_logits, nodes_logits)
pair; this module turns that into RDKit SMILES using the QM9/PC9/Both decoders
(identical across all three datasets — see qm9_meta). Kept in one place so live
generation and the offline curated-bank builder decode identically.
"""

from __future__ import annotations

from . import qm9_meta as meta


def _hard_gumbel(logits):
    import torch.nn.functional as F

    flat = logits.contiguous().view(-1, logits.size(-1))
    return F.gumbel_softmax(flat, hard=True).view(logits.size())


def logits_to_smiles(
    edges_logits,
    nodes_logits,
    seed: int | None = None,
    largest_fragment: bool = False,
    min_heavy: int = 1,
):
    """Decode a batch of (edges_logits, nodes_logits) to a list of SMILES.

    Args:
        edges_logits: (n, V, V, bond_types) tensor.
        nodes_logits: (n, V, atom_types) tensor.
        seed:         if set, fixes the torch RNG so the Gumbel sample is
                      reproducible.
        largest_fragment: keep only the largest connected fragment of each mol
                      (zero-shot output is often multi-fragment).
        min_heavy:    reject mols/fragments below this many heavy atoms ("" ).

    Returns:
        list[str] of canonical SMILES; "" marks an invalid/rejected draw so the
        caller can measure honest validity.
    """
    import torch
    from rdkit import Chem

    if seed is not None:
        torch.manual_seed(int(seed))

    edges_hard = torch.max(_hard_gumbel(edges_logits), -1)[1]
    nodes_hard = torch.max(_hard_gumbel(nodes_logits), -1)[1]

    out: list[str] = []
    for e_, n_ in zip(edges_hard, nodes_hard):
        mol = meta.matrices2mol(n_.cpu().numpy(), e_.cpu().numpy(), strict=True)
        if mol is None:
            out.append("")
            continue
        try:
            s = Chem.MolToSmiles(mol)
        except Exception:
            out.append("")
            continue
        if not s or "*" in s:  # reject dummy/padding (atomic-number-0) atoms
            out.append("")
            continue
        if largest_fragment:
            frags = [f for f in s.split(".") if f]
            mols = [Chem.MolFromSmiles(f) for f in frags]
            mols = [m for m in mols if m is not None]
            if not mols:
                out.append("")
                continue
            best = max(mols, key=lambda m: m.GetNumHeavyAtoms())
            if best.GetNumHeavyAtoms() < min_heavy:
                out.append("")
                continue
            s = Chem.MolToSmiles(best)
        elif min_heavy > 1:
            m = Chem.MolFromSmiles(s)
            if m is None or m.GetNumHeavyAtoms() < min_heavy:
                out.append("")
                continue
        out.append(s)
    return out
