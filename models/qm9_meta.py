"""
QM9 9-node dataset metadata — hardcoded decoders matching the MolGAN training setup.

This avoids loading the full .sparsedataset pickle while preserving exactly the
atom/bond type mapping used when the 46000-G.ckpt weights were trained.
"""

from __future__ import annotations

import numpy as np
from rdkit import Chem

VERTEXES = 9
ATOM_NUM_TYPES = 5  # pad + C, N, O, F
BOND_NUM_TYPES = 5  # no-bond, SINGLE, DOUBLE, TRIPLE, AROMATIC

# index → atomic number  (0 = padding atom)
ATOM_DECODER: dict[int, int] = {0: 0, 1: 6, 2: 7, 3: 8, 4: 9}

# index → RDKit BondType
BOND_DECODER: dict[int, Chem.rdchem.BondType] = {
    0: Chem.rdchem.BondType.ZERO,
    1: Chem.rdchem.BondType.SINGLE,
    2: Chem.rdchem.BondType.DOUBLE,
    3: Chem.rdchem.BondType.TRIPLE,
    4: Chem.rdchem.BondType.AROMATIC,
}


def matrices2mol(
    node_labels: np.ndarray, edge_labels: np.ndarray, strict: bool = False
) -> Chem.RWMol | None:
    """Convert (node_labels, edge_labels) matrices to an RDKit molecule.

    Padding atoms (decoder index 0 → atomic number 0) are omitted and bonds
    remapped onto the remaining atoms. Including pads as RDKit dummy atoms
    either fails ``SanitizeMol`` or yields ``*`` SMILES that the decoder
    rejects — which previously produced false 0/1000 validity on several
    Both checkpoints.
    """
    mol = Chem.RWMol()
    mapping: dict[int, int] = {}
    for i, nl in enumerate(node_labels):
        atomic_num = ATOM_DECODER[int(nl)]
        if atomic_num == 0:
            continue
        mapping[i] = mol.AddAtom(Chem.Atom(atomic_num))

    if not mapping:
        return None

    for start, end in zip(*np.nonzero(edge_labels)):
        if start <= end:
            continue
        if start not in mapping or end not in mapping:
            continue
        bond_type = int(edge_labels[start, end])
        if bond_type == 0:
            continue
        a, b = mapping[start], mapping[end]
        if a == b or mol.GetBondBetweenAtoms(a, b) is not None:
            continue
        mol.AddBond(a, b, BOND_DECODER[bond_type])

    if strict:
        try:
            Chem.SanitizeMol(mol)
        except Exception:
            return None
    return mol
