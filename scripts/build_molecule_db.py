#!/usr/bin/env python3
"""Build (or extend) the SQLite molecule database from precomputed data.

Run:
    uv run python scripts/build_molecule_db.py            # full build
    uv run python scripts/build_molecule_db.py --dry-run  # stats only
    uv run python scripts/build_molecule_db.py --source banks
    uv run python scripts/build_molecule_db.py --source probs
    uv run python scripts/build_molecule_db.py --model vvrq_none --dataset qm9
    uv run python scripts/build_molecule_db.py --backend fake_brisbane
    uv run python scripts/build_molecule_db.py --stats     # print DB stats
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(_ROOT))

from models.backends import BACKEND_LABELS, PAPER_SHOTS
from models.base import compute_metrics
from models.decode import logits_to_smiles
from models.generators import (
    _DATA_DIR,
    _DSKEY,
    MODEL_SPECS,
    MolGANGenerator,
)
from models.molecule_db import combo_stats, has_combo, insert_molecules, run_history

_FAKE_BACKENDS = [k for k in BACKEND_LABELS if k != "ideal"]


# ── helpers ───────────────────────────────────────────────────────────────────


def _all_quantum_combos() -> list[tuple[str, str, str, str]]:
    """Return (slug, ds_key, ds_label, display_name) for available quantum ckpts."""
    seen: set[tuple[str, str]] = set()
    combos = []
    for display_name, (gen_kind, cycle_kind, is_quantum) in MODEL_SPECS.items():
        if not is_quantum:
            continue
        slug = f"{gen_kind}_{cycle_kind}"
        for ds_label, ds_key in _DSKEY.items():
            gen = MolGANGenerator(ds_label, display_name)
            if gen.ckpt_path.exists() and (slug, ds_key) not in seen:
                seen.add((slug, ds_key))
                combos.append((slug, ds_key, ds_label, display_name))
    return combos


def _all_combos_with_banks() -> list[tuple[str, str, str, str]]:
    """Return quantum + classical combos that have a bank file."""
    seen: set[tuple[str, str]] = set()
    combos = []
    for display_name, (gen_kind, cycle_kind, is_quantum) in MODEL_SPECS.items():
        slug = f"{gen_kind}_{cycle_kind}"
        for ds_label, ds_key in _DSKEY.items():
            gen = MolGANGenerator(ds_label, display_name)
            if gen.bank_path.exists() and (slug, ds_key) not in seen:
                seen.add((slug, ds_key))
                combos.append((slug, ds_key, ds_label, display_name))
    return combos


# ── import banks ──────────────────────────────────────────────────────────────


def import_banks(filter_slug=None, filter_ds=None, dry_run=False) -> int:
    combos = _all_combos_with_banks()
    if filter_slug:
        combos = [c for c in combos if c[0] == filter_slug]
    if filter_ds:
        combos = [c for c in combos if c[1] == filter_ds]

    total_inserted = 0
    for slug, ds_key, ds_label, display_name in combos:
        gen = MolGANGenerator(ds_label, display_name)
        d = np.load(str(gen.bank_path), allow_pickle=True)
        smiles_list = [str(s) for s in d["smiles"]]
        qed_list = [float(v) for v in d["qed"]]
        sa_list = [float(v) for v in d["sa"]]
        logp_list = [float(v) for v in d["logp"]]

        from rdkit import Chem

        rows = []
        for smi, qed, sa, logp in zip(smiles_list, qed_list, sa_list, logp_list):
            mol = Chem.MolFromSmiles(smi) if smi else None
            if mol is None:
                continue
            rows.append(
                {
                    "smiles": smi,
                    "qed": qed,
                    "sa": sa,
                    "logp": logp,
                    "heavy_atoms": mol.GetNumHeavyAtoms(),
                }
            )

        print(f"  Bank  {slug}/{ds_key}: {len(rows)} valid molecules", end="")
        if dry_run:
            print(" [dry-run]")
            continue

        n = insert_molecules(
            rows,
            model_slug=slug,
            dataset=ds_key,
            backend="ideal",
            shots=0,
            source="bank",
            n_generated=len(smiles_list),
            notes=f"imported from {gen.bank_path.name}",
        )
        total_inserted += n
        print(f" → {n} new rows")

    return total_inserted


# ── decode probs ──────────────────────────────────────────────────────────────


def decode_probs_file(
    gen: "MolGANGenerator", probs_path: Path, seed: int = 0
) -> tuple[list[dict], int]:
    """Decode all prob vectors in a .npy file; return (valid_rows, n_total)."""
    probs = np.load(str(probs_path))  # (N, 256)
    n_total = len(probs)

    if not gen._loaded:
        gen.load()

    # _postprocess_probs + _classical_head: works for both VVRQ and EFQ
    features = gen._postprocess_probs(probs)
    el, nl = gen._classical_head(features)
    raw = logits_to_smiles(el, nl, seed=seed, largest_fragment=True, min_heavy=2)
    result = compute_metrics(raw)

    from rdkit import Chem

    rows = []
    for smi, qed, sa, logp in zip(result.smiles, result.qed, result.sa, result.logp):
        mol = Chem.MolFromSmiles(smi) if smi else None
        if mol is None:
            continue
        rows.append(
            {
                "smiles": smi,
                "qed": qed,
                "sa": sa if result.sa else None,
                "logp": logp,
                "heavy_atoms": mol.GetNumHeavyAtoms(),
            }
        )

    return rows, n_total


def import_probs(
    filter_slug=None, filter_ds=None, filter_backend=None, skip_existing=True, dry_run=False
) -> int:
    combos = _all_quantum_combos()
    if filter_slug:
        combos = [c for c in combos if c[0] == filter_slug]
    if filter_ds:
        combos = [c for c in combos if c[1] == filter_ds]

    backends = _FAKE_BACKENDS
    if filter_backend:
        backends = [b for b in backends if b == filter_backend]

    total_inserted = 0
    for slug, ds_key, ds_label, display_name in combos:
        gen = MolGANGenerator(ds_label, display_name)
        for backend_key in backends:
            probs_path = _DATA_DIR / f"probs_{slug}_{ds_key}_{backend_key}.npy"
            if not probs_path.exists():
                continue

            blabel = BACKEND_LABELS.get(backend_key, backend_key)
            if skip_existing and has_combo(slug, ds_key, backend_key, min_rows=1):
                print(f"  Skip  {slug}/{ds_key}/{backend_key}: already in DB")
                continue

            print(
                f"  Probs {slug}/{ds_key}/{blabel}: decoding {probs_path.name}…", end="", flush=True
            )
            if dry_run:
                probs = np.load(str(probs_path))
                print(f" {len(probs)} vectors [dry-run]")
                continue

            t0 = time.time()
            rows, n_total = decode_probs_file(gen, probs_path)
            elapsed = time.time() - t0
            validity = len(rows) / n_total if n_total else 0.0

            print(
                f" {len(rows)}/{n_total} valid ({validity:.1%}, {elapsed:.1f}s)", end="", flush=True
            )

            n = insert_molecules(
                rows,
                model_slug=slug,
                dataset=ds_key,
                backend=backend_key,
                shots=PAPER_SHOTS,
                source="precomputed_probs",
                n_generated=n_total,
                notes=f"decoded from {probs_path.name}",
            )
            total_inserted += n
            print(f" → {n} new rows")

    return total_inserted


# ── stats ─────────────────────────────────────────────────────────────────────


def print_stats():
    stats = combo_stats()
    if not stats:
        print("  DB is empty or does not exist.")
        return

    print(f"\n{'Model':<22} {'DS':<6} {'Backend':<20} {'N':>6} {'QED':>6} {'SA':>5}")
    print("-" * 70)
    for r in stats:
        print(
            f"  {r['model_slug']:<20} {r['dataset']:<6} {r['backend']:<20} "
            f"{r['n_mols']:>6} {r['mean_qed'] or 0:>6.3f} {r['mean_sa'] or 0:>5.2f}"
        )

    history = run_history(limit=10)
    if history:
        print(f"\nLast {len(history)} build runs:")
        for h in history:
            print(
                f"  {h['ran_at'][:19]}  {h['model_slug']}/{h['dataset']}/{h['backend']}"
                f"  valid={h['n_valid']}/{h['n_generated']}"
                f"  inserted={h['n_inserted']}"
                f"  src={h['source']}"
            )


# ── main ─────────────────────────────────────────────────────────────────────


def main():
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--dry-run", action="store_true", help="Print plan without writing to the DB."
    )
    parser.add_argument("--stats", action="store_true", help="Print DB statistics and exit.")
    parser.add_argument(
        "--source",
        choices=["banks", "probs", "both"],
        default="both",
        help="Which data source to process (default: both).",
    )
    parser.add_argument(
        "--model", metavar="SLUG", help="Filter to a specific model slug, e.g. vvrq_none."
    )
    parser.add_argument(
        "--dataset", metavar="KEY", help="Filter to a dataset key: qm9 | pc9 | both."
    )
    parser.add_argument(
        "--backend", metavar="KEY", help="Filter to one backend, e.g. fake_brisbane."
    )
    parser.add_argument(
        "--overwrite", action="store_true", help="Re-import even if the combo is already in the DB."
    )
    args = parser.parse_args()

    if args.stats:
        print_stats()
        return

    total = 0

    if args.source in ("banks", "both"):
        print("\n=== Importing curated banks (ideal backend) ===")
        n = import_banks(
            filter_slug=args.model,
            filter_ds=args.dataset,
            dry_run=args.dry_run,
        )
        total += n
        if not args.dry_run:
            print(f"  Banks: {n} new rows total")

    if args.source in ("probs", "both"):
        print("\n=== Decoding precomputed probs (fake backends) ===")
        n = import_probs(
            filter_slug=args.model,
            filter_ds=args.dataset,
            filter_backend=args.backend,
            skip_existing=not args.overwrite,
            dry_run=args.dry_run,
        )
        total += n
        if not args.dry_run:
            print(f"  Probs: {n} new rows total")

    if not args.dry_run:
        print(f"\nTotal new rows inserted: {total}")
        print_stats()
    else:
        print("\n[dry-run] Nothing written.\n")


if __name__ == "__main__":
    main()
