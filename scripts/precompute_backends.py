#!/usr/bin/env python3
"""Precompute PQC probability vectors for all available checkpoints on every

Output: models/data/probs_{slug}_{dataset}_{backend}.npy  — shape (N, 256)

Run:
    uv run python scripts/precompute_backends.py            # all combos
    uv run python scripts/precompute_backends.py --dry-run  # estimate only
    uv run python scripts/precompute_backends.py --model vvrq_none --dataset qm9
    uv run python scripts/precompute_backends.py --backend fake_brisbane

Timing (M2 Pro): ~1.7 s/circuit @ 200k shots per fake-device backend.
Full run (18 checkpoints × 3 backends × 1k seeds):  ~25 min total (parallelisable).
Use --n-seeds 100 for a quick smoke-test (~15 s total).
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(_ROOT))

from models.backends import (
    BACKEND_KEYS,
    BACKEND_LABELS,
    N_QUBITS,
    PAPER_SHOTS,
    get_backend,
)
from models.generators import _DATA_DIR, _DSKEY, MODEL_SPECS, MolGANGenerator

_FAKE_BACKENDS = [k for k in BACKEND_KEYS if k != "ideal"]
_N_OUTPUT = 2**N_QUBITS  # 256

# ── helpers ───────────────────────────────────────────────────────────────────


def _all_combos() -> list[tuple[str, str, str]]:
    """All (slug, dataset_key, gen_kind) for quantum checkpoints that exist."""
    combos = []
    for display_name, (gen_kind, cycle_kind, is_quantum) in MODEL_SPECS.items():
        if not is_quantum:
            continue
        slug = f"{gen_kind}_{cycle_kind}"
        for ds_label, ds_key in _DSKEY.items():
            gen = MolGANGenerator(ds_label, display_name)
            if gen.ckpt_path.exists():
                combos.append((slug, ds_key, ds_label, display_name, gen_kind))
    # De-duplicate by (slug, ds_key) — multiple display_names share the same PQC
    seen: set[tuple[str, str]] = set()
    unique = []
    for item in combos:
        key = (item[0], item[1])
        if key not in seen:
            seen.add(key)
            unique.append(item)
    return unique


def _out_path(slug: str, ds_key: str, backend_key: str) -> Path:
    return _DATA_DIR / f"probs_{slug}_{ds_key}_{backend_key}.npy"


def _pqc_weights(display_name: str, ds_label: str) -> tuple[np.ndarray, str]:
    """Load the trained PQC weights from the checkpoint."""
    import torch

    from models.molgan_nets import Generator
    from models.quantum_model import QuantumExponentialGenerator, QuantumGenerator

    gen_kind = MODEL_SPECS[display_name][0]
    gen = MolGANGenerator(ds_label, display_name)
    V, B, A, Z, CONV = 9, 5, 5, 8, [64, 128, 256]
    dev = torch.device("cpu")

    if gen_kind == "classic":
        model = Generator(CONV, Z, V, B, A, 0.0)
    elif gen_kind == "vvrq":
        model = QuantumGenerator(CONV, Z, V, B, A, 0.0, dev)
    else:
        model = QuantumExponentialGenerator(CONV, Z, V, B, A, 0.0, dev)

    model.load_state_dict(torch.load(str(gen.ckpt_path), map_location="cpu"))
    model.eval()
    weights = model.quantum_params.detach().cpu().numpy()
    return weights, gen_kind


def _run_backend(
    backend_key: str, gen_kind: str, weights: np.ndarray, z: np.ndarray, shots: int
) -> np.ndarray:
    """Run z through the PQC on backend_key → probability vectors (N, 256)."""
    backend = get_backend(backend_key, ansatz=gen_kind)  # type: ignore[arg-type]
    probs = []
    n = len(z)
    for i, vec in enumerate(z):
        p = backend.run_pqc(vec, weights, shots=shots)
        probs.append(p)
        if (i + 1) % 10 == 0 or (i + 1) == n:
            print(f"    {i + 1}/{n} circuits done", end="\r", flush=True)
    print()
    return np.stack(probs, axis=0)


def _print_plan(combos, backends, n_seeds, shots, dry_run: bool):
    missing, existing = [], []
    for slug, ds_key, *_ in combos:
        for bkey in backends:
            path = _out_path(slug, ds_key, bkey)
            (existing if path.exists() else missing).append((slug, ds_key, bkey))

    # Timing estimate: ~1.7 s/circuit @ 200k shots for fake-device
    s_per_circuit = 1.7 * (shots / 200_000)
    total_s = len(missing) * n_seeds * s_per_circuit

    print("\n=== Precompute plan ===")
    print(f"  Combos to compute : {len(missing)}")
    print(f"  Already done      : {len(existing)}")
    print(f"  Seeds per combo   : {n_seeds}")
    print(f"  Shots             : {shots:,}")
    print(f"  Est. time         : {total_s / 60:.1f} min ({total_s:.0f} s)")

    if missing:
        print("\n  Missing:")
        for slug, ds_key, bkey in missing:
            blabel = BACKEND_LABELS.get(bkey, bkey)
            print(f"    {slug} / {ds_key} / {blabel}")

    if dry_run:
        print("\n[dry-run] Nothing written.\n")
    return missing


# ── main ─────────────────────────────────────────────────────────────────────


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--dry-run", action="store_true", help="Print plan and exit without computing."
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
        "--n-seeds",
        type=int,
        default=1000,
        help="Number of random z vectors to precompute (default 1000).",
    )
    parser.add_argument(
        "--shots",
        type=int,
        default=PAPER_SHOTS,
        help=f"Shots per PQC circuit (default {PAPER_SHOTS}).",
    )
    parser.add_argument(
        "--overwrite", action="store_true", help="Recompute even if output file already exists."
    )
    args = parser.parse_args()

    combos = _all_combos()
    if args.model:
        combos = [c for c in combos if c[0] == args.model]
    if args.dataset:
        combos = [c for c in combos if c[1] == args.dataset]

    backends = _FAKE_BACKENDS
    if args.backend:
        backends = [b for b in backends if b == args.backend]

    if not combos:
        print("No matching checkpoints found.")
        return
    if not backends:
        print("No matching backends.")
        return

    _print_plan(combos, backends, args.n_seeds, args.shots, args.dry_run)
    if args.dry_run:
        return

    if not args.overwrite:
        to_run = [
            (s, d, dl, dn, gk, bk)
            for (s, d, dl, dn, gk) in combos
            for bk in backends
            if not _out_path(s, d, bk).exists()
        ]
    else:
        to_run = [(s, d, dl, dn, gk, bk) for (s, d, dl, dn, gk) in combos for bk in backends]

    if not to_run:
        print("\nAll files already exist. Use --overwrite to recompute.\n")
        return

    # Reuse z vectors from noise_vector_list.npy when available (VVRQ/QM9 compat).
    z_pool_path = _DATA_DIR / "noise_vector_list.npy"
    if z_pool_path.exists():
        z_pool = np.load(str(z_pool_path)).astype(np.float32)
        if len(z_pool) >= args.n_seeds:
            z_base = z_pool[: args.n_seeds]
        else:
            rng = np.random.default_rng(0)
            extra = rng.standard_normal((args.n_seeds - len(z_pool), N_QUBITS)).astype(np.float32)
            z_base = np.concatenate([z_pool, extra], axis=0)
    else:
        rng = np.random.default_rng(0)
        z_base = rng.standard_normal((args.n_seeds, N_QUBITS)).astype(np.float32)

    print(f"\nUsing {len(z_base)} seed vectors.\n")

    _DATA_DIR.mkdir(parents=True, exist_ok=True)

    # Group by (slug, ds_key, gen_kind) so we load each checkpoint only once.

    to_run.sort(key=lambda x: (x[0], x[1]))

    weights_cache: dict[tuple[str, str], tuple[np.ndarray, str]] = {}

    total_start = time.time()
    done = 0
    total = len(to_run)

    for slug, ds_key, ds_label, display_name, gen_kind, backend_key in to_run:
        blabel = BACKEND_LABELS.get(backend_key, backend_key)
        out = _out_path(slug, ds_key, backend_key)
        print(f"[{done + 1}/{total}] {slug} / {ds_key} / {blabel}")

        ckpt_key = (slug, ds_key)
        if ckpt_key not in weights_cache:
            print(f"  Loading checkpoint for {display_name} / {ds_label}...")
            weights, gk = _pqc_weights(display_name, ds_label)
            weights_cache[ckpt_key] = (weights, gk)
        weights, gk = weights_cache[ckpt_key]

        t0 = time.time()
        probs = _run_backend(backend_key, gk, weights, z_base, args.shots)
        elapsed = time.time() - t0

        np.save(str(out), probs)
        print(f"  Saved {probs.shape} → {out.name}  ({elapsed:.1f}s)")
        done += 1

    total_elapsed = time.time() - total_start
    print(f"\nDone. {done}/{total} files written in {total_elapsed / 60:.1f} min.\n")


if __name__ == "__main__":
    main()
