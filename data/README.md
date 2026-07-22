# Model data — Both (QM9+PC9) MolGAN matrix

The live app uses the joint **Both** training set only. Each cell of the 3×3
generator×cycle matrix is backed by a trained G checkpoint
(`../models/weights/<slug>_both-G.ckpt`) and, when available, a frozen curated
bank (`bank_<slug>_both.npz`).

`<slug>` = `<gen>_<cycle>` where gen ∈ {classic, vvrq, efq}, cycle ∈ {none,
classical, quantum}. Decoder dims match QM9/PC9 (9 nodes; atoms C,N,O,F; 5 bond
types — see `../models/qm9_meta.py`).

## Active coverage (Both)

|                     | Both |
|---------------------|------|
| Classic · None      | ✓    |
| Classic · Classical | ✓    |
| Classic · Quantum   | ✓    |
| VVRQ · None         | ✓    |
| VVRQ · Classical    | ✓    |
| VVRQ · Quantum      | ✓    |
| EFQ · None          | ✓    |
| EFQ · Classical     | ✓    |
| EFQ · Quantum       | ✓    |

Single-dataset checkpoints (QM9 / PC9) are archived under
`../models/weights/deprecated/` and are not used by the app.

## Curated banks

Training runs mode-collapse late. Banks freeze QED-ranked valid largest
fragments with fields `smiles`, `qed`, `sa`, `logp`, `heavy`, `src_iter`,
`src_seed`, and `src_temp`. Classic·None·Both may rely on the live checkpoint
when its bank file is absent.

## Noisy-backend precomputes

Files `probs_<slug>_both_<backend>.npy` store offline 200k-shot PQC probability
vectors for curated mode on fake IBM devices. `molecules.db` caches decoded
valid molecules for instant serving.

## Deprecated archive (`deprecated/`)

Everything not tied to an available **Both** checkpoint lives under `deprecated/`
and is never loaded by the live app:

- Single-dataset banks/probs — `bank_<slug>_{qm9,pc9}.npz`,
  `probs_<slug>_{qm9,pc9}_<backend>.npy` — no QM9/PC9 checkpoints ship with the app.
- Legacy `forward_pass_*.npy` / `noise_vector_list.npy` files, kept for bank
  rebuild provenance (`../models/build_curated_bank.py`) and not exposed in the UI.
