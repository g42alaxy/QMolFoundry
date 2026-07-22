from __future__ import annotations

from pathlib import Path

import numpy as np

from .backends import BACKEND_KEYS, BACKEND_LABELS, get_backend
from .base import GenerationResult, MoleculeGenerator, _nanmean, compute_metrics
from .decode import logits_to_smiles

_HERE = Path(__file__).parent
_WEIGHTS = _HERE / "weights"
_DATA_DIR = _HERE.parent / "data"

V, B, A, Z, CONV = 9, 5, 5, 8, [64, 128, 256]  # QM9/PC9/Both share these dims
_DSKEY = {"QM9": "qm9", "PC9": "pc9", "Both": "both"}

# display_name -> (gen_kind, cycle_kind, is_quantum)
MODEL_SPECS: dict[str, tuple[str, str, bool]] = {
    "MolGAN": ("classic", "none", False),
    "HQ-MolGAN (VVRQ)": ("vvrq", "none", True),
    "HQ-MolGAN (EFQ)": ("efq", "none", True),
    "Cycle MolGAN": ("classic", "classical", False),
    "HQ Cycle MolGAN (VVRQ)": ("vvrq", "classical", True),
    "HQ Cycle MolGAN (EFQ)": ("efq", "classical", True),
    "Hybrid Cycle MolGAN": ("classic", "quantum", False),
    "Hybrid-Cycle HQ-MolGAN (VVRQ)": ("vvrq", "quantum", True),
    "Hybrid-Cycle HQ-MolGAN (EFQ)": ("efq", "quantum", True),
}
MODEL_NAMES = list(MODEL_SPECS.keys())


def _is_fake_backend(backend_key: str) -> bool:
    return backend_key.startswith("fake_")


def _sample_valid_for_display(
    res: GenerationResult, n: int, rng: np.random.Generator
) -> GenerationResult:
    """Keep full-draw means; show up to n randomly chosen valid molecules."""
    n_valid = len(res.smiles)
    if n_valid <= n:
        return res
    idx = rng.choice(n_valid, size=n, replace=False)
    res.smiles = [res.smiles[i] for i in idx]
    # Per-molecule lists are for grid legends only; mean_* stay over the full draw.
    res.qed = [res.qed[i] for i in idx]
    res.sa = [res.sa[i] for i in idx] if res.sa else []
    res.logp = [res.logp[i] for i in idx]
    return res


def _with_means(res: GenerationResult) -> GenerationResult:
    """Fill mean_* from the current per-molecule lists (curated / DB paths)."""
    res.mean_qed = _nanmean(res.qed)
    res.mean_sa = _nanmean(res.sa)
    res.mean_logp = _nanmean(res.logp)
    return res


class MolGANGenerator(MoleculeGenerator):
    """One cell of the matrix for one dataset, backed by a G checkpoint."""

    def __init__(self, dataset: str, display_name: str):
        super().__init__(dataset=dataset)
        self.display_name = display_name
        self.gen_kind, self.cycle_kind, self.is_quantum = MODEL_SPECS[display_name]
        self.slug = f"{self.gen_kind}_{self.cycle_kind}"
        self._dskey = _DSKEY[dataset]
        self.ckpt_path = _WEIGHTS / f"{self.slug}_{self._dskey}-G.ckpt"
        self.bank_path = _DATA_DIR / f"bank_{self.slug}_{self._dskey}.npz"
        self._model = None

    @property
    def available(self) -> bool:
        return self.bank_path.exists() or self.ckpt_path.exists()

    def load(self) -> None:
        import torch

        from .molgan_nets import Generator
        from .quantum_model import QuantumExponentialGenerator, QuantumGenerator

        dev = torch.device("cpu")
        if self.gen_kind == "classic":
            self._model = Generator(CONV, Z, V, B, A, 0.0)
        elif self.gen_kind == "vvrq":
            self._model = QuantumGenerator(CONV, Z, V, B, A, 0.0, dev)
        else:
            self._model = QuantumExponentialGenerator(CONV, Z, V, B, A, 0.0, dev)
        state_dict = torch.load(str(self.ckpt_path), map_location="cpu", weights_only=True)
        self._model.load_state_dict(state_dict)
        self._model.eval()
        if self.is_quantum:
            self.pqc_weights = self._model.quantum_params.detach().cpu().numpy()
        self._loaded = True

    def _forward_logits_ideal(self, z):
        """z (n, Z) tensor -> (edges_logits, nodes_logits) via PennyLane."""
        import torch

        x = torch.as_tensor(z, dtype=torch.float32)
        with torch.no_grad():
            if self.gen_kind == "vvrq":
                return self._model(x, "Default")
            return self._model(x)

    def _postprocess_probs(self, raw_probs: np.ndarray) -> np.ndarray:
        # Ancilla truncation matching QuantumGenerator.PostProcess.
        import torch

        from .quantum_model import PostProcess

        n_q = self._model.n_qubits
        n_a = self._model.n_ancillas
        rows = []
        for p in raw_probs:
            t = torch.as_tensor(p, dtype=torch.float32)
            rows.append(PostProcess(t, n_q, n_a).numpy())
        return np.stack(rows, axis=0)

    def _classical_head(self, features: np.ndarray):
        # Run the trained classical decoder head on PQC features.
        import torch

        x = torch.as_tensor(features, dtype=torch.float32)
        with torch.no_grad():
            out = self._model.layers(x)
            el = self._model.edges_layer(out).view(
                -1,
                self._model.edges,
                self._model.vertexes,
                self._model.vertexes,
            )
            el = (el + el.permute(0, 1, 3, 2)) / 2
            el = self._model.dropoout(el.permute(0, 2, 3, 1))
            nl = self._model.nodes_layer(out)
            nl = self._model.dropoout(nl.view(-1, self._model.vertexes, self._model.nodes))
        return el, nl

    def _forward_logits_backend(self, z, backend_key: str, shots: int):
        # Run the PQC on a fake/noisy backend, then the classical decoder head.
        backend = get_backend(backend_key, ansatz=self.gen_kind)  # type: ignore[arg-type]
        raw = np.stack([backend.run_pqc(vec, self.pqc_weights, shots=shots) for vec in z])
        features = self._postprocess_probs(raw)
        return self._classical_head(features)

    def _forward_logits(self, z, backend_key: str = "ideal", shots: int = 8192):
        # Dispatch to PennyLane (ideal) or Qiskit Aer fake-device backends.
        if self.is_quantum and _is_fake_backend(backend_key):
            return self._forward_logits_backend(z, backend_key, shots)
        return self._forward_logits_ideal(z)

    def _noisy_probs_path(self, backend_key: str) -> Path:
        # data/probs_{slug}_{dataset}_{backend}.npy, shape (N, 256).
        return _DATA_DIR / f"probs_{self.slug}_{self._dskey}_{backend_key}.npy"

    def _serve_precomputed(self, n: int, backend_key: str, seed: int) -> "GenerationResult | None":
        # Try the SQLite DB first (pre-decoded, QED-ranked), then fall back to
        # decoding a raw .npy probs file. Returns None if nothing valid is found.
        from .molecule_db import has_combo, query_molecules

        blabel = BACKEND_LABELS.get(backend_key, backend_key)

        if has_combo(self.slug, self._dskey, backend_key, min_rows=1):
            # Shuffle deterministically so repeated calls with different seeds
            # surface different molecules from the pool.
            import random as _random

            rng_py = _random.Random(seed)
            rows = query_molecules(
                self.slug,
                self._dskey,
                backend_key,
                n=min(n * 4, 200),  # fetch a wider pool …
                order_by="RANDOM()",  # … in random order
            )
            if not rows:
                return None
            rng_py.shuffle(rows)
            rows = rows[:n]
            res = GenerationResult(
                smiles=[r["smiles"] for r in rows],
                qed=[r["qed"] for r in rows if r.get("qed") is not None],
                sa=[r["sa"] for r in rows if r.get("sa") is not None],
                logp=[r["logp"] for r in rows if r.get("logp") is not None],
                validity=1.0,
                uniqueness=len({r["smiles"] for r in rows}) / len(rows),
                sample_count=len(rows),
                valid_count=len(rows),
            )
            _with_means(res)
            res.notes = (
                f"DB: {len(rows)} pre-validated molecules from {blabel} "
                f"(200k shots, offline) — instant, QED-ranked.  •  "
                f"{self.display_name} · {self.dataset}."
            )
            return res

        # Raw .npy fallback.
        path = self._noisy_probs_path(backend_key)
        if not path.exists():
            return None
        if not self._loaded:
            self.load()

        probs = np.load(str(path))  # (N, 256)
        rng = np.random.default_rng(seed)
        probs_sub = probs[rng.permutation(len(probs))]

        # works for both VVRQ and EFQ (bypasses model.forward IBM mode)
        features = self._postprocess_probs(probs_sub)
        el, nl = self._classical_head(features)

        raw = logits_to_smiles(el, nl, seed=seed, largest_fragment=True, min_heavy=2)
        res = compute_metrics(raw)
        if not res.smiles:
            return None
        res.smiles = res.smiles[:n]
        res.qed, res.sa, res.logp = res.qed[:n], res.sa[:n], res.logp[:n]
        res.notes = (
            f"Precomputed: {len(probs)} probability vectors from {blabel} "
            f"(200k shots, offline) — instant, reproducible.  •  "
            f"{self.display_name} · {self.dataset}."
        )
        return res

    def _serve_bank(self, n: int) -> GenerationResult:
        d = np.load(str(self.bank_path), allow_pickle=True)
        smiles = [str(s) for s in d["smiles"][:n]]
        res = GenerationResult(
            smiles=smiles,
            qed=[float(v) for v in d["qed"][:n]],
            sa=[float(v) for v in d["sa"][:n]],
            logp=[float(v) for v in d["logp"][:n]],
            validity=1.0,
            uniqueness=len(set(smiles)) / len(smiles) if smiles else 0.0,
            sample_count=len(smiles),
            valid_count=len(smiles),
        )
        _with_means(res)
        res.notes = (
            f"Curated bank: top {len(smiles)} of {len(d['smiles'])} verified "
            f"molecules (QED-ranked, largest connected fragment), frozen for "
            f"reproducibility.  •  {self.display_name} · {self.dataset}."
        )
        return res

    def _live_generation(
        self,
        n: int,
        seed: int,
        backend_key: str,
        shots: int,
        curated_fallback: bool = False,
        n_draw: int | None = None,
    ) -> GenerationResult:
        if not self._loaded:
            self.load()
        rng = np.random.default_rng(seed)
        if n_draw is not None:
            n_sample = n_draw
        elif self.is_quantum and _is_fake_backend(backend_key):
            n_sample = max(n * 2, 16)
        else:
            n_sample = max(n * 6, 96)
        z = rng.normal(0, 1, size=(n_sample, Z)).astype(np.float32)
        el, nl = self._forward_logits(z, backend_key=backend_key, shots=shots)
        raw = logits_to_smiles(el, nl, seed=seed, largest_fragment=True, min_heavy=2)
        res = compute_metrics(raw)
        res.z_used = z
        # Validity/uniqueness stay over the full draw; display a random subset.
        res = _sample_valid_for_display(res, n, rng)

        if self.is_quantum and _is_fake_backend(backend_key):
            blabel = BACKEND_LABELS.get(backend_key, backend_key)
            engine = f"live PQC on {blabel} ({shots} shots)"
        elif self.is_quantum:
            engine = "quantum PennyLane ideal"
        else:
            engine = "classical"

        prefix = ""
        if curated_fallback:
            prefix = (
                "Curated bank is ideal-backend only — running live inference on the "
                "selected noisy simulator instead. "
            )

        res.notes = (
            f"{prefix}Live generation from {self.ckpt_path.name} "
            f"({engine} forward), largest-fragment decode over {n_sample} draws "
            f"(showing {len(res.smiles)} random valid).  •  "
            f"{self.display_name} · {self.dataset}."
        )
        return res

    def compare_backends(
        self, n: int, seed: int = 0, shots: int = 8192
    ) -> dict[str, GenerationResult]:
        # Run the same latent batch through every backend (quantum models only).
        if not self.is_quantum:
            raise ValueError("compare_backends requires a quantum model")
        if not self.ckpt_path.exists():
            raise FileNotFoundError(f"No checkpoint: {self.ckpt_path}")

        if not self._loaded:
            self.load()

        rng = np.random.default_rng(seed)
        z_path = _DATA_DIR / f"compare_z_{self.slug}_{self._dskey}.npy"
        if z_path.exists():
            z_pool = np.load(str(z_path)).astype(np.float32)
            if len(z_pool) == 0:
                z_pool = None
        else:
            z_pool = None

        if z_pool is not None:
            n_sample = min(len(z_pool), max(n, 16))
            idx = rng.choice(len(z_pool), size=n_sample, replace=False)
            z = z_pool[idx]
            z_note = f"curated compare-z pool ({len(z_pool)} vectors)"
        else:
            n_sample = max(n * 2, 16)
            z = rng.normal(0, 1, size=(n_sample, Z)).astype(np.float32)
            z_note = "random z"

        out: dict[str, GenerationResult] = {}
        for bkey in BACKEND_KEYS:
            el, nl = self._forward_logits(z, backend_key=bkey, shots=shots)
            raw = logits_to_smiles(el, nl, seed=seed, largest_fragment=True, min_heavy=2)
            res = compute_metrics(raw)
            res.z_used = z
            res = _sample_valid_for_display(res, n, np.random.default_rng(seed))
            blabel = BACKEND_LABELS.get(bkey, bkey)
            res.notes = (
                f"Backend comparison: {blabel}, {shots} shots, same {z_note} "
                f"({n_sample} draws, showing {len(res.smiles)} random valid).  •  "
                f"{self.display_name} · {self.dataset}."
            )
            out[bkey] = res
        return out

    def generate(
        self,
        n: int,
        mode: str = "curated",
        seed: int = 0,
        backend_key: str = "ideal",
        shots: int = 8192,
        **kwargs,
    ) -> GenerationResult:
        if not isinstance(n, int) or not 1 <= n <= 16:
            raise ValueError("n must be an integer between 1 and 16")
        if mode not in {"curated", "random"}:
            raise ValueError(f"unsupported generation mode: {mode!r}")
        if backend_key not in BACKEND_KEYS:
            raise ValueError(f"unsupported backend: {backend_key!r}")
        if not isinstance(seed, int):
            raise ValueError("seed must be an integer")
        if not isinstance(shots, int) or shots <= 0:
            raise ValueError("shots must be a positive integer")

        if not self.available:
            res = GenerationResult(
                smiles=[],
                validity=0.0,
                uniqueness=0.0,
                sample_count=0,
                valid_count=0,
            )
            res.notes = (
                f"⚠ {self.display_name} was not trained on {self.dataset} — no "
                "checkpoint available for this combination. Try another dataset "
                "or model."
            )
            return res

        # Curated bank: instant reproducible molecules on the ideal backend only.
        if mode == "curated" and backend_key == "ideal" and self.bank_path.exists():
            return self._serve_bank(n)

        if not self.ckpt_path.exists():
            if self.bank_path.exists() and backend_key == "ideal":
                return self._serve_bank(n)
            res = GenerationResult(smiles=[], validity=0.0, uniqueness=0.0)
            res.notes = f"⚠ No checkpoint for {self.display_name} · {self.dataset}."
            return res

        # Curated + fake backend: try precomputed probs (instant, paper-quality).
        if mode == "curated" and _is_fake_backend(backend_key):
            result = self._serve_precomputed(n, backend_key, seed)
            if result is not None:
                return result

        curated_fallback = mode == "curated" and _is_fake_backend(backend_key)
        return self._live_generation(
            n,
            seed,
            backend_key,
            shots,
            curated_fallback=curated_fallback,
            n_draw=1000 if mode == "random" else None,
        )


def build_registry(dataset: str = "Both") -> dict[str, MoleculeGenerator]:
    return {name: MolGANGenerator(dataset, name) for name in MODEL_NAMES}
