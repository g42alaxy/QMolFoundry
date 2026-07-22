from __future__ import annotations

import functools
import html
import math

import gradio as gr

from models import BACKEND_KEYS, DEFAULT_SHOTS, build_registry
from utils import circuit_title, draw_vqc, empty_circuit_figure, smiles_grid_html

# ── constants ─────────────────────────────────────────────────────────────────
DEFAULT_DATASET = "Both"
SEED_MODES = {
    "Curated samples": "curated",
    "Random samples": "random",
}

_BACKEND_SHORT = ["Ideal", "Brisbane", "Sherbrooke", "Osaka"]
_SHORT_TO_KEY = dict(zip(_BACKEND_SHORT, BACKEND_KEYS))
_KEY_TO_SHORT = {v: k for k, v in _SHORT_TO_KEY.items()}

SHOTS_CHOICES = ["1024", "8192"]
DEFAULT_SHOTS_STR = str(DEFAULT_SHOTS)

# ── 3×3 model matrix ──────────────────────────────────────────────────────────
# gen:   "Classic" | "VVRQ" | "EFQ"
# cycle: "None"    | "Classical" | "Quantum"
_MODEL_MAP = {
    ("Classic", "None"): "MolGAN",
    ("VVRQ", "None"): "HQ-MolGAN (VVRQ)",
    ("EFQ", "None"): "HQ-MolGAN (EFQ)",
    ("Classic", "Classical"): "Cycle MolGAN",
    ("VVRQ", "Classical"): "HQ Cycle MolGAN (VVRQ)",
    ("EFQ", "Classical"): "HQ Cycle MolGAN (EFQ)",
    ("Classic", "Quantum"): "Hybrid Cycle MolGAN",
    ("VVRQ", "Quantum"): "Hybrid-Cycle HQ-MolGAN (VVRQ)",
    ("EFQ", "Quantum"): "Hybrid-Cycle HQ-MolGAN (EFQ)",
}


def _resolve(gen: str, cycle: str) -> str:
    return _MODEL_MAP[(gen, cycle)]


# ── model metadata ─────────────────────────────────────────────────────────────
_MINFO = {
    "MolGAN": dict(
        badge="Classic",
        bbg="#d1fae5",
        bfg="#065f46",
        desc="Baseline graph GAN: classical generator maps latent z → adjacency A and atom features X "
        "via Gumbel-softmax. R-GCN discriminator. No quantum, no cycle component.",
        arch="R-GCN · Gumbel-softmax · Wasserstein loss",
        qm9={"QED": "0.47", "SA": "4.46", "Unique": "9 %"},
    ),
    "HQ-MolGAN (VVRQ)": dict(
        badge="Quantum",
        bbg="#eef2ff",
        bfg="#3730a3",
        desc="Variational–Variational Rotation-Quantum circuit: 8-qubit PQC with RY angle-encoding "
        "of z, then 3 layers of Rot + CNOT ladder. Output probability vector feeds the classical decoder.",
        arch="8 qubits · 3 layers · RY encode · Rot+CNOT",
        qm9={"QED": "0.47", "SA": "4.24", "Unique": "9 %"},
    ),
    "HQ-MolGAN (EFQ)": dict(
        badge="Quantum",
        bbg="#eef2ff",
        bfg="#3730a3",
        desc="Efficient Feature-map Quantum circuit: denser angle-encoding with fewer entangling "
        "gates than VVRQ — faster while retaining quantum expressivity.",
        arch="8 qubits · 3 layers · dense feature map",
        qm9={"QED": "0.47", "SA": "4.44", "Unique": "10 %"},
    ),
    "Cycle MolGAN": dict(
        badge="Cycle",
        bbg="#fef3c7",
        bfg="#92400e",
        desc="Classic MolGAN + classical cycle-consistency: encoder reconstructs z̃ from the "
        "generated graph, ‖z − z̃‖ minimised jointly. Improves latent coverage and uniqueness.",
        arch="R-GCN + classical cycle encoder · GAN + recon loss",
        qm9={"QED": "0.54", "SA": "4.12", "Unique": "28 %"},
    ),
    "HQ Cycle MolGAN (VVRQ)": dict(
        badge="Q+Cycle ★",
        bbg="#fef3c7",
        bfg="#92400e",
        desc="VVRQ quantum generator combined with a classical cycle-consistency encoder. "
        "+30 % QED and +44 % uniqueness vs no-cycle baseline.",
        arch="VVRQ + classical cycle encoder · joint GAN + recon loss",
        qm9={"QED": "0.61", "SA": "3.38", "Unique": "53 %"},
    ),
    "HQ Cycle MolGAN (EFQ)": dict(
        badge="Q+Cycle ★",
        bbg="#fef3c7",
        bfg="#92400e",
        desc="EFQ quantum generator with classical cycle-consistency. "
        "Faster per-step than VVRQ variant, similar stability gains.",
        arch="EFQ + classical cycle encoder · joint GAN + recon loss",
        qm9={"QED": "0.58", "SA": "3.55", "Unique": "47 %"},
    ),
    "Hybrid Cycle MolGAN": dict(
        badge="QDI Cycle",
        bbg="#fdf4ff",
        bfg="#7e22ce",
        desc="Classic generator + quantum cycle (QDI chain): the cycle encoder is replaced by a "
        "parametrized quantum inverter circuit. Fully quantum cycle component, classical base.",
        arch="R-GCN + quantum QDI encoder · GAN + quantum recon loss",
        qm9={"QED": "—", "SA": "—", "Unique": "—"},
    ),
    "Hybrid-Cycle HQ-MolGAN (VVRQ)": dict(
        badge="Full-Q ★★",
        bbg="#fdf4ff",
        bfg="#7e22ce",
        desc="Full hybrid: VVRQ quantum generator + quantum cycle (QDI chain). "
        "Both the generation and the cycle-consistency encoder are presented as parametrized quantum circuits.",
        arch="VVRQ generator + quantum QDI cycle encoder",
        qm9={"QED": "—", "SA": "—", "Unique": "—"},
    ),
    "Hybrid-Cycle HQ-MolGAN (EFQ)": dict(
        badge="Full-Q ★★",
        bbg="#fdf4ff",
        bfg="#7e22ce",
        desc="Full hybrid: EFQ quantum generator + quantum cycle (QDI chain). "
        "Faster EFQ circuit combined with additional quantum circuit for cycle consistency.",
        arch="EFQ generator + quantum QDI cycle encoder",
        qm9={"QED": "—", "SA": "—", "Unique": "—"},
    ),
}

_BINFO = {
    "ideal": "Noiseless statevector (PennyLane default.qubit)",
    "fake_brisbane": "IBM Brisbane 127q Eagle r3 noise (Qiskit Aer)",
    "fake_sherbrooke": "IBM Sherbrooke 127q Eagle r3 noise (Qiskit Aer)",
    "fake_osaka": "IBM Osaka 127q Eagle r3 noise (Qiskit Aer)",
}


def _timing(key: str, shots: int) -> str:
    if key == "ideal":
        return "~0.1 s / circuit"
    t = shots / 8192  # 8192 shots ≈ 1 s reference
    if t < 60:
        return f"~{max(0.1, t):.1f} s / circuit"
    return f"~{t / 60:.1f} min / circuit"


# ── model selection helpers ───────────────────────────────────────────────────
def _effective_gen(gen_type: str, gen_arch: str) -> str:
    """Map (gen_type, gen_arch) → the gen key used in _MODEL_MAP."""
    return "Classic" if gen_type == "Classic" else gen_arch


# ── HTML builders ─────────────────────────────────────────────────────────────
def _gen_active_html(gen, cycle):
    name = _resolve(gen, cycle)
    m = _MINFO.get(name, {})
    return (
        f'<div class="comp-active-row">'
        f"Active: <strong>{name}</strong>"
        f'<span class="comp-badge" style="background:{m.get("bbg", "#eef2ff")};color:{m.get("bfg", "#3730a3")}">'
        f"{m.get('badge', '—')}</span></div>"
    )


def _cycle_active_html(gen, cycle):
    if cycle == "None":
        return '<div class="comp-active-row">No cycle consistency</div>'
    name = _resolve(gen, cycle)
    m = _MINFO.get(name, {})
    return (
        f'<div class="comp-active-row">'
        f"Active: <strong>{name}</strong>"
        f'<span class="comp-badge" style="background:{m.get("bbg", "#fef3c7")};color:{m.get("bfg", "#92400e")}">'
        f"{m.get('badge', 'Cycle')}</span></div>"
    )


def _spec_active_html(short_key, shots_str=DEFAULT_SHOTS_STR):
    key = _SHORT_TO_KEY.get(short_key, "ideal")
    desc = _BINFO.get(key, "—")
    if key == "ideal":
        timing = "exact statevector · shots N/A"
    else:
        timing = _timing(key, int(shots_str))
    return (
        f'<div class="comp-active-row spec-active">'
        f'<span style="color:var(--muted)">Device:</span> <strong>{desc}</strong>'
        f'<span class="spec-timing">{timing}</span></div>'
    )


def _model_info_html(gen, cycle, short_bk, shots_str=DEFAULT_SHOTS_STR):
    name = _resolve(gen, cycle)
    m = _MINFO.get(name, {})
    bkey = _SHORT_TO_KEY.get(short_bk, "ideal")
    bdesc = _BINFO.get(bkey, "—")
    btiming = _timing(bkey, int(shots_str))
    stats = "".join(
        f'<div class="mi-stat"><span class="st-lbl">{k}</span><span class="st-val">{v}</span></div>'
        for k, v in m.get("qm9", {}).items()
    )
    return (
        f'<div class="model-info-card">'
        f'<div class="mi-head">'
        f'  <span class="mi-name">{name}</span>'
        f'  <span class="mi-badge" style="background:{m.get("bbg", "#eef2ff")};color:{m.get("bfg", "#3730a3")}">'
        f"{m.get('badge', '—')}</span></div>"
        f'<div class="mi-desc">{m.get("desc", "")}</div>'
        f'<div class="mi-arch">Architecture: {m.get("arch", "")}</div>'
        f'<div class="mi-stats">{stats}</div>'
        f'<div class="mi-backend"><strong>Backend:</strong> {bdesc}'
        f'<span class="mi-timing">{btiming}</span></div>'
        f"</div>"
    )


def _chips_html(short_key="Ideal"):
    active = _SHORT_TO_KEY.get(short_key, "ideal")
    items = []
    for k in BACKEND_KEYS:
        cls = "ideal" if k == "ideal" else "noisy"
        dot = "● " if k == active else ""
        items.append(f'<span class="backend-chip {cls}">{dot}{_KEY_TO_SHORT[k]}</span>')
    return (
        '<div class="chips-section">'
        '<div class="section-lbl">Quantum backend</div>'
        '<div class="chips-row">' + "".join(items) + "</div></div>"
    )


def _metrics_html(
    validity=None,
    uniqueness=None,
    qed=None,
    sa=None,
    logp=None,
    n_valid=None,
    n_sample=None,
    validity_active=True,
    metrics_over_all=False,
):
    def f(v, pct=False):
        if v is None or (isinstance(v, float) and math.isnan(v)):
            return "—"
        return f"{v:.1%}" if pct else f"{v:.3f}"

    if validity_active:
        validity_value = f(validity, pct=True)
        validity_sub = "fraction of valid draws"
    else:
        validity_value = "—"
        validity_sub = "n/a for curated samples"

    if metrics_over_all and n_valid is not None:
        over = f"over all {n_valid} valid"
        qed_sub = f"drug-likeness ↑ · {over}"
        sa_sub = f"synth. access. ↓ · {over}"
        logp_sub = f"lipophilicity · {over}"
    else:
        qed_sub = "drug-likeness ↑"
        sa_sub = "synth. access. ↓"
        logp_sub = "lipophilicity"

    cards = [
        ("Mean QED", f(qed), qed_sub, False),
        ("Mean SA", f(sa), sa_sub, False),
        ("Mean logP", f(logp), logp_sub, False),
        ("Uniqueness", f(uniqueness, pct=True), "distinct", False),
        ("Validity", validity_value, validity_sub, not validity_active),
    ]
    body = "".join(
        f'<div class="metric-card{" inactive" if inactive else ""}">'
        f'<div class="m-lbl">{label}</div>'
        f'<div class="m-val">{value}</div><div class="m-sub">{subtitle}</div></div>'
        for label, value, subtitle, inactive in cards
    )
    return f'<div class="metrics-grid">{body}</div>'


# ── static HTML ───────────────────────────────────────────────────────────────
_HERO = """
<div id="hero-html">
  <div class="hero-chip">⚛ QMolFoundry</div>
  <h1>Drug molecule generation via hybrid quantum algorithms (GANs)</h1>
  <p>Parametrized quantum circuits inside a GAN — ideal noiseless simulator or IBM Eagle device
  noise model. Same trained weights, different quantum backend: watch how hardware noise shifts
  generated molecules and drug-likeness metrics.</p>
  <div class="hero-pills">
    <span class="pill">QM9+PC9 joint training</span>
    <span class="pill">8 qubits · 3 PQC layers</span>
    <span class="pill">QED / SA / logP metrics</span>
    <span class="pill">IBM Brisbane · Sherbrooke · Osaka devices</span>
  </div>
</div>"""

_PAPER_HTML = """
<div id="paper-results">

  <div class="pr-card">
    <div class="pr-card-header">
      <span class="pr-title">Key results — QM9 dataset</span>
      <span class="pr-sub">Anosin M.A., ITMO University, 2025</span>
    </div>

    <div class="pr-group-label">No cycle consistency</div>
    <table class="pr-table">
      <thead>
        <tr>
          <th class="col-model">Model</th>
          <th>Validity</th><th>Uniqueness</th>
          <th>QED <span class="th-arrow">↑</span></th>
          <th>SA <span class="th-arrow">↓</span></th>
          <th>logP</th>
        </tr>
      </thead>
      <tbody>
        <tr>
          <td class="col-model"><span class="model-name">MolGAN</span>
            <span class="model-badge classic">Classic</span></td>
          <td>0.98</td><td>0.09</td><td>0.47</td><td>4.46</td><td>1.49</td>
        </tr>
        <tr>
          <td class="col-model"><span class="model-name">HQ-MolGAN (VVRQ)</span>
            <span class="model-badge quantum">Quantum</span></td>
          <td>0.98</td><td>0.09</td><td>0.47</td><td>4.24</td><td>1.41</td>
        </tr>
        <tr>
          <td class="col-model"><span class="model-name">HQ-MolGAN (EFQ)</span>
            <span class="model-badge quantum">Quantum</span></td>
          <td>0.98</td><td>0.10</td><td>0.47</td><td>4.44</td><td>1.47</td>
        </tr>
      </tbody>
    </table>

    <div class="pr-group-label" style="margin-top:1.1rem">Classical cycle consistency</div>
    <table class="pr-table">
      <thead>
        <tr>
          <th class="col-model">Model</th>
          <th>Validity</th><th>Uniqueness</th>
          <th>QED <span class="th-arrow">↑</span></th>
          <th>SA <span class="th-arrow">↓</span></th>
          <th>logP</th>
        </tr>
      </thead>
      <tbody>
        <tr>
          <td class="col-model"><span class="model-name">Cycle MolGAN</span>
            <span class="model-badge classic">Classic</span></td>
          <td>0.98</td><td>0.28</td><td>0.54</td><td>4.12</td><td>1.38</td>
        </tr>
        <tr class="pr-best-row">
          <td class="col-model">
            <span class="model-name">HQ Cycle MolGAN (VVRQ)</span>
            <span class="model-badge best">Best ★</span>
          </td>
          <td><strong>0.98</strong></td>
          <td><strong>0.53</strong></td>
          <td><strong>0.61</strong></td>
          <td><strong>3.38</strong></td>
          <td><strong>1.05</strong></td>
        </tr>
        <tr>
          <td class="col-model"><span class="model-name">HQ Cycle MolGAN (EFQ)</span>
            <span class="model-badge quantum">Q+Cycle</span></td>
          <td>0.98</td><td>0.47</td><td>0.58</td><td>3.55</td><td>1.12</td>
        </tr>
      </tbody>
    </table>

    <div class="pr-group-label" style="margin-top:1.1rem">Quantum cycle — QDI circuit</div>
    <table class="pr-table">
      <thead>
        <tr>
          <th class="col-model">Model</th>
          <th>Validity</th><th>Uniqueness</th>
          <th>QED <span class="th-arrow">↑</span></th>
          <th>SA <span class="th-arrow">↓</span></th>
          <th>logP</th>
        </tr>
      </thead>
      <tbody>
        <tr class="pr-pending">
          <td class="col-model"><span class="model-name">Hybrid Cycle MolGAN</span>
            <span class="model-badge qdi">QDI</span></td>
          <td>—</td><td>—</td><td>—</td><td>—</td><td>—</td>
        </tr>
        <tr class="pr-pending">
          <td class="col-model"><span class="model-name">Hybrid-Cycle HQ-MolGAN (VVRQ)</span>
            <span class="model-badge qdi">QDI</span></td>
          <td>—</td><td>—</td><td>—</td><td>—</td><td>—</td>
        </tr>
        <tr class="pr-pending">
          <td class="col-model"><span class="model-name">Hybrid-Cycle HQ-MolGAN (EFQ)</span>
            <span class="model-badge qdi">QDI</span></td>
          <td>—</td><td>—</td><td>—</td><td>—</td><td>—</td>
        </tr>
      </tbody>
    </table>

    <div class="pr-footnote">
      Hybrid Cycle (QDI) results pending publication. All other results: QM9 dataset, 200k shots.
    </div>
  </div>

</div>"""

# ── CSS ───────────────────────────────────────────────────────────────────────
_CSS = """
@import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@400;500;600&family=IBM+Plex+Mono:wght@400;500&display=swap');

:root {
    --ink:      #111827;
    --ink2:     #374151;
    --muted:    #6b7280;
    --border:   #e5e7eb;
    --surface:  #f8fafc;
    --teal:     #0d9488;
    --teal-bg:  #f0fdfa;
    --teal-bdr: #5eead4;
    --blue-bg:  #eff6ff;
    --blue-bdr: #93c5fd;
    --ind-bg:   #eef2ff;
    --ind-bdr:  #a5b4fc;
    --amb-bg:   #fffbeb;
    --amb-bdr:  #fcd34d;
    --pur-bg:   #fdf4ff;
    --pur-bdr:  #d8b4fe;
    --radius:   12px;
    --pill:     999px;
}

body, .gradio-container { font-family: 'Space Grotesk', sans-serif !important; }
code, pre, .mi-arch, .mi-timing, .st-val, .m-val { font-family: 'IBM Plex Mono', monospace !important; }

/* ══════════════════════════════════════════════
   HIDE GRADIO LOADING / PROGRESS BARS
   The gray bar that appears under radio buttons
   during callbacks is the Gradio status tracker.
   ══════════════════════════════════════════════ */
.progress-bar,
.progress-bar-inner,
.progress-level,
.progress-level-inner,
.eta-bar,
.loader,
.progress-text,
.meta-text,
.meta-text-center,
.progress-bar-wrap,
.status-tracker         { display: none !important; }

/* ── hero ── */
#hero-html {
    background: linear-gradient(135deg, #0f172a 0%, #1e3a5f 60%, #0f4c75 100%);
    border-radius: var(--radius); padding: 2rem 2.25rem 1.75rem; margin-bottom: 1.25rem;
}
#hero-html .hero-chip {
    display:inline-flex; align-items:center; font-size:11px; font-weight:600;
    letter-spacing:.06em; text-transform:uppercase;
    background:rgba(255,255,255,.12); border:1px solid rgba(255,255,255,.2);
    border-radius:var(--pill); padding:4px 12px; margin-bottom:.9rem; color:#a5f3fc;
}
#hero-html h1 { font-size:1.75rem; font-weight:600; margin:0 0 .55rem; color:#fff; }
#hero-html p  { font-size:.92rem; color:#cbd5e1; margin:0; line-height:1.65; }
#hero-html .hero-pills { display:flex; flex-wrap:wrap; gap:8px; margin-top:1.1rem; }
#hero-html .pill {
    font-size:11px; font-weight:500; padding:3px 11px; border-radius:var(--pill);
    background:rgba(255,255,255,.1); color:#e2e8f0; border:1px solid rgba(255,255,255,.15);
}

/* ════════════════════════════════════════
   CARD SHELLS
   ════════════════════════════════════════ */
#gen-card, #cycle-card, #backend-spec-card {
    background: #fff !important;
    border: 1px solid var(--border) !important;
    border-radius: var(--radius) !important;
    overflow: hidden !important;
    padding: 0 !important;
    gap: 0 !important;
    margin-bottom: .8rem !important;
    box-shadow: 0 1px 4px rgba(0,0,0,.06) !important;
}

.card-header-html {
    font-size: 11px; font-weight:600; letter-spacing:.08em; text-transform:uppercase;
    color: var(--muted); padding: .6rem 1rem;
    border-bottom: 1px solid var(--border);
    background: var(--surface);
}

/* Strip ALL Gradio decoration (block + form + fieldset = the 3 layers of dark bg) */
#gen-card .block,       #gen-card .form,       #gen-card fieldset,
#cycle-card .block,     #cycle-card .form,     #cycle-card fieldset,
#backend-spec-card .block, #backend-spec-card .form, #backend-spec-card fieldset {
    background: transparent !important;
    border: none !important;
    box-shadow: none !important;
    padding: 0 !important;
    margin: 0 !important;
    overflow: visible !important;
}

/* ── card rows ── */
#comp-gen-row, #comp-arch-row, #comp-cycle-row,
#spec-device-row, #spec-shots-row {
    padding: .5rem 1rem !important;
    border-bottom: 1px solid var(--border) !important;
    gap: .75rem !important;
    align-items: center !important;
    margin: 0 !important;
    min-height: 0 !important;
}

.comp-lbl-cell {
    flex: 0 0 60px !important;
    min-width: 0 !important;
    padding: 0 !important;
}
.comp-lbl {
    font-size: 10.5px; font-weight:600; letter-spacing:.07em; text-transform:uppercase;
    color: var(--muted); white-space: nowrap;
}

.comp-active-row {
    padding: .45rem 1rem .5rem;
    font-size: 12.5px; color: var(--muted);
    border-top: 1px solid var(--border);
    background: var(--surface);
    display: flex; align-items: center; flex-wrap: wrap; gap: 5px;
}
.comp-active-row strong { color: var(--ink); }
.comp-badge { font-size:10.5px; font-weight:600; padding:2px 9px; border-radius:var(--pill); flex-shrink:0; }
.spec-timing { font-size:10.5px; color:var(--muted); font-family:'IBM Plex Mono',monospace; margin-left:6px; }

/* ════════════════════════════════════════════════════════
   PILL RADIO BUTTONS

   Key: use :nth-child(n) (= every child) for BASE style
   and explicit :nth-child(N) for active states.
   Both have higher CSS specificity than Gradio's
   .selected class rule — so dark bg never shows.
   ════════════════════════════════════════════════════════ */

.comp-radio .wrap {
    display: flex !important; flex-direction: row !important;
    flex-wrap: wrap !important; gap: 6px !important;
    background: transparent !important; border: none !important; padding: 0 !important;
}
.comp-radio .wrap input[type=radio] {
    position: absolute !important; opacity: 0 !important;
    width: 0 !important; height: 0 !important; pointer-events: none !important;
}

/* ── base pill style (all children, beats Gradio .selected) ── */
#gen-type-radio .wrap label:nth-child(n),
#gen-arch-radio .wrap label:nth-child(n),
#cycle-radio .wrap label:nth-child(n),
#backend-radio .wrap label:nth-child(n),
#shots-radio .wrap label:nth-child(n) {
    display: inline-flex !important; align-items: center !important;
    padding: 5px 16px !important;
    border: 1.5px solid var(--border) !important;
    border-radius: var(--pill) !important;
    background: #fff !important; color: var(--ink2) !important;
    cursor: pointer !important;
    font-size: 13px !important; font-weight: 500 !important;
    font-family: 'Space Grotesk', sans-serif !important;
    white-space: nowrap !important;
    transition: background .12s, border-color .12s, color .12s !important;
    flex-shrink: 0 !important; user-select: none !important;
    line-height: 1.3 !important; box-shadow: none !important;
}

/* hover */
#gen-type-radio .wrap label:nth-child(n):hover,
#gen-arch-radio .wrap label:nth-child(n):hover,
#cycle-radio .wrap label:nth-child(n):hover,
#backend-radio .wrap label:nth-child(n):hover,
#shots-radio .wrap label:nth-child(n):hover {
    border-color: var(--blue-bdr) !important; background: var(--blue-bg) !important;
}

/* ── GENERATOR TYPE active states ──
   [1]=Classic(teal)  [2]=Quantum(indigo)  */
#gen-type-radio .wrap label:nth-child(1):has(input:checked) {
    background: var(--teal-bg) !important; border-color: var(--teal-bdr) !important;
    color: #065f46 !important; font-weight:600 !important;
}
#gen-type-radio .wrap label:nth-child(2):has(input:checked) {
    background: var(--ind-bg) !important; border-color: var(--ind-bdr) !important;
    color: #3730a3 !important; font-weight:600 !important;
}

/* ── GENERATOR CIRCUIT (arch) active states ── */
#gen-arch-radio .wrap label:nth-child(1):has(input:checked),
#gen-arch-radio .wrap label:nth-child(2):has(input:checked) {
    background: var(--ind-bg) !important; border-color: var(--ind-bdr) !important;
    color: #3730a3 !important; font-weight:600 !important;
}

/* Disabled arch pills — when Classic is selected */
#gen-arch-radio .wrap label:has(input:disabled) {
    opacity: 0.35 !important; cursor: not-allowed !important;
    pointer-events: none !important;
}

/* ── CYCLE active states ──
   [1]=None(neutral)  [2]=Classical(amber)  [3]=Quantum(purple)  */
#cycle-radio .wrap label:nth-child(1):has(input:checked) {
    background: var(--surface) !important; border-color: var(--border) !important;
    color: var(--muted) !important; font-weight:500 !important;
}
#cycle-radio .wrap label:nth-child(2):has(input:checked) {
    background: var(--amb-bg) !important; border-color: var(--amb-bdr) !important;
    color: #92400e !important; font-weight:600 !important;
}
#cycle-radio .wrap label:nth-child(3):has(input:checked) {
    background: var(--pur-bg) !important; border-color: var(--pur-bdr) !important;
    color: #7e22ce !important; font-weight:600 !important;
}

/* ── BACKEND active states ──
   [1]=Ideal(teal)   [2,3,4]=IBM(blue)  */
#backend-radio .wrap label:nth-child(1):has(input:checked) {
    background: var(--teal-bg) !important; border-color: var(--teal-bdr) !important;
    color: #065f46 !important; font-weight:600 !important;
}
#backend-radio .wrap label:nth-child(2):has(input:checked),
#backend-radio .wrap label:nth-child(3):has(input:checked),
#backend-radio .wrap label:nth-child(4):has(input:checked) {
    background: var(--blue-bg) !important; border-color: var(--blue-bdr) !important;
    color: #1e40af !important; font-weight:600 !important;
}

/* ── SHOTS: any = indigo ── */
#shots-radio .wrap label:nth-child(1):has(input:checked),
#shots-radio .wrap label:nth-child(2):has(input:checked),
#shots-radio .wrap label:nth-child(3):has(input:checked) {
    background: var(--ind-bg) !important; border-color: var(--ind-bdr) !important;
    color: #3730a3 !important; font-weight:600 !important;
}

/* Ideal backend: shots are irrelevant (statevector) */
#shots-radio .wrap label:has(input:disabled) {
    opacity: 0.35 !important;
    cursor: not-allowed !important;
    pointer-events: none !important;
    background: var(--surface) !important;
    border-color: var(--border) !important;
    color: var(--muted) !important;
    font-weight: 500 !important;
}
#spec-shots-row.shots-disabled {
    opacity: 0.55;
}

/* Compare backends: classical generators have no PQC */
#cmp-panel.cmp-disabled {
    opacity: 0.5;
    pointer-events: none;
}
#cmp-panel.cmp-disabled button {
    cursor: not-allowed !important;
}
.cmp-unavailable {
    font-size: 13.5px;
    color: var(--muted);
    margin: 0 0 .75rem;
    line-height: 1.6;
    padding: .65rem .85rem;
    border: 1px dashed var(--border);
    border-radius: var(--radius);
    background: var(--surface);
}

/* ── backend chips ── */
.chips-section { margin-bottom:.8rem; }
.section-lbl {
    font-size:10.5px; font-weight:600; letter-spacing:.08em; text-transform:uppercase;
    color:var(--muted); margin-bottom:.5rem;
}
.chips-row { display:flex; flex-wrap:wrap; gap:6px; }
.backend-chip {
    font-size:12px; font-weight:500; padding:4px 13px; border-radius:var(--pill);
    border:1.5px solid var(--border); background:#fff; color:var(--ink2);
}
.backend-chip.ideal { border-color:var(--teal-bdr); background:var(--teal-bg); color:#065f46; }
.backend-chip.noisy { border-color:var(--blue-bdr); background:var(--blue-bg); color:#1e40af; }

/* ── circuit ── */
.plot-container, .gradio-plot { background:var(--surface) !important; }

/* ── model info panel ── */
.model-info-card {
    background:#fff; border:1px solid var(--border); border-radius:var(--radius);
    padding:1rem 1.1rem; margin-bottom:.8rem;
    box-shadow:0 1px 4px rgba(0,0,0,.05);
}
.mi-head { display:flex; align-items:center; gap:8px; margin-bottom:.5rem; }
.mi-name { font-size:15px; font-weight:600; color:var(--ink); }
.mi-badge { font-size:10.5px; font-weight:600; padding:2px 10px; border-radius:var(--pill); }
.mi-desc { font-size:13px; color:var(--ink2); line-height:1.6; margin-bottom:.55rem; }
.mi-arch { font-size:11.5px; color:var(--muted); margin-bottom:.55rem; }
.mi-stats { display:flex; gap:8px; flex-wrap:wrap; margin-bottom:.55rem; }
.mi-stat  { background:var(--surface); border:1px solid var(--border); border-radius:8px; padding:.3rem .7rem; font-size:11.5px; }
.st-lbl { color:var(--muted); }
.st-val { font-weight:600; color:var(--ink); margin-left:4px; }
.mi-backend { font-size:12px; color:var(--muted); border-top:1px solid var(--border); padding-top:.45rem; }
.mi-backend strong { color:var(--ink2); }
.mi-timing { font-size:10.5px; margin-left:8px; }

/* ── metrics grid ── */
#metrics-html .metrics-grid { display:grid; grid-template-columns:repeat(auto-fit,minmax(110px,1fr)); gap:8px; }
.metric-card { background:var(--surface); border:1px solid var(--border); border-radius:var(--radius); padding:.65rem .9rem; }
.metric-card .m-lbl { font-size:11px; color:var(--muted); margin:0 0 3px; }
.metric-card .m-val { font-size:22px; font-weight:600; color:var(--ink); margin:0; line-height:1.1; }
.metric-card .m-sub { font-size:10.5px; color:var(--muted); margin:3px 0 0; }
.metric-card.inactive { opacity:.45; }
.metric-card.inactive .m-val { color:var(--muted); font-weight:500; }

/* ── compare ── */
.bm-row { display:flex; align-items:center; gap:10px; font-size:13px; color:var(--ink2); padding:6px 0; border-bottom:1px solid var(--border); }
.bm-row:last-child { border-bottom:none; }
.bm-name { font-weight:600; min-width:130px; }
.bm-val  { font-family:'IBM Plex Mono',monospace; font-size:12.5px; }
.bm-ideal { color:var(--teal); }
.bm-noisy { color:#2563eb; }

/* ── paper results ── */
#paper-results { padding: .25rem 0; }

.pr-card {
    background: #fff;
    border: 1px solid var(--border);
    border-radius: var(--radius);
    overflow: hidden;
    box-shadow: 0 1px 4px rgba(0,0,0,.06);
}
.pr-card-header {
    display: flex; align-items: baseline; justify-content: space-between;
    flex-wrap: wrap; gap: 6px;
    padding: .75rem 1.1rem;
    border-bottom: 1px solid var(--border);
    background: var(--surface);
}
.pr-title { font-size: 13px; font-weight: 600; color: var(--ink); }
.pr-sub   { font-size: 11.5px; color: var(--muted); }

.pr-group-label {
    font-size: 10.5px; font-weight: 600; letter-spacing: .07em; text-transform: uppercase;
    color: var(--muted); padding: .55rem 1.1rem .2rem;
}

.pr-table {
    width: 100%; border-collapse: collapse;
    font-size: 13px; font-family: 'Space Grotesk', sans-serif;
}
.pr-table thead th {
    background: var(--surface);
    font-size: 10.5px; font-weight: 600; letter-spacing: .06em; text-transform: uppercase;
    color: var(--muted);
    padding: 6px 12px; text-align: left;
    border-top: 1px solid var(--border);
    border-bottom: 1px solid var(--border);
    white-space: nowrap;
}
.pr-table thead th:not(.col-model) { text-align: right; }
.pr-table td {
    padding: 9px 12px; color: var(--ink2);
    border-bottom: 1px solid var(--border);
    vertical-align: middle;
}
.pr-table td:not(.col-model) {
    text-align: right;
    font-family: 'IBM Plex Mono', monospace;
    font-size: 12.5px;
}
.pr-table tbody tr:last-child td { border-bottom: none; }
.pr-table tbody tr:hover td { background: var(--surface); }

.col-model { min-width: 220px; }
.model-name { font-weight: 500; color: var(--ink); margin-right: 7px; }
.th-arrow { font-size: 10px; }

.model-badge {
    display: inline-block;
    font-size: 10px; font-weight: 600;
    padding: 1px 7px; border-radius: 999px;
    vertical-align: middle; white-space: nowrap;
}
.model-badge.classic { background: var(--teal-bg);  color: #065f46; border: 1px solid var(--teal-bdr); }
.model-badge.quantum { background: var(--ind-bg);   color: #3730a3; border: 1px solid var(--ind-bdr); }
.model-badge.best    { background: #fef9c3;          color: #854d0e; border: 1px solid #fde047; }
.model-badge.qdi     { background: var(--pur-bg);   color: #7e22ce; border: 1px solid var(--pur-bdr); }

.pr-best-row td { background: #f0fdf4 !important; }
.pr-best-row td strong { color: var(--ink); }
.pr-best-row .col-model .model-name { color: #065f46; font-weight: 600; }

.pr-pending td { color: var(--muted) !important; font-style: italic; }
.pr-pending .model-name { color: var(--muted) !important; font-style: normal; }

.pr-footnote {
    font-size: 11.5px; color: var(--muted);
    padding: .6rem 1.1rem .75rem;
    border-top: 1px solid var(--border);
    background: var(--surface);
    line-height: 1.55;
}

.notes-box { background:var(--amb-bg); border-left:3px solid #d97706; border-radius:0 var(--radius) var(--radius) 0; padding:.65rem 1rem; font-size:12.5px; color:#78350f; margin-top:.75rem; }
footer { display:none !important; }
"""


# ── registry ──────────────────────────────────────────────────────────────────
@functools.lru_cache(maxsize=None)
def _registry():
    return build_registry(dataset=DEFAULT_DATASET)


def _avg(xs):
    xs = [x for x in xs if x == x]
    return sum(xs) / len(xs) if xs else float("nan")


def _compare_intro_html(is_quantum: bool) -> str:
    if is_quantum:
        return (
            '<p style="font-size:13.5px;color:var(--muted);margin-bottom:.75rem;line-height:1.6">'
            "Run the <strong>same latent batch</strong> through all four backends — "
            "ideal PennyLane simulator vs Brisbane / Sherbrooke / Osaka noisy "
            "simulators (Qiskit Aer). Works for every available "
            "<strong>VVRQ</strong> or <strong>EFQ</strong> checkpoint.</p>"
        )
    return (
        '<p class="cmp-unavailable">'
        "<strong>Unavailable</strong> for classical generators — backend comparison "
        "needs a quantum model: VVRQ or EFQ. "
        "Switch Generator type to Quantum.</p>"
    )


# ── callbacks ─────────────────────────────────────────────────────────────────
def on_change(gen_type, gen_arch, cycle, short_bk, shots_str):
    gen = _effective_gen(gen_type, gen_arch)
    name = _resolve(gen, cycle)
    is_q = _registry()[name].is_quantum
    # Quantum models draw the arch-specific ansatz (VVRQ vs EFQ); the classical
    # "Classic" generator has no circuit.
    arch = gen_arch if gen_type == "Quantum" else "VVRQ"
    if is_q:
        fig = draw_vqc(arch=arch)
        label = circuit_title(arch)
    else:
        fig = empty_circuit_figure()
        label = "Quantum circuit"
    arch_interactive = gen_type == "Quantum"
    return (
        gr.update(value=fig, visible=is_q, label=label),
        gr.update(interactive=arch_interactive),
        _gen_active_html(gen, cycle),
        _cycle_active_html(gen, cycle),
        _model_info_html(gen, cycle, short_bk, shots_str),
        gr.update(interactive=is_q),
        _compare_intro_html(is_q),
        gr.update(elem_classes=[] if is_q else ["cmp-disabled"]),
    )


def on_backend_change(short_bk, shots_str, gen_type, gen_arch, cycle):
    gen = _effective_gen(gen_type, gen_arch)
    shots_active = short_bk != "Ideal"
    return (
        _chips_html(short_bk),
        _spec_active_html(short_bk, shots_str),
        _model_info_html(gen, cycle, short_bk, shots_str),
        gr.update(interactive=shots_active),
        gr.update(elem_classes=["shots-disabled"] if not shots_active else []),
    )


def run_generation(gen_type, gen_arch, cycle, seed_label, short_bk, shots_str, n_mols, seed):
    gen = _effective_gen(gen_type, gen_arch)
    name = _resolve(gen, cycle)
    model = _registry()[name]
    bkey = _SHORT_TO_KEY.get(short_bk, "ideal")
    mode = SEED_MODES[seed_label]

    try:
        result = model.generate(
            n=int(n_mols),
            mode=mode,
            seed=int(seed),
            backend_key=bkey,
            shots=int(shots_str),
        )
    except (TypeError, ValueError, FileNotFoundError) as exc:
        message = html.escape(str(exc))
        return (
            smiles_grid_html([]),
            _metrics_html(),
            f'<div class="notes-box">Generation failed: {message}</div>',
        )

    n = len(result.smiles)
    cols = min(5, n)
    legs = [f"QED {q:.2f}" if q == q else "n/a" for q in result.qed] if result.qed else None
    mol_html = smiles_grid_html(result.smiles, legends=legs, cols=cols)
    curated = mode == "curated"
    metrics = _metrics_html(
        validity=None if curated else result.validity,
        uniqueness=result.uniqueness,
        qed=result.mean_qed,
        sa=result.mean_sa,
        logp=result.mean_logp,
        n_valid=None if curated else result.valid_count,
        n_sample=None if curated else result.sample_count,
        validity_active=not curated,
        metrics_over_all=not curated,
    )
    return mol_html, metrics, ""


def compare_backends(gen_type, gen_arch, cycle, seed_label, shots_str, n_mols, seed):
    """Compare Ideal vs fake IBM device backends on the same latent batch."""
    gen = _effective_gen(gen_type, gen_arch)
    name = _resolve(gen, cycle)
    model = _registry()[name]
    blank = smiles_grid_html([])
    shots = int(shots_str)

    if not model.is_quantum or not model.available:
        return (
            blank,
            blank,
            blank,
            blank,
            (
                '<div class="notes-box">Backend comparison requires an available '
                "<strong>quantum</strong> model (VVRQ or EFQ) with a trained Both-dataset "
                "checkpoint.</div>"
            ),
        )

    if not model.ckpt_path.exists():
        return (
            blank,
            blank,
            blank,
            blank,
            (
                f'<div class="notes-box">No checkpoint for <strong>{name}</strong> — '
                "cannot run live backend comparison.</div>"
            ),
        )

    try:
        results = model.compare_backends(n=int(n_mols), seed=int(seed), shots=shots)
    except (TypeError, ValueError, FileNotFoundError, RuntimeError) as exc:
        return (
            blank,
            blank,
            blank,
            blank,
            (f'<div class="notes-box">Backend comparison failed: {html.escape(str(exc))}</div>'),
        )

    grids, rows = [], []
    for bkey in BACKEND_KEYS:
        r = results[bkey]
        cols = min(5, len(r.smiles)) or 1
        legs = [f"QED {q:.2f}" if q == q else "n/a" for q in r.qed] if r.qed else None
        grids.append(smiles_grid_html(r.smiles, legends=legs, cols=cols))
        short = _KEY_TO_SHORT.get(bkey, bkey)
        cls = "bm-ideal" if bkey == "ideal" else "bm-noisy"
        rows.append(
            f'<div class="bm-row"><span class="bm-name">{short}</span>'
            f'<span class="bm-val {cls}">QED {r.mean_qed:.3f}</span>'
            f'<span class="bm-val">valid {r.validity:.0%}</span>'
            f'<span class="bm-val">unique {r.uniqueness:.0%}</span></div>'
        )
    foot = (
        f'<div style="margin-top:.5rem;font-size:12px;color:var(--muted)">'
        f"Same random z batch · {shots} shots · live PQC per backend "
        f"(the Generate-tab seed mode does not alter this controlled comparison)</div>"
    )
    return (
        grids[0],
        grids[1],
        grids[2],
        grids[3],
        '<div style="margin-top:.5rem">' + "".join(rows) + foot + "</div>",
    )


# ── theme ─────────────────────────────────────────────────────────────────────
def _theme():
    return gr.themes.Soft(
        font=[gr.themes.GoogleFont("Space Grotesk"), "sans-serif"],
        font_mono=[gr.themes.GoogleFont("IBM Plex Mono"), "monospace"],
        primary_hue=gr.themes.colors.blue,
        neutral_hue=gr.themes.colors.slate,
    ).set(
        body_background_fill="#f1f5f9",
        block_background_fill="#ffffff",
        block_border_width="1px",
        block_border_color="#e2e8f0",
        block_radius="10px",
    )


# ── UI ─────────────────────────────────────────────────────────────────────────
def build_ui():
    with gr.Blocks(title="Hybrid Quantum MolGAN") as demo:
        gr.HTML(_HERO)

        with gr.Row(equal_height=False):
            # ══════════════════════════════
            #  LEFT SIDEBAR
            # ══════════════════════════════
            with gr.Column(scale=3, min_width=270):
                # ── CARD 1: GENERATOR ──
                with gr.Column(elem_id="gen-card"):
                    gr.HTML('<div class="card-header-html">Generator</div>')

                    with gr.Row(elem_id="comp-gen-row"):
                        gr.HTML(
                            '<div class="comp-lbl-cell"><span class="comp-lbl">Type</span></div>'
                        )
                        gen_type_radio = gr.Radio(
                            ["Classic", "Quantum"],
                            value="Quantum",
                            show_label=False,
                            elem_id="gen-type-radio",
                            elem_classes=["comp-radio"],
                        )

                    with gr.Row(elem_id="comp-arch-row"):
                        gr.HTML(
                            '<div class="comp-lbl-cell"><span class="comp-lbl">Circuit</span></div>'
                        )
                        gen_arch_radio = gr.Radio(
                            ["VVRQ", "EFQ"],
                            value="VVRQ",
                            show_label=False,
                            elem_id="gen-arch-radio",
                            elem_classes=["comp-radio"],
                            interactive=True,
                        )

                    gen_active = gr.HTML(_gen_active_html("VVRQ", "None"))

                # ── CARD 2: CYCLE CONSISTENCY ──
                with gr.Column(elem_id="cycle-card"):
                    gr.HTML('<div class="card-header-html">Cycle consistency</div>')

                    with gr.Row(elem_id="comp-cycle-row"):
                        gr.HTML(
                            '<div class="comp-lbl-cell"><span class="comp-lbl">Mode</span></div>'
                        )
                        cycle_radio = gr.Radio(
                            ["None", "Classical", "Quantum"],
                            value="None",
                            show_label=False,
                            elem_id="cycle-radio",
                            elem_classes=["comp-radio"],
                        )

                    cycle_active = gr.HTML(_cycle_active_html("VVRQ", "None"))

                # ── CARD 3: BACKEND SPECIFICATION ──
                with gr.Column(elem_id="backend-spec-card"):
                    gr.HTML('<div class="card-header-html">Backend specification</div>')

                    with gr.Row(elem_id="spec-device-row"):
                        gr.HTML(
                            '<div class="comp-lbl-cell"><span class="comp-lbl">Device</span></div>'
                        )
                        backend_radio = gr.Radio(
                            _BACKEND_SHORT,
                            value="Ideal",
                            show_label=False,
                            elem_id="backend-radio",
                            elem_classes=["comp-radio"],
                        )

                    shots_row = gr.Row(elem_id="spec-shots-row", elem_classes=["shots-disabled"])
                    with shots_row:
                        gr.HTML(
                            '<div class="comp-lbl-cell"><span class="comp-lbl">Shots</span></div>'
                        )
                        shots_radio = gr.Radio(
                            SHOTS_CHOICES,
                            value=DEFAULT_SHOTS_STR,
                            show_label=False,
                            elem_id="shots-radio",
                            elem_classes=["comp-radio"],
                            interactive=False,
                        )

                    spec_active = gr.HTML(_spec_active_html("Ideal", DEFAULT_SHOTS_STR))

                # ── Chips + Circuit ──
                chips_out = gr.HTML(_chips_html("Ideal"))
                circuit_plot = gr.Plot(
                    label="Quantum circuit — VVRQ · 8 qubits · depth 3",
                    visible=True,
                )

            # ══════════════════════════════
            #  RIGHT PANEL
            # ══════════════════════════════
            with gr.Column(scale=7):
                model_info_out = gr.HTML(
                    _model_info_html("VVRQ", "None", "Ideal", DEFAULT_SHOTS_STR)
                )

                with gr.Row():
                    seed_mode_dd = gr.Dropdown(
                        list(SEED_MODES.keys()),
                        value="Curated samples",
                        label="Generation mode",
                        info="Curated: sampling from precomputed bank of probs."
                        " Random: live sampling from N(0,I).",
                        scale=1,
                    )
                    seed_num = gr.Number(
                        value=42,
                        label="Seed",
                        info="Used for Random samples and curated+noisy backends "
                        "(Curated + Ideal serves the frozen bank)",
                        precision=0,
                        scale=1,
                    )
                    n_slider = gr.Slider(
                        1,
                        16,
                        value=16,
                        step=1,
                        label="Number of molecules",
                        scale=1,
                    )

                with gr.Tabs():
                    with gr.Tab("Generate"):
                        run_btn = gr.Button("⚛  Generate molecules", variant="primary", size="lg")
                        mol_out = gr.HTML()
                        metrics_out = gr.HTML(_metrics_html(), elem_id="metrics-html")
                        notes_out = gr.HTML()

                    with gr.Tab("Compare backends"):
                        cmp_panel = gr.Column(elem_id="cmp-panel")
                        with cmp_panel:
                            cmp_intro = gr.HTML(_compare_intro_html(True))
                            cmp_btn = gr.Button(
                                "Compare backends",
                                variant="primary",
                                size="lg",
                                interactive=True,
                            )
                            with gr.Row():
                                cmp_ideal = gr.HTML(label="Ideal")
                                cmp_bris = gr.HTML(label="Brisbane")
                            with gr.Row():
                                cmp_sher = gr.HTML(label="Sherbrooke")
                                cmp_osaka = gr.HTML(label="Osaka")
                            cmp_notes = gr.HTML()

                    with gr.Tab("Paper results"):
                        gr.HTML(_PAPER_HTML)

        # ── wire events ──────────────────────────────────────────────────────
        _ch_in = [gen_type_radio, gen_arch_radio, cycle_radio, backend_radio, shots_radio]
        _ch_out = [
            circuit_plot,
            gen_arch_radio,
            gen_active,
            cycle_active,
            model_info_out,
            cmp_btn,
            cmp_intro,
            cmp_panel,
        ]

        gen_type_radio.change(on_change, inputs=_ch_in, outputs=_ch_out)
        gen_arch_radio.change(on_change, inputs=_ch_in, outputs=_ch_out)
        cycle_radio.change(on_change, inputs=_ch_in, outputs=_ch_out)

        _bk_in = [backend_radio, shots_radio, gen_type_radio, gen_arch_radio, cycle_radio]
        _bk_out = [chips_out, spec_active, model_info_out, shots_radio, shots_row]
        backend_radio.change(on_backend_change, inputs=_bk_in, outputs=_bk_out)
        shots_radio.change(on_backend_change, inputs=_bk_in, outputs=_bk_out)

        demo.load(on_change, inputs=_ch_in, outputs=_ch_out)
        demo.load(on_backend_change, inputs=_bk_in, outputs=_bk_out)

        run_btn.click(
            run_generation,
            inputs=[
                gen_type_radio,
                gen_arch_radio,
                cycle_radio,
                seed_mode_dd,
                backend_radio,
                shots_radio,
                n_slider,
                seed_num,
            ],
            outputs=[mol_out, metrics_out, notes_out],
        )
        cmp_btn.click(
            compare_backends,
            inputs=[
                gen_type_radio,
                gen_arch_radio,
                cycle_radio,
                seed_mode_dd,
                shots_radio,
                n_slider,
                seed_num,
            ],
            outputs=[cmp_ideal, cmp_bris, cmp_sher, cmp_osaka, cmp_notes],
        )

    return demo


if __name__ == "__main__":
    build_ui().launch(theme=_theme(), css=_CSS)
