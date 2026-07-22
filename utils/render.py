"""Molecule rendering helpers."""

from __future__ import annotations

from typing import Optional

_CARD_BG = "#ffffff"
_GRID_BG = "#f8fafc"
_BORDER = "#e5e7eb"
_TEXT_MUTED = "#6b7280"
_TEXT_INK2 = "#374151"


def _mol_to_svg(mol, size: tuple[int, int] = (300, 300)) -> str:
    """Render a single RDKit Mol to an SVG string (transparent background)."""
    import re

    from rdkit.Chem.Draw import rdMolDraw2D

    w, h = size
    drawer = rdMolDraw2D.MolDraw2DSVG(w, h)
    opts = drawer.drawOptions()
    opts.clearBackground = False
    opts.addStereoAnnotation = True
    rdMolDraw2D.PrepareAndDrawMolecule(drawer, mol)
    drawer.FinishDrawing()
    svg = drawer.GetDrawingText()

    # strip XML declaration
    svg = svg[svg.find("<svg") :]

    # replace fixed px dimensions with 100%/100% while keeping viewBox
    svg = re.sub(r"width='[^']*'", "width='100%'", svg, count=1)
    svg = re.sub(r"height='[^']*'", "height='100%'", svg, count=1)

    # ensure viewBox exists (RDKit always adds it, but be safe)
    if "viewBox" not in svg:
        svg = svg.replace("<svg ", f"<svg viewBox='0 0 {w} {h}' ", 1)

    return svg


def smiles_grid_html(
    smiles: list[str],
    legends: Optional[list[str]] = None,
    cols: int = 3,
    card_size: int = 300,  # SVG render resolution (internal px, not layout px)
) -> str:
    """Return an HTML string of a responsive grid of molecule SVG cards."""
    from rdkit import Chem

    mols, kept_legends = [], []
    for i, smi in enumerate(smiles):
        mol = Chem.MolFromSmiles(smi) if smi else None
        if mol is None:
            continue
        mols.append(mol)
        if legends is not None:
            kept_legends.append(legends[i] if i < len(legends) else "")

    if not mols:
        return _empty_html("No valid molecules in this batch")

    cards_html = []
    for i, mol in enumerate(mols):
        svg = _mol_to_svg(mol, size=(card_size, card_size))
        legend = kept_legends[i] if kept_legends else ""
        legend_block = (
            f'<div style="'
            f"font-size:11px;font-weight:600;text-align:center;"
            f"color:{_TEXT_INK2};font-family:IBM Plex Mono,monospace;"
            f"margin-top:6px;letter-spacing:.02em;white-space:nowrap"
            f'">{legend}</div>'
            if legend
            else ""
        )
        cards_html.append(
            # card — min-width:0 prevents grid blowout
            f'<div style="'
            f"min-width:0;box-sizing:border-box;"
            f"background:{_CARD_BG};border:1px solid {_BORDER};border-radius:12px;"
            f"padding:10px;display:flex;flex-direction:column;align-items:center;"
            f'">'
            # SVG wrapper: square aspect-ratio, fills card width
            f'  <div style="width:100%;aspect-ratio:1/1;display:flex;align-items:center;justify-content:center;">'
            f"    {svg}"
            f"  </div>"
            f"  {legend_block}"
            f"</div>"
        )

    card_max_px = 220  # max width per molecule card in px
    grid_max_px = cols * card_max_px

    grid = (
        # outer wrapper — capped width so a single mol doesn't stretch full screen
        f'<div style="width:100%;box-sizing:border-box;'
        f'background:{_GRID_BG};border-radius:10px;padding:12px;">'
        f'<div style="'
        f"display:grid;"
        f"grid-template-columns:repeat({cols},minmax(0,1fr));"
        f"gap:10px;"
        f"max-width:{grid_max_px}px;"
        f"box-sizing:border-box;"
        f'">' + "".join(cards_html) + "</div></div>"
    )
    return grid


def _empty_html(msg: str) -> str:
    return (
        f'<div style="'
        f"padding:2rem;text-align:center;"
        f"color:{_TEXT_MUTED};font-size:13px;"
        f"background:{_GRID_BG};border-radius:10px;"
        f'">{msg}</div>'
    )
