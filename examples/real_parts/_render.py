"""Render a saved OpenAccess schematic to PNG (faithful to the emitted geometry).

Reads instances / pins / wires / terminals back from a saved ADS OpenAccess
design and draws them with matplotlib, so the screenshot shows what ADS actually
stored -- not a synthetic mock. Used by ``build_examples.py shots`` to capture the
hierarchical unit-cell and top-level cascade drawings.

Requires Keysight ADS Python (``keysight.ads.de``) and matplotlib.
"""
from __future__ import annotations

from pathlib import Path

__all__ = ["read_geometry", "draw"]


def _outline_pts(line):
    o = line.get_outline()
    for meth in ("points", "get_points", "vertices", "get_vertices"):
        h = getattr(o, meth, None)
        if h is None:
            continue
        try:
            seq = h() if callable(h) else h
            return [(float(p.x), float(p.y)) for p in seq]
        except Exception:  # noqa: BLE001
            continue
    bb = line.bbox
    return [(bb.x1, bb.y1), (bb.x2, bb.y2)]


def read_geometry(ws_path, design_name):
    """Read instances/pins/wires/terminals from a saved OA design."""
    import keysight.ads.de as de
    from keysight.ads.de import db_uu as db

    if de.workspace_is_open():
        de.close_workspace()
    de.open_workspace(str(ws_path))
    d = db.open_design(design_name, mode=db.DesignMode.READ_ONLY)
    insts = []
    for inst in d.instances:
        bb = inst.bbox
        master = str(getattr(inst, "master_cell", "") or "")
        short = master.split(":")[-1].rstrip('">') if master else ""
        pins = []
        for p in inst.inst_pins:
            pb = getattr(p, "bbox", None)
            if pb is not None:
                pins.append(((pb.x1 + pb.x2) / 2.0, (pb.y1 + pb.y2) / 2.0))
        insts.append({"name": inst.name, "short": short,
                      "bbox": (bb.x1, bb.y1, bb.x2, bb.y2), "pins": pins})
    wires, dots = [], []
    for s in d.shapes:
        nm = type(s).__name__
        if nm == "Line":
            wires.append(_outline_pts(s))
        elif nm == "Dot":
            bb = s.bbox
            dots.append(((bb.x1 + bb.x2) / 2.0, (bb.y1 + bb.y2) / 2.0))
    terms = [(t.name, int(t.number)) for t in d.terms]
    if de.workspace_is_open():
        de.close_workspace()
    return {"insts": insts, "wires": wires, "dots": dots, "terms": terms}


def draw(geo, out_png, title, *, label_terms=True):
    """Draw a geometry dict (from :func:`read_geometry`) to a PNG."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle

    xs, ys = [], []
    for it in geo["insts"]:
        x1, y1, x2, y2 = it["bbox"]
        xs += [x1, x2]; ys += [y1, y2]
    for w in geo["wires"]:
        for (x, y) in w:
            xs.append(x); ys.append(y)
    for (x, y) in geo["dots"]:
        xs.append(x); ys.append(y)
    if not xs:
        xs, ys = [0, 1], [0, 1]
    pad = 2.0
    xlo, xhi = min(xs) - pad, max(xs) + pad
    ylo, yhi = min(ys) - pad, max(ys) + pad
    w_in = max(7.0, min(26.0, (xhi - xlo) * 0.32))
    h_in = max(4.0, min(26.0, (yhi - ylo) * 0.32))
    fig, ax = plt.subplots(figsize=(w_in, h_in))

    for w in geo["wires"]:
        if len(w) >= 2:
            ax.plot([p[0] for p in w], [p[1] for p in w], color="#3a6ea5", lw=1.1, zorder=1)
    for it in geo["insts"]:
        x1, y1, x2, y2 = it["bbox"]
        bw, bh = x2 - x1, y2 - y1
        px = max(0.45, bw * 0.15); py = max(0.45, bh * 0.15)
        cx, cy = (x1 + x2) / 2.0, (y1 + y2) / 2.0
        # GROUND symbols are placed coincident with the pin they ground (e.g. each
        # source return), so a source and its ground sit at one spot. Drawing the
        # ground's auto instance-name + cell text there stacks two labels on top of
        # the source's labels (the SMV2020 left-edge tangle). A ground is self-evident
        # from its symbol, so render it as a small unlabelled marker -- the source's
        # own labels then read cleanly. (Presentation only; geometry is unchanged.)
        if it["short"] == "GROUND":
            ax.add_patch(Rectangle((x1 - 0.15, y1 - 0.15), bw + 0.3, bh + 0.3,
                         fc="#d7e0ea", ec="#6b7d90", lw=0.8, zorder=2))
            for (pxc, pyc) in it["pins"]:
                ax.plot([pxc], [pyc], marker="o", ms=2.0, color="#6b7d90", zorder=3)
            continue
        ax.add_patch(Rectangle((x1 - px, y1 - py), bw + 2 * px, bh + 2 * py,
                     fc="#eef3fb", ec="#22405e", lw=1.1, zorder=2))
        ax.text(cx, y2 + py + 0.5, it["name"], ha="center", va="bottom", fontsize=6.0,
                zorder=4, color="#11324e")
        if it["short"]:
            ax.text(cx, cy, it["short"], ha="center", va="center", fontsize=5.0,
                    zorder=4, color="#557")
        for (pxc, pyc) in it["pins"]:
            ax.plot([pxc], [pyc], marker="o", ms=2.2, color="#c0392b", zorder=3)
    dots = sorted(geo["dots"])
    names = {1: "IN", 2: "OUT"}
    term_by_num = {n: nm for (nm, n) in geo["terms"]}
    for i, (x, y) in enumerate(dots):
        ax.plot([x], [y], marker=">", ms=9, color="#138d5a", zorder=5)
        if label_terms:
            nm = (term_by_num.get(1) if i == 0 else term_by_num.get(2)) or names.get(i + 1, "T")
            ax.text(x, y + 1.0, nm, ha="center", va="bottom", fontsize=9,
                    fontweight="bold", color="#0d5c3a", zorder=6)

    ax.set_xlim(xlo, xhi); ax.set_ylim(ylo, yhi)
    ax.set_aspect("equal", adjustable="box")
    ax.set_title(title, fontsize=10, fontweight="bold")
    ax.axis("off")
    fig.tight_layout()
    Path(out_png).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=140)
    plt.close(fig)
