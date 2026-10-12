"""OA-geometry overlap scan: count STRICT body-body overlaps and wires-under-bodies.

Operates on a geometry dict read back from a SAVED OpenAccess design (the same
readback ``examples/real_parts/_render.read_geometry`` produces: per-instance
axis-aligned bbox + pin centres, and wire polylines). It is a pure measurement on
emitted geometry -- no ADS, no emitter change.

Two failure classes ("strict" = interior, not edge-touch):

1. body-body overlap  -- two instance bounding boxes share positive interior area
   (both overlap width and height exceed ``lin_eps``).  GROUND symbols are placed
   coincident with the pin they ground (sources/vias) BY DESIGN, so a pair that
   involves a GROUND is reported separately and excluded from the strict count.

2. wire-under-body     -- a wire segment passes through the INTERIOR of an instance
   body.  The segment/body intersection is measured directly; a wire that merely
   lands on a boundary pin has zero interior length and does not count.  This also
   catches the subtle case where a wire starts/ends at a pin but then runs along the
   body axis through the symbol interior.

``strict_total`` = non-ground body-body overlaps + wire-under-body hits.  The
generalization test expects ``strict_total == 0``.
"""
from __future__ import annotations

GROUND_SHORTS = {"GROUND", "gnd", "GND"}


def _rect(b):
    x1, y1, x2, y2 = b
    return (min(x1, x2), min(y1, y2), max(x1, x2), max(y1, y2))


def _bbox_overlap(a, b, *, lin_eps):
    ax1, ay1, ax2, ay2 = _rect(a)
    bx1, by1, bx2, by2 = _rect(b)
    ox = min(ax2, bx2) - max(ax1, bx1)
    oy = min(ay2, by2) - max(ay1, by1)
    if ox > lin_eps and oy > lin_eps:
        return {"overlap_w": ox, "overlap_h": oy, "area": ox * oy}
    return None


def _pt_in_box(p, b, *, pad=0.0):
    x1, y1, x2, y2 = _rect(b)
    return (x1 - pad) <= p[0] <= (x2 + pad) and (y1 - pad) <= p[1] <= (y2 + pad)


def _seg_box_interior_len(p0, p1, box, *, shrink, eps=1.0e-9):
    """Length of segment p0->p1 inside ``box`` interior (Liang-Barsky)."""
    x1, y1, x2, y2 = _rect(box)
    xmin, xmax = x1 + shrink + eps, x2 - shrink - eps
    ymin, ymax = y1 + shrink + eps, y2 - shrink - eps
    if xmin >= xmax or ymin >= ymax:
        return 0.0  # degenerate after shrink
    dx = p1[0] - p0[0]
    dy = p1[1] - p0[1]
    t0, t1 = 0.0, 1.0
    for p, q in ((-dx, p0[0] - xmin), (dx, xmax - p0[0]),
                 (-dy, p0[1] - ymin), (dy, ymax - p0[1])):
        if abs(p) < 1e-12:
            if q < 0:
                return 0.0  # parallel and outside
            continue
        r = q / p
        if p < 0:
            if r > t1:
                return 0.0
            if r > t0:
                t0 = r
        else:
            if r < t0:
                return 0.0
            if r < t1:
                t1 = r
    if t1 <= t0:
        return 0.0
    seg_len = (dx * dx + dy * dy) ** 0.5
    return seg_len * (t1 - t0)


def scan(geo, *, lin_eps=0.05, shrink=0.0, min_wire_pen=0.30):
    """Return a strict-overlap report for a geometry dict."""
    insts = geo["insts"]
    body_overlaps = []          # non-ground body-body interior overlaps
    ground_overlaps = []        # overlaps that involve a GROUND symbol (expected)
    for i in range(len(insts)):
        for j in range(i + 1, len(insts)):
            a, b = insts[i], insts[j]
            ov = _bbox_overlap(a["bbox"], b["bbox"], lin_eps=lin_eps)
            if ov is None:
                continue
            rec = {"a": a["name"], "a_kind": a["short"], "b": b["name"],
                   "b_kind": b["short"], **{k: round(v, 3) for k, v in ov.items()}}
            if a["short"] in GROUND_SHORTS or b["short"] in GROUND_SHORTS:
                ground_overlaps.append(rec)
            else:
                body_overlaps.append(rec)

    wire_hits = []
    for wi, wire in enumerate(geo["wires"]):
        for si in range(len(wire) - 1):
            p0, p1 = wire[si], wire[si + 1]
            for inst in insts:
                if inst["short"] in GROUND_SHORTS:
                    continue
                bb = inst["bbox"]
                pen = _seg_box_interior_len(p0, p1, bb, shrink=shrink)
                if pen > min_wire_pen:
                    wire_hits.append({"wire": wi, "seg": si, "inst": inst["name"],
                                      "kind": inst["short"], "pen_len": round(pen, 3)})

    strict_total = len(body_overlaps) + len(wire_hits)
    return {
        "n_instances": len(insts),
        "n_wires": len(geo["wires"]),
        "body_overlaps_strict": body_overlaps,
        "ground_coincident_overlaps": ground_overlaps,
        "wires_under_bodies": wire_hits,
        "strict_total": strict_total,
        "params": {"lin_eps": lin_eps, "shrink": shrink, "min_wire_pen": min_wire_pen},
    }


def summarize(rep, label=""):
    head = f"  [{label}] " if label else "  "
    lines = [
        f"{head}instances={rep['n_instances']} wires={rep['n_wires']}  "
        f">>> STRICT OVERLAPS = {rep['strict_total']}",
        f"{head}  body-body interior overlaps (non-ground): {len(rep['body_overlaps_strict'])}",
    ]
    for r in rep["body_overlaps_strict"][:12]:
        lines.append(f"{head}    {r['a']}({r['a_kind']}) x {r['b']}({r['b_kind']})  "
                     f"w={r['overlap_w']} h={r['overlap_h']} area={r['area']}")
    lines.append(f"{head}  wires under element bodies: {len(rep['wires_under_bodies'])}")
    for r in rep["wires_under_bodies"][:12]:
        lines.append(f"{head}    wire#{r['wire']} seg{r['seg']} under {r['inst']}"
                     f"({r['kind']})  pen={r['pen_len']}")
    lines.append(f"{head}  (ground-coincident overlaps, expected by design: "
                 f"{len(rep['ground_coincident_overlaps'])})")
    return "\n".join(lines)
