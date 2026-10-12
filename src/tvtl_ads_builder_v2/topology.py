"""Topology SVG renderer for a v2 ``DeviceSpec``.

Renders a schematic-style topology graph straight from a device spec's
element + net structure -- a second, ADS-independent view of a circuit,
useful because the auto-routed ADS schematic can obscure the actual
through-path. Pure Python and the standard library only: no Keysight
API, no third-party dependency, no ADS installation required.

Layout
------
Component placement is *not* computed here. It is delegated to
:mod:`tvtl_ads_builder_v2.layout`, the shared source of truth that the
OA-Python emitter also consumes -- so the topology drawing and the ADS
schematic agree on where every component sits. ``layout`` works in
dimensionless *layout units*; this module scales them to SVG pixels:

* Through-path components are placed horizontally, left to right.
* Shunt branches hang vertically below their tap point.
* Ground symbols are drawn at every net in ``spec.ground_nodes``.
* Open-circuit symbols are drawn at every net in
  ``spec.open_terminations``.
* A topology that does not fit the linear / one-shunt model falls back
  to a simple grid layout (``layout`` logs the warning).
* Entity types the renderer does not recognise are drawn as plain
  boxes and a warning is logged; they never crash the render.

Visual conventions
------------------
* Component boxes: rounded rectangles labelled with the instance name,
  the type string, and one or two identifying parameter values.
* Multi-port components (``MTEE``) are drawn with numbered port labels.
* ``P_1Tone`` sources: rounded boxes in a distinct green fill, placed
  beside the net each drives (left of the input net, etc.), labelled
  with the source name and its tone (``f=`` / ``P=``), and joined to
  the target net by a short wire.
* Wires: black 1px orthogonal polylines. A junction dot marks any net
  with three or more connections.
* Ground: a stem with three shrinking bars. Open termination: a short
  perpendicular bar at the end of the line.

The only public entry point is :func:`render_topology_svg`.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from xml.sax.saxutils import escape

from .device_spec import DeviceSpec, Element
from .layout import CircuitLayout, build_layout

_LOG = logging.getLogger(__name__)

# Element types the renderer lays out and labels knowingly. Anything
# else is still drawn (as a plain box) but triggers a warning.
# ``NAME`` annotations are skipped by the renderer entirely (filtered
# in :func:`build_layout`), so adding NAME here is purely to suppress
# the "unrecognised element type" warning on probed cascades.
_KNOWN_TYPES = frozenset(
    {
        "TermG", "MLIN", "MTEE", "R", "L", "C", "RLC_SERIES", "SDD2P",
        "I_Probe", "NAME",
        # Emitter-accuracy additions: the distributed termination elements
        # (1-port). VIAGND = physical via-to-ground; MLEF = open-end-effect stub
        # (with fringe); MLOC = open-circuited stub (ideal open).
        "VIAGND", "MLEF", "MLOC",
    }
)

# Layout geometry (SVG user units == pixels).
_BOX_W = 116.0
_BOX_H = 50.0
_DX = 158.0          # horizontal step between through-path box centres
_DY = 140.0          # vertical step down a shunt branch
_PAD = 34.0          # canvas margin
_TITLE_H = 26.0      # space reserved at the top for the circuit title

# Layout-unit -> pixel scaling. One layout x-unit is one column step
# ``_DX``; one layout y-unit is one shunt-row step ``_DY``. The origin
# ``(_X0, _Y0)`` is the pixel centre of the layout-unit (0, 0) cell.
_X0 = _PAD + _BOX_W / 2.0
_Y0 = _TITLE_H + _PAD + _BOX_H / 2.0

# Pin-side token -> (dx, dy) pixel offset from a component box centre.
_PIN_OFFSET: dict[str, tuple[float, float]] = {
    "left": (-_BOX_W / 2.0, 0.0),
    "right": (_BOX_W / 2.0, 0.0),
    "top": (0.0, -_BOX_H / 2.0),
    "bottom": (0.0, _BOX_H / 2.0),
}


@dataclass
class _Placed:
    """One element placed on the canvas, in SVG pixels."""

    element: Element
    role: str                              # "through" | "shunt" | "port" | "grid"
    cx: float
    cy: float
    pin_xy: dict[int, tuple[float, float]] = field(default_factory=dict)


@dataclass
class _Symbol:
    """A ground or open-termination marker anchored to a net, in pixels."""

    kind: str                              # "ground" | "open"
    net: str
    x: float                               # connection point (top of symbol)
    y: float


@dataclass
class _PlacedSource:
    """A ``P_1Tone`` power source placed on the canvas, in SVG pixels.

    ``name`` is the source instance name; ``target_net`` the net it
    drives; ``cx`` / ``cy`` the box centre. The wire to the target net
    is drawn by the shared net-wiring pass -- the source's connection
    pin is added to ``target_net`` as just another point on that net.
    """

    name: str
    target_net: str
    cx: float
    cy: float


@dataclass
class _Layout:
    placements: list[_Placed] = field(default_factory=list)
    symbols: list[_Symbol] = field(default_factory=list)
    sources: list[_PlacedSource] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------


def render_topology_svg(spec: DeviceSpec, out_path: str | Path) -> None:
    """Render ``spec`` as a standalone SVG topology graph at ``out_path``.

    Never raises on a drawable spec: an unclassifiable topology falls
    back to a grid layout, and an unrecognised element type is drawn as
    a plain box. Both cases log a warning.
    """

    for element in spec.elements:
        if element.type not in _KNOWN_TYPES:
            _LOG.warning(
                "topology renderer: unrecognised element type %r on %r in "
                "circuit %r -- drawn as a generic box",
                element.type,
                element.id,
                spec.label,
            )

    pixel_layout = _to_pixel_layout(build_layout(spec))
    svg = _render_svg(spec, pixel_layout)
    Path(out_path).write_text(svg, encoding="utf-8")


# ---------------------------------------------------------------------------
# Layout-unit -> pixel scaling
# ---------------------------------------------------------------------------


def _scale_x(x: float) -> float:
    """Layout-unit x -> SVG pixel x (a component-centre coordinate)."""

    return _X0 + x * _DX


def _scale_y(y: float) -> float:
    """Layout-unit y -> SVG pixel y (a component-centre coordinate)."""

    return _Y0 + y * _DY


def _to_pixel_layout(layout: CircuitLayout) -> _Layout:
    """Scale a layout-unit :class:`CircuitLayout` to a pixel-space ``_Layout``.

    Component centres scale uniformly by the column/row pitch; pin
    coordinates are then the scaled centre plus a fixed box-edge offset,
    and a termination symbol's connection point is the bottom edge of
    the box it hangs from (or the through-row coordinate directly).

    ``P_1Tone`` sources sit left of the through path, at negative layout
    x. Every coordinate is shifted right by ``x_shift`` so the whole
    drawing stays on the positive canvas; the shift is zero for a
    source-free spec, so existing source-free renders are unchanged.
    """

    x_shift = -min(
        [comp.x for comp in layout.placements]
        + [src.x for src in layout.sources]
        + [0.0]
    )

    placements: list[_Placed] = []
    for comp in layout.placements:
        cx = _scale_x(comp.x + x_shift)
        cy = _scale_y(comp.y)
        pin_xy: dict[int, tuple[float, float]] = {}
        for pin_number, side in comp.pin_sides.items():
            dx, dy = _PIN_OFFSET[side]
            pin_xy[pin_number] = (cx + dx, cy + dy)
        placements.append(_Placed(comp.element, comp.role, cx, cy, pin_xy))

    sources: list[_PlacedSource] = []
    for src in layout.sources:
        sources.append(
            _PlacedSource(
                src.name,
                src.target_net,
                _scale_x(src.x + x_shift),
                _scale_y(src.y),
            )
        )

    symbols: list[_Symbol] = []
    for sym in layout.symbols:
        sx = _scale_x(sym.x + x_shift)
        sy = _scale_y(sym.y)
        if sym.attach == "below_box":
            sy += _BOX_H / 2.0
        symbols.append(_Symbol(sym.kind, sym.net, sx, sy))

    return _Layout(placements=placements, symbols=symbols, sources=sources)


# ---------------------------------------------------------------------------
# SVG rendering
# ---------------------------------------------------------------------------


def _n(value: float) -> str:
    """Format a coordinate compactly."""

    return f"{value:.1f}"


def _key_params(element: Element) -> str:
    """A short, identifying parameter string for a component box."""

    p = element.parameters
    try:
        if element.type == "MLIN":
            return f"w={_si(p['w_m'], 1e3)}mm l={_si(p['l_m'], 1e3)}mm"
        if element.type == "MTEE":
            return (
                f"W1/W2/W3={_si(p['w1_m'], 1e3)}/{_si(p['w2_m'], 1e3)}/"
                f"{_si(p['w3_m'], 1e3)}mm"
            )
        if element.type == "R":
            return f"R={_si(p['r_ohm'], 1.0)} Ohm"
        if element.type == "L":
            return f"L={_si(p['l_h'], 1e9)} nH"
        if element.type == "C":
            return f"C={_si(p['c_f'], 1e12)} pF"
        if element.type == "RLC_SERIES":
            return (
                f"R={_si(p['r_ohm'], 1.0)} Ohm  L={_si(p['l_h'], 1e9)} nH  "
                f"C={_si(p['c_f'], 1e12)} pF"
            )
        if element.type == "SDD2P":
            reference = p.get("q_poly_reference")
            if reference:
                return f"Q(v) ref={reference}"
            degree = max(0, len(tuple(p.get("q_poly_coefficients", ()))) - 1)
            return f"Q(v) charge poly deg={degree}"
        if element.type == "TermG":
            return f"Z={_si(p['z_ohm'], 1.0)} Ohm"
        if element.type == "VIAGND":
            return f"via D={_si(p['d_m'], 1e3)}mm (to gnd)"
        if element.type in ("MLEF", "MLOC"):
            kind = "open-end fringe" if element.type == "MLEF" else "open stub"
            return f"{kind} w={_si(p['w_m'], 1e3)}mm"
    except (KeyError, TypeError, ValueError):
        return ""
    return ""


def _si(value, scale: float) -> str:
    number = float(value) * scale
    text = f"{number:.4g}"
    return text


def _bbox(layout: _Layout) -> tuple[float, float]:
    """Canvas (width, height) covering every placement and symbol."""

    xs = [_PAD]
    ys = [_TITLE_H]
    for placed in layout.placements:
        xs += [placed.cx - _BOX_W / 2.0, placed.cx + _BOX_W / 2.0]
        ys += [placed.cy - _BOX_H / 2.0, placed.cy + _BOX_H / 2.0]
    for source in layout.sources:
        xs += [source.cx - _BOX_W / 2.0, source.cx + _BOX_W / 2.0]
        ys += [source.cy - _BOX_H / 2.0, source.cy + _BOX_H / 2.0]
    for symbol in layout.symbols:
        xs += [symbol.x - 26.0, symbol.x + 26.0]
        ys += [symbol.y, symbol.y + 64.0]
    width = max(xs) + _PAD
    height = max(ys) + _PAD
    return max(width, 320.0), max(height, 200.0)


def _render_svg(spec: DeviceSpec, layout: _Layout) -> str:
    width, height = _bbox(layout)
    parts: list[str] = []
    parts.append('<?xml version="1.0" encoding="UTF-8"?>')
    parts.append(
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{_n(width)}" '
        f'height="{_n(height)}" viewBox="0 0 {_n(width)} {_n(height)}" '
        f'font-family="Consolas, monospace">'
    )
    parts.append(
        f'<rect class="background" x="0" y="0" width="{_n(width)}" '
        f'height="{_n(height)}" fill="#ffffff"/>'
    )
    parts.append(
        f'<text class="title" x="{_n(width / 2.0)}" y="18" '
        f'text-anchor="middle" font-size="13" font-weight="bold">'
        f'{escape(spec.label or "device_spec")} -- topology</text>'
    )

    # Net -> connection points, gathered from placements and symbols.
    net_points: dict[str, list[tuple[float, float]]] = {}
    for placed in layout.placements:
        for pin in placed.element.pins:
            xy = placed.pin_xy.get(pin.pin)
            if xy is not None:
                net_points.setdefault(pin.node, []).append(xy)
    for symbol in layout.symbols:
        net_points.setdefault(symbol.net, []).append((symbol.x, symbol.y))

    # P_1Tone sources: each source's connection pin joins its target
    # net, so the shared wire pass below draws a short wire from the
    # source to the net. The pin is the source box edge facing the net.
    for source in layout.sources:
        target = net_points.get(source.target_net)
        toward = target[0] if target else (source.cx, source.cy)
        net_points.setdefault(source.target_net, []).append(
            _source_pin(source, toward)
        )

    # Wires: connect every point on a net to the net's first point.
    for net, points in sorted(net_points.items()):
        if len(points) < 2:
            continue
        anchor = points[0]
        for point in points[1:]:
            parts.append(_wire(anchor, point))

    # Junction dots at nets with three or more connections.
    for net, points in sorted(net_points.items()):
        if len(points) >= 3:
            jx, jy = points[0]
            parts.append(
                f'<circle class="junction" cx="{_n(jx)}" cy="{_n(jy)}" '
                f'r="3.4" fill="#000000"/>'
            )

    # Ground / open symbols.
    for symbol in layout.symbols:
        if symbol.kind == "ground":
            parts.append(_ground_symbol(symbol))
        else:
            parts.append(_open_symbol(symbol))

    # Component / port boxes (drawn on top of the wires).
    for placed in layout.placements:
        parts.append(_box(placed))

    # P_1Tone source boxes (also on top of the wires).
    spec_sources = {source.name: source for source in spec.sources}
    for source in layout.sources:
        parts.append(_source_box(source, spec_sources.get(source.name)))

    parts.append("</svg>")
    parts.append("")
    return "\n".join(parts)


def _wire(a: tuple[float, float], b: tuple[float, float]) -> str:
    """An orthogonal (horizontal-then-vertical) wire from ``a`` to ``b``."""

    ax, ay = a
    bx, by = b
    points = f"{_n(ax)},{_n(ay)} {_n(bx)},{_n(ay)} {_n(bx)},{_n(by)}"
    return (
        f'<polyline class="wire" points="{points}" fill="none" '
        f'stroke="#000000" stroke-width="1"/>'
    )


def _box(placed: _Placed) -> str:
    element = placed.element
    cx, cy = placed.cx, placed.cy
    x = cx - _BOX_W / 2.0
    y = cy - _BOX_H / 2.0
    is_port = placed.role == "port"
    fill = "#fbe9d7" if is_port else "#e9f1fb"
    stroke = "#8a4b16" if is_port else "#1b3a5b"
    css = "port" if is_port else "component"

    name = escape(element.id)
    type_text = escape(element.type)
    params = escape(_key_params(element))

    lines = [
        f'<g class="{css}" data-name="{escape(element.id)}" '
        f'data-type="{escape(element.type)}" data-role="{escape(placed.role)}" '
        f'data-x="{_n(cx)}" data-y="{_n(cy)}">',
        f'<rect x="{_n(x)}" y="{_n(y)}" width="{_n(_BOX_W)}" '
        f'height="{_n(_BOX_H)}" rx="7" ry="7" fill="{fill}" '
        f'stroke="{stroke}" stroke-width="1.5"/>',
        f'<text x="{_n(cx)}" y="{_n(cy - 8.0)}" text-anchor="middle" '
        f'font-size="12" font-weight="bold">{name}</text>',
        f'<text x="{_n(cx)}" y="{_n(cy + 4.0)}" text-anchor="middle" '
        f'font-size="9.5" fill="#555555">{type_text}</text>',
    ]
    if params:
        lines.append(
            f'<text x="{_n(cx)}" y="{_n(cy + 16.0)}" text-anchor="middle" '
            f'font-size="8.5" fill="#333333">{params}</text>'
        )
    # Numbered pin labels for a multi-port component (e.g. MTEE).
    if len(placed.pin_xy) >= 3:
        for pin_number, (px, py) in sorted(placed.pin_xy.items()):
            lx = px + (7.0 if px > cx else (-7.0 if px < cx else 0.0))
            ly = py + (14.0 if py > cy else -5.0)
            lines.append(
                f'<text class="pin-label" x="{_n(lx)}" y="{_n(ly)}" '
                f'text-anchor="middle" font-size="8" fill="#1b3a5b">'
                f'{pin_number}</text>'
            )
    lines.append("</g>")
    return "".join(lines)


def _source_pin(
    source: _PlacedSource, toward: tuple[float, float]
) -> tuple[float, float]:
    """The source box edge facing ``toward`` -- its wire connection point.

    The pin is on whichever box edge points at the target net, so the
    shared wire pass draws a short, axis-aligned segment.
    """

    cx, cy = source.cx, source.cy
    tx, ty = toward
    if abs(tx - cx) >= abs(ty - cy):
        return (cx + _BOX_W / 2.0, cy) if tx >= cx else (cx - _BOX_W / 2.0, cy)
    return (cx, cy + _BOX_H / 2.0) if ty >= cy else (cx, cy - _BOX_H / 2.0)


def _source_params(spec_source) -> str:
    """A short ``f=.. P=..`` label for a P_1Tone source box.

    ``spec_source`` is the spec's :class:`P1ToneSource` (``None`` if it
    could not be matched by name). ``freq`` / ``power_dBm`` are usually
    resolved literals by render time; a still-symbolic VAR-block name
    is shown verbatim.
    """

    if spec_source is None:
        return ""

    def _fmt(value, scale: float, unit: str) -> str:
        if isinstance(value, str):
            return value
        try:
            return f"{float(value) * scale:.4g} {unit}"
        except (TypeError, ValueError):
            return "?"

    return (
        f"f={_fmt(spec_source.freq, 1e-9, 'GHz')} "
        f"P={_fmt(spec_source.power_dBm, 1.0, 'dBm')}"
    )


def _source_box(source: _PlacedSource, spec_source) -> str:
    """Draw one P_1Tone source as a labelled box (name, ``P_1Tone``, tone).

    Visually parallel to :func:`_box` -- a rounded rectangle -- but in a
    distinct green fill so a source reads apart from a routed component
    or a port.
    """

    cx, cy = source.cx, source.cy
    x = cx - _BOX_W / 2.0
    y = cy - _BOX_H / 2.0
    params = escape(_source_params(spec_source))
    lines = [
        f'<g class="source" data-name="{escape(source.name)}" '
        f'data-net="{escape(source.target_net)}" '
        f'data-x="{_n(cx)}" data-y="{_n(cy)}">',
        f'<rect x="{_n(x)}" y="{_n(y)}" width="{_n(_BOX_W)}" '
        f'height="{_n(_BOX_H)}" rx="7" ry="7" fill="#e7f5e9" '
        f'stroke="#2f6b3a" stroke-width="1.5"/>',
        f'<text x="{_n(cx)}" y="{_n(cy - 8.0)}" text-anchor="middle" '
        f'font-size="12" font-weight="bold">{escape(source.name)}</text>',
        f'<text x="{_n(cx)}" y="{_n(cy + 4.0)}" text-anchor="middle" '
        f'font-size="9.5" fill="#555555">P_1Tone</text>',
    ]
    if params:
        lines.append(
            f'<text x="{_n(cx)}" y="{_n(cy + 16.0)}" text-anchor="middle" '
            f'font-size="8.5" fill="#333333">{params}</text>'
        )
    lines.append("</g>")
    return "".join(lines)


def _ground_symbol(symbol: _Symbol) -> str:
    x, y = symbol.x, symbol.y
    stem = 18.0
    lines = [
        f'<g class="ground" data-net="{escape(symbol.net)}">',
        f'<line x1="{_n(x)}" y1="{_n(y)}" x2="{_n(x)}" y2="{_n(y + stem)}" '
        f'stroke="#000000" stroke-width="1"/>',
    ]
    for index, half in enumerate((11.0, 7.0, 3.5)):
        by = y + stem + index * 4.0
        lines.append(
            f'<line x1="{_n(x - half)}" y1="{_n(by)}" x2="{_n(x + half)}" '
            f'y2="{_n(by)}" stroke="#000000" stroke-width="1"/>'
        )
    lines.append("</g>")
    return "".join(lines)


def _open_symbol(symbol: _Symbol) -> str:
    x, y = symbol.x, symbol.y
    stem = 16.0
    half = 11.0
    return (
        f'<g class="open-termination" data-net="{escape(symbol.net)}">'
        f'<line x1="{_n(x)}" y1="{_n(y)}" x2="{_n(x)}" y2="{_n(y + stem)}" '
        f'stroke="#000000" stroke-width="1"/>'
        f'<line x1="{_n(x - half)}" y1="{_n(y + stem)}" x2="{_n(x + half)}" '
        f'y2="{_n(y + stem)}" stroke="#000000" stroke-width="1.6"/>'
        f"</g>"
    )


