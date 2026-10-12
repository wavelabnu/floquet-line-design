"""Top-level rail placement of the hierarchical OA emitter (base Python, no ADS).

The hierarchical top level compresses the ``unit_cell:Xk`` cascade onto a fixed
``_TOP_CELL_STRIDE`` pitch and special-cases the scaffolding that rides the rail
(inter-cell probes, IPR_IN / IPR_OUT, P_OUT) so every connecting wire is short.
A low-edge-impedance design additionally carries single-stub / quarter-wave
**matching networks** (``MATCH_{IN,OUT}_*``); these tests lock that those land on
the rail beside their port probe -- the input network LEFT of IPR_IN, the output
network RIGHT of IPR_OUT, the load port rightmost -- with no body overlaps,
instead of keeping their uncompressed flat coordinate (which lands MATCH_IN_* on
top of X1 and strands MATCH_OUT_* past the rail end).

The placement is geometry only; :func:`_rail_overrides` is a pure coordinate map,
so these run with no ADS installation. The companion full-render check lives in
the ADSPY verification script.
"""

from __future__ import annotations

import numpy as np
import pytest

from tvtl.optimization.stage3_matching import MatchingSection
from tvtl.optimization.topology_generator import (
    Termination,
    TopologyFamily,
    ViaConfig,
)
from tvtl.varactors import RealisticVaractor
from tvtl_ads_builder_v2.oa.oa_emitter import _effective_elements
from tvtl_ads_builder_v2.oa.oa_hier_emitter import (
    _HB_GAP,
    _INSTR_DX,
    _MSUB_Y,
    _SPARAM_Y,
    _SRC_DX,
    _SRC_PITCH,
    _SRC_Y0,
    _TOP_CELL_STRIDE,
    _UC_SYM_HALF_H,
    _UC_SYM_HALF_W,
    _cell_index,
    _has_match_in,
    _rail_overrides,
)
from tvtl_ads_builder_v2.parity.optimized_design import optimized_control_port_spec

_VAR_A = RealisticVaractor(CJ0_f=0.3e-12, VJ_v=3.5, M=1.7, CP_f=0.0,
                           L_pkg_h=0.3e-9, R_series_ohm=0.5)
_FAMILY = TopologyFamily(n_segments=2, via=ViaConfig.DOUBLE, termination=Termination.SHORT)
_PARAMS = np.array([2.0e-3, 7.0e-3, 4.0e-3, 14.0e-3])  # n2_double_short, ~15 ohm edge


def _single_stub(stub_kind: str) -> MatchingSection:
    """A minimal single-stub section the emitter reads (.topology, .elements)."""

    return MatchingSection(
        topology="single_stub",
        elements=(
            ("series_line", 1.5e-3, 4.0e-3, 50.0),
            (stub_kind, 1.0e-3, 3.0e-3, 50.0),
        ),
        s11_before_dB=-2.0, s11_after_dB=-25.0, frac_bandwidth_pct=20.0,
        realizable=True,
    )


def _matched_spec(*, input_match=True, output_match=True):
    """An n2_double_short cascade with single-stub matching on the chosen ports.

    Input uses a short stub (exercises MATCH_IN_SERIES/STUB/STUB_VIA); output uses
    an open stub (exercises MATCH_OUT_SERIES/STUB, no via).
    """

    return optimized_control_port_spec(
        topology=_FAMILY, params=_PARAMS, varactor=_VAR_A, n_cells=15,
        main_width_m=2.49e-3, spec_id="n2_double_short_matched",
        input_match=_single_stub("shunt_short_stub") if input_match else None,
        output_match=_single_stub("shunt_open_stub") if output_match else None,
    )


# Conservative top-level body half-extents (ADS units). Real ADS symbols are no
# larger; the ADSPY render scan re-checks against the live geometry.
_UC_HW, _UC_HH = _UC_SYM_HALF_W, _UC_SYM_HALF_H        # 3.0 x 2.0
_COL_HW, _COL_HH = 3.0, 2.0                             # sources / controllers
_FOOT = {                                               # by element type
    "I_Probe": (1.0, 0.75),
    "MLIN": (2.0, 1.0),
    "L": (1.0, 1.0),
    "TermG": (1.5, 1.5),
}


def _scaffolding(spec):
    return [e for e in _effective_elements(spec) if _cell_index(e.id) is None]


def _n_cells(spec):
    return max(k for e in _effective_elements(spec)
               if (k := _cell_index(e.id)) is not None)


def _top_bodies(spec):
    """Every top-level body as (id, x, y, half_w, half_h) -- the emitter's layout.

    Replicates the hierarchical top level for the bodies the rail layout owns:
    the N unit-cell symbols, the special-cased scaffolding (via _rail_overrides),
    the stacked sources, and the controller column. (Coincident source/branch
    GROUNDs follow their pins and are covered by the live ADSPY scan.)
    """

    scaff = _scaffolding(spec)
    n_cells = _n_cells(spec)
    etype = {e.id: e.type for e in scaff}
    overrides = _rail_overrides(scaff, n_cells)

    bodies = []
    # N unit-cell symbols on the tight pitch
    for k in range(1, n_cells + 1):
        bodies.append((f"X{k}", (k - 1) * _TOP_CELL_STRIDE, 0.0, _UC_HW, _UC_HH))
    # special-cased scaffolding (probes, matching networks, output port)
    for eid, (x, y) in overrides.items():
        hw, hh = _FOOT[etype[eid]]
        bodies.append((eid, x, y, hw, hh))
    # source stack + controller column (their x steps left when input matches)
    span = _TOP_CELL_STRIDE if _has_match_in(scaff) else 0.0
    src_x = -_TOP_CELL_STRIDE / 2.0 - span - _SRC_DX
    instr_x = -_TOP_CELL_STRIDE / 2.0 - span - _INSTR_DX
    for i, src in enumerate(spec.sources):
        bodies.append((src.name, src_x, _SRC_Y0 - _SRC_PITCH * i, _COL_HW, _COL_HH))
    for name, y in (("MSUB", _MSUB_Y), ("SP1", _SPARAM_Y), ("HB1", _SPARAM_Y - _HB_GAP)):
        bodies.append((name, instr_x, y, _COL_HW, _COL_HH))
    return bodies


def _x(bodies, name):
    return next(b[1] for b in bodies if b[0] == name)


def _strict_overlap(a, b) -> bool:
    """Positive-area (interior) intersection of two axis-aligned boxes."""

    _ia, ax, ay, ahw, ahh = a
    _ib, bx, by, bhw, bhh = b
    return abs(ax - bx) < (ahw + bhw) - 1e-9 and abs(ay - by) < (ahh + bhh) - 1e-9


# --------------------------------------------------------------------------- #
# Placement
# --------------------------------------------------------------------------- #
def test_match_in_lands_left_of_ipr_in_and_off_x1():
    spec = _matched_spec()
    bodies = _top_bodies(spec)
    ipr_in_x = _x(bodies, "IPR_IN")
    x1_x = _x(bodies, "X1")
    for eid in ("MATCH_IN_SERIES", "MATCH_IN_STUB", "MATCH_IN_STUB_VIA"):
        mx = _x(bodies, eid)
        assert mx < ipr_in_x, f"{eid} x={mx} not left of IPR_IN x={ipr_in_x}"
        # clearly off X1 (the flat-layout bug parked MATCH_IN_SERIES at X1's x)
        assert x1_x - mx >= _TOP_CELL_STRIDE - 1e-9


def test_match_out_lands_right_of_ipr_out_and_port_is_rightmost():
    spec = _matched_spec()
    bodies = _top_bodies(spec)
    ipr_out_x = _x(bodies, "IPR_OUT")
    p_out_x = _x(bodies, "P_OUT")
    for eid in ("MATCH_OUT_SERIES", "MATCH_OUT_STUB"):
        mx = _x(bodies, eid)
        assert mx > ipr_out_x, f"{eid} x={mx} not right of IPR_OUT x={ipr_out_x}"
    # the load port stays the rightmost rail element
    assert p_out_x == max(b[1] for b in bodies)


def test_through_row_rail_is_straight():
    """Series matching lines, probes, symbols and the port share the y=0 row."""

    spec = _matched_spec()
    bodies = {b[0]: b for b in _top_bodies(spec)}
    on_rail = ["X1", "X15", "IPR_IN", "IPR_OUT", "IPR_C1C2",
               "MATCH_IN_SERIES", "MATCH_OUT_SERIES", "P_OUT"]
    for name in on_rail:
        assert bodies[name][2] == pytest.approx(0.0), f"{name} off the rail row"
    # the shunt stub / via drop BELOW the rail (read as a shunt branch)
    assert bodies["MATCH_IN_STUB"][2] < 0.0
    assert bodies["MATCH_IN_STUB_VIA"][2] < bodies["MATCH_IN_STUB"][2]


def test_no_strict_overlaps_anywhere():
    spec = _matched_spec()
    bodies = _top_bodies(spec)
    hits = [
        (a[0], b[0])
        for i, a in enumerate(bodies)
        for b in bodies[i + 1:]
        if _strict_overlap(a, b)
    ]
    assert hits == [], f"overlapping bodies: {hits}"


@pytest.mark.parametrize("which", ["input", "output"])
def test_single_side_matching_also_clean(which):
    spec = _matched_spec(input_match=(which == "input"),
                         output_match=(which == "output"))
    bodies = _top_bodies(spec)
    hits = [(a[0], b[0]) for i, a in enumerate(bodies) for b in bodies[i + 1:]
            if _strict_overlap(a, b)]
    assert hits == []


# --------------------------------------------------------------------------- #
# Regression: the no-match path (the three baked examples) is unchanged
# --------------------------------------------------------------------------- #
def test_no_match_overrides_match_historical_formula():
    """With no matching network the rail overrides are byte-for-byte the historical
    layout: IPR_IN one half-stride left of X1, P_OUT one stride past the last cell,
    and no MATCH_* keys -- so the baked no-match examples are untouched."""

    spec = _matched_spec(input_match=False, output_match=False)
    scaff = _scaffolding(spec)
    n_cells = _n_cells(spec)
    assert not _has_match_in(scaff)
    overrides = _rail_overrides(scaff, n_cells)
    half = _TOP_CELL_STRIDE / 2.0
    assert overrides["IPR_IN"] == (0.0 - half, 0.0)
    assert overrides["IPR_OUT"] == ((n_cells - 1) * _TOP_CELL_STRIDE + half, 0.0)
    assert overrides["P_OUT"] == ((n_cells - 1) * _TOP_CELL_STRIDE + _TOP_CELL_STRIDE, 0.0)
    assert not any(k.startswith("MATCH_") for k in overrides)
