"""Stage 3 -- matching-network design (quarter-wave + single-stub).

After Stage 2 has adapted the main line, the periodic cascade presents an edge
impedance ``Z_edge`` at each port (the loaded-line Bloch impedance for a
well-formed design; ~50 ohm for weakly-loaded stubs, low for high-gain stubs).
Stage 3 designs a **distributed-microstrip** matching network per side to
transform ``Z_edge`` to 50 ohm, conditional on the unmatched reflection:

* if ``|S11_side| < -15 dB`` already, "no matching needed" for that side;
* else compute **both** a quarter-wave transformer and a single-stub match,
  report each (element geometry, achieved |S11|, approx. bandwidth), and
  recommend one. Input and output are evaluated **independently**.

All sections are distributed microstrip on the configured substrate -- no
lumped surrogates -- so they cross-validate in the fabrication medium.
The decision is semi-automated: the framework recommends; the designer confirms
the final selection.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from tvtl.abcd.matrices import line_abcd, shunt_admittance_abcd
from tvtl.materials import SPEED_OF_LIGHT_M_PER_S
from tvtl.microstrip.models import hammerstad_kirschning
from tvtl.varactors import RealisticVaractor

from .stage2_main_tl import MainTLDesign, static_cell_abcd
from .substrate import ROGERS_4003C, SubstrateConfig, microstrip_width_for_z0

__all__ = [
    "MatchingSection",
    "SideMatching",
    "edge_impedance",
    "s11_dB",
    "quarter_wave_design",
    "single_stub_design",
    "select_matching",
    "MATCH_THRESHOLD_DB",
    "Z_REF_OHM",
]

Z_REF_OHM = 50.0
MATCH_THRESHOLD_DB = -15.0


@dataclass(frozen=True)
class MatchingSection:
    """One matching-network topology realized as distributed microstrip.

    ``elements`` is an ordered tuple of ``(kind, w_m, l_m, z0_ohm)`` microstrip
    sections (``kind`` in ``{"series_line", "shunt_open_stub",
    "shunt_short_stub"}``), input-side first.
    """

    topology: str  # "quarter_wave" | "single_stub" | "none"
    elements: tuple[tuple[str, float, float, float], ...]
    s11_before_dB: float
    s11_after_dB: float
    frac_bandwidth_pct: float
    realizable: bool
    note: str = ""

    @property
    def n_elements(self) -> int:
        return len(self.elements)


@dataclass(frozen=True)
class SideMatching:
    """Per-port matching design (input or output)."""

    side: str  # "input" | "output"
    edge_z_ohm: complex
    s11_unmatched_dB: float
    needed: bool
    quarter_wave: MatchingSection | None
    single_stub: MatchingSection | None
    recommended: str  # "none" | "quarter_wave" | "single_stub"
    note: str = ""


# --------------------------------------------------------------------------- #
# Edge impedance + S-parameter helpers                                        #
# --------------------------------------------------------------------------- #
def s11_dB(z_in: complex, z_ref: float = Z_REF_OHM) -> float:
    """Return ``20*log10|Gamma|`` for an impedance ``z_in`` vs ``z_ref``."""

    gamma = (complex(z_in) - z_ref) / (complex(z_in) + z_ref)
    mag = abs(gamma)
    return float(20.0 * math.log10(max(mag, 1e-12)))


def edge_impedance(
    main_tl: MainTLDesign,
    varactor: RealisticVaractor,
    n_cells: int,
    *,
    substrate: SubstrateConfig = ROGERS_4003C,
    z_load_ohm: float = Z_REF_OHM,
) -> tuple[complex, complex]:
    """Input / output port impedance of the static N-cell cascade at f0 [ohm].

    Builds the static unit cell at the Stage-2 ``Z_TL``, raises it to ``n_cells``
    (block ABCD power), and returns ``(Z_in, Z_out)`` with the opposite port
    terminated in ``z_load_ohm`` (50 ohm). For a symmetric cascade the two
    coincide; both are returned so the framework supports per-side matching.
    """

    cell = static_cell_abcd(
        main_tl.stub, varactor, main_tl.z_tl_ohm, substrate=substrate,
        f0_Hz=main_tl.f0_Hz, use_mtee=True,
    )
    casc = np.linalg.matrix_power(np.asarray(cell, dtype=np.complex128), int(n_cells))
    a, b, c, d = casc[0, 0], casc[0, 1], casc[1, 0], casc[1, 1]
    zl = complex(z_load_ohm)
    z_in = (a * zl + b) / (c * zl + d)
    z_out = (d * zl + b) / (c * zl + a)
    return complex(z_in), complex(z_out)


# --------------------------------------------------------------------------- #
# Microstrip line helpers                                                      #
# --------------------------------------------------------------------------- #
def _eps_eff(width_m: float, substrate: SubstrateConfig, f0: float) -> float:
    m = substrate.stub_substrate_mapping()
    _z0, eps = hammerstad_kirschning(float(width_m), m["t_m"], m["h_m"], m["eps_r"], [float(f0)])
    return float(eps[0])


def _beta(width_m: float, substrate: SubstrateConfig, f0: float) -> float:
    return 2.0 * math.pi * float(f0) * math.sqrt(_eps_eff(width_m, substrate, f0)) / SPEED_OF_LIGHT_M_PER_S


def _quarter_wave_len(width_m: float, substrate: SubstrateConfig, f0: float) -> float:
    return SPEED_OF_LIGHT_M_PER_S / (4.0 * float(f0) * math.sqrt(_eps_eff(width_m, substrate, f0)))


def _frac_bandwidth_pct(
    section_abcd_at_freq, z_edge: complex, f0: float, *, span_frac: float = 0.6, n: int = 121
) -> float:
    """Approximate -10 dB |S11| fractional bandwidth (%) of a matching section.

    Sweeps ``f`` over ``f0*(1 +/- span_frac/2)``, holding ``z_edge`` fixed (its
    f0 value -- a first-order estimate ignoring edge dispersion), builds the
    section ABCD via ``section_abcd_at_freq(f)``, and finds the contiguous band
    around f0 with ``|S11| <= -10 dB``.
    """

    freqs = np.linspace(f0 * (1.0 - span_frac / 2.0), f0 * (1.0 + span_frac / 2.0), n)
    ok = np.zeros(n, dtype=bool)
    for i, f in enumerate(freqs):
        m = section_abcd_at_freq(float(f))
        a, b, c, d = m[0, 0], m[0, 1], m[1, 0], m[1, 1]
        z_in = (a * z_edge + b) / (c * z_edge + d)
        ok[i] = s11_dB(z_in) <= -10.0
    i0 = int(np.argmin(np.abs(freqs - f0)))
    if not ok[i0]:
        return 0.0
    lo = i0
    while lo - 1 >= 0 and ok[lo - 1]:
        lo -= 1
    hi = i0
    while hi + 1 < n and ok[hi + 1]:
        hi += 1
    return float(100.0 * (freqs[hi] - freqs[lo]) / f0)


# --------------------------------------------------------------------------- #
# Quarter-wave transformer (D1)                                               #
# --------------------------------------------------------------------------- #
def quarter_wave_design(
    z_edge: complex,
    f0: float,
    *,
    substrate: SubstrateConfig = ROGERS_4003C,
    z_ref: float = Z_REF_OHM,
) -> MatchingSection:
    """Quarter-wave transformer: ``Z_qw = sqrt(Z_ref * Re(Z_edge))``, l = lambda/4.

    Matches the real part of ``Z_edge`` exactly; any reactive part leaves a
    residual reflection (reported via ``s11_after_dB``). Realizable only if
    ``Z_qw`` is a microstrip-realizable impedance.
    """

    s11_before = s11_dB(z_edge, z_ref)
    re = float(z_edge.real)
    if re <= 0.0:
        return MatchingSection(
            "quarter_wave", (), s11_before, s11_before, 0.0, False,
            note=f"Re(Z_edge)={re:.2f} <= 0; quarter-wave transformer undefined.",
        )
    z_qw = math.sqrt(float(z_ref) * re)
    try:
        w_qw = microstrip_width_for_z0(z_qw, substrate, f0)
    except ValueError as exc:
        return MatchingSection(
            "quarter_wave", (), s11_before, s11_before, 0.0, False, note=str(exc),
        )
    len_qw = _quarter_wave_len(w_qw, substrate, f0)

    def _abcd(f: float) -> np.ndarray:
        b = _beta(w_qw, substrate, f0) * (f / f0) * len_qw  # electrical length scales with f
        return np.asarray(line_abcd(z_qw, 1j * b))

    m0 = _abcd(f0)
    z_in = (m0[0, 0] * z_edge + m0[0, 1]) / (m0[1, 0] * z_edge + m0[1, 1])
    s11_after = s11_dB(z_in, z_ref)
    fbw = _frac_bandwidth_pct(_abcd, z_edge, f0)
    return MatchingSection(
        topology="quarter_wave",
        elements=(("series_line", float(w_qw), float(len_qw), float(z_qw)),),
        s11_before_dB=s11_before, s11_after_dB=s11_after,
        frac_bandwidth_pct=fbw, realizable=True,
        note=f"Z_qw={z_qw:.1f} ohm, w={w_qw*1e3:.3f} mm, l={len_qw*1e3:.3f} mm.",
    )


# --------------------------------------------------------------------------- #
# Single-stub match (D3)                                                       #
# --------------------------------------------------------------------------- #
def single_stub_design(
    z_edge: complex,
    f0: float,
    *,
    substrate: SubstrateConfig = ROGERS_4003C,
    z_ref: float = Z_REF_OHM,
    stub_type: str = "open",
) -> MatchingSection:
    """Shunt single-stub match: series line (length d) + shunt open/short stub.

    Closed-form (Pozar): on a ``z_ref``-ohm main line, move distance ``d`` from
    the load so the real part of the normalized admittance is 1, then a shunt
    stub cancels the residual susceptance. Both sections are ``z_ref``-ohm
    microstrip. Verified by reconstructing the section ABCD and reporting the
    achieved |S11|.
    """

    s11_before = s11_dB(z_edge, z_ref)
    y0 = 1.0 / float(z_ref)
    y_l = (1.0 / complex(z_edge)) / y0  # normalized load admittance
    g, b = y_l.real, y_l.imag

    # Solve for t = tan(beta*d): coeff*t^2 + 2b*t + (g-1) = 0, coeff = g - g^2 - b^2.
    coeff = g - g * g - b * b
    ts: list[float] = []
    if abs(coeff) < 1e-12:
        if abs(b) > 1e-12:
            ts = [(1.0 - g) / (2.0 * b)]
    else:
        disc = b * b - coeff * (g - 1.0)
        if disc >= 0.0:
            root = math.sqrt(disc)
            ts = [(-b + root) / coeff, (-b - root) / coeff]
    if not ts:
        return MatchingSection(
            "single_stub", (), s11_before, s11_before, 0.0, False,
            note="no real single-stub solution (load on the unit conductance circle "
                 "is unreachable).",
        )

    try:
        w_line = microstrip_width_for_z0(z_ref, substrate, f0)
    except ValueError as exc:
        return MatchingSection("single_stub", (), s11_before, s11_before, 0.0, False, note=str(exc))
    beta = _beta(w_line, substrate, f0)

    # pick the smallest positive d
    best = None
    for t in ts:
        d_elec = math.atan(t)
        if d_elec < 0:
            d_elec += math.pi
        d_len = d_elec / beta
        # admittance at d: y(d) = (y_l + j t)/(1 + j y_l t); B_in = Im
        yd = (y_l + 1j * t) / (1.0 + 1j * y_l * t)
        b_in = yd.imag  # normalized residual susceptance to cancel with the stub
        b_stub = -b_in
        if stub_type == "open":
            # open stub normalized susceptance = tan(beta*l)
            l_elec = math.atan(b_stub)
            if l_elec < 0:
                l_elec += math.pi
        else:  # short stub: susceptance = -cot(beta*l)
            l_elec = math.atan(-1.0 / b_stub) if abs(b_stub) > 1e-12 else math.pi / 2.0
            if l_elec < 0:
                l_elec += math.pi
        l_stub = l_elec / beta
        cand = (d_len, l_stub, b_stub)
        if best is None or d_len < best[0]:
            best = cand
    d_len, l_stub, _b_stub = best

    # Reconstruct ABCD (input -> load): shunt stub @ series line(d).
    def _abcd(f: float) -> np.ndarray:
        bb = _beta(w_line, substrate, f0) * (f / f0)
        series = np.asarray(line_abcd(z_ref, 1j * bb * d_len))
        if stub_type == "open":
            y_stub = 1j * math.tan(bb * l_stub) / float(z_ref)
        else:
            tn = math.tan(bb * l_stub)
            y_stub = (-1j / math.tan(bb * l_stub) / float(z_ref)) if abs(tn) > 1e-12 else 0j
        shunt = np.asarray(shunt_admittance_abcd(y_stub))
        return shunt @ series

    m0 = _abcd(f0)
    z_in = (m0[0, 0] * z_edge + m0[0, 1]) / (m0[1, 0] * z_edge + m0[1, 1])
    s11_after = s11_dB(z_in, z_ref)
    fbw = _frac_bandwidth_pct(_abcd, z_edge, f0)
    kind = "shunt_open_stub" if stub_type == "open" else "shunt_short_stub"
    return MatchingSection(
        topology="single_stub",
        elements=(
            ("series_line", float(w_line), float(d_len), float(z_ref)),
            (kind, float(w_line), float(l_stub), float(z_ref)),
        ),
        s11_before_dB=s11_before, s11_after_dB=s11_after,
        frac_bandwidth_pct=fbw, realizable=True,
        note=f"d={d_len*1e3:.3f} mm, stub({stub_type})={l_stub*1e3:.3f} mm, "
             f"w={w_line*1e3:.3f} mm.",
    )


# --------------------------------------------------------------------------- #
# Per-side selection                                                           #
# --------------------------------------------------------------------------- #
def select_matching(
    z_edge: complex,
    side: str,
    f0: float,
    *,
    substrate: SubstrateConfig = ROGERS_4003C,
    z_ref: float = Z_REF_OHM,
    threshold_dB: float = MATCH_THRESHOLD_DB,
) -> SideMatching:
    """Per-side matching decision: none / quarter-wave / single-stub.

    If ``|S11| < threshold`` (default -15 dB) the side needs no matching. Else
    both topologies are designed and the one with the lower achieved |S11|
    (tie-break: the realizable one with the wider bandwidth) is recommended; the
    designer confirms the final selection.
    """

    s11_un = s11_dB(z_edge, z_ref)
    if s11_un < threshold_dB:
        return SideMatching(
            side=side, edge_z_ohm=complex(z_edge), s11_unmatched_dB=s11_un,
            needed=False, quarter_wave=None, single_stub=None, recommended="none",
            note=f"|S11|={s11_un:.1f} dB < {threshold_dB:.0f} dB; no matching needed.",
        )

    qw = quarter_wave_design(z_edge, f0, substrate=substrate, z_ref=z_ref)
    ss = single_stub_design(z_edge, f0, substrate=substrate, z_ref=z_ref)

    def _rank(sec: MatchingSection) -> tuple:
        # prefer realizable, then lower s11_after, then wider bandwidth
        return (0 if sec.realizable else 1, sec.s11_after_dB, -sec.frac_bandwidth_pct)

    candidates = [c for c in (qw, ss) if c.realizable]
    if not candidates:
        recommended = "none"
    else:
        best = min(candidates, key=_rank)
        recommended = best.topology
    return SideMatching(
        side=side, edge_z_ohm=complex(z_edge), s11_unmatched_dB=s11_un,
        needed=True, quarter_wave=qw, single_stub=ss, recommended=recommended,
    )


