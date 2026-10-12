"""Frozen optimizer-picked PROXY example designs (n2/n3/n4 SHORT cells).

Three worked TVTL designs the *public optimizer* picked, built entirely from the
public ``tvtl`` physics core and the public ``tvtl_ads_builder_v2`` ADS builder.
They extend the three real-part examples (``real_part_designs.py``) to topology
cells the trio does not cover, and they exercise the input/output **matching
networks** (the trio's edge impedances need none) -- the real-world confirmation
that the hierarchical OA emitter's matching-network placement draws clean.

Provenance (reproducible offline, no spreadsheet, no private path):

    floquet-line-design optimize-amplifier --freq-ghz 2.0 --out <dir>

(the public CLI; default budget 300 Sobol samples x 12 linear families = 3247
screened, top-10 refined, seed 0, n_cells 15, drive -3 dBm) is DETERMINISTIC and
its three diversified picks are, in rank order::

    1. n2_double_short   (2 segments, double via, SHORT)   <- best candidate
    2. n4_double_short   (4 segments, double via, SHORT)
    3. n3_single_short   (3 segments, single via, SHORT)

each a distinct ``(stub-count x termination)`` cell, none equal to the trio
(SMV1245 = n3_none_open, SMV2020 = n1_single_short, MAVR = n4_single_combined)
and none equal to each other. The picks all share Z_TL = 61.27 ohm / w = 2.494 mm
(the screen's A1 ceiling) and single-stub input/output matching. Every geometry
literal below is the FROZEN optimizer pick (full precision, metres); the varactor
is the optimizer's default low-parasitic analytical proxy.

Each spec is rebuilt the way the optimizer builds its candidates internally
(``parity.self_pumped_design._evaluate_candidate_proxy``): the Stage-2 Bloch match
``match_z_tl`` at its defaults (Z_TL search 20..120 ohm), Stage-3 ``edge_impedance``
-> ``select_matching`` for the input/output single-stub networks, on the accurate
distributed terminations (ADS ``VIAGND`` physical via + ``MLEF`` open-end fringe) --
the same construction path the real-part examples use in
``real_part_designs.build_fixed_geom``, driven by the optimizer pick instead of a
frozen real-part geometry. The reconstruction is bit-faithful (each design's
recovered Z_TL and main width equal the recorded optimizer pick exactly; checked
in ``test`` / ``mode_specs``).

``EXAMPLES_PROXY`` lists the three with the library / top-cell names the
hierarchical emitter uses; ``REFERENCE_VOUT_SIGNAL_PROXY`` records each design's
signal-frequency output voltage at its convergent operating point (the parity
gate). ``OPERATING_PUMP_DBM`` is each design's convergent, bounded-lobe operating
pump (HB converges and the parametric analysis is meaningful -- NOT an
oscillation-edge non-convergent point); see ``analyze_proxy.py``.
"""
from __future__ import annotations

from dataclasses import dataclass

from tvtl.optimization.stage1_per_cell import StubCandidate
from tvtl.optimization.stage2_main_tl import match_z_tl
from tvtl.optimization.stage3_matching import edge_impedance, select_matching
from tvtl.optimization.substrate import ROGERS_4003C
from tvtl.optimization.topology_generator import (
    Termination,
    TopologyFamily,
    ViaConfig,
)
from tvtl.varactors import RealisticVaractor
from tvtl_ads_builder_v2.device_spec import resolve_spec
from tvtl_ads_builder_v2.models import MlefOpen, ViagndVia
from tvtl_ads_builder_v2.parity.self_pumped_design import optimized_self_pumped_spec

__all__ = [
    "SUBSTRATE",
    "PROXY_VARACTOR",
    "PROXY_PICKS",
    "OPERATING_PUMP_DBM",
    "build_proxy_spec",
    "spec_proxy_n2_double_short",
    "spec_proxy_n3_single_short",
    "spec_proxy_n4_double_short",
    "resolved_spec",
    "EXAMPLES_PROXY",
    "REFERENCE_VOUT_SIGNAL_PROXY",
]

SUBSTRATE = ROGERS_4003C
_F_S_HZ = 2.0e9
_N_CELLS = 15

# The optimizer's default low-parasitic analytical proxy varactor
# (verbatim tvtl.optimization.cli._default_varactor). Idealized: the realized ADS
# gain is a faithfulness RECORD, never reported as a real-world device gain.
PROXY_VARACTOR = RealisticVaractor(
    CJ0_f=0.3e-12, VJ_v=3.5, M=1.7, CP_f=0.0, L_pkg_h=0.3e-9, R_series_ohm=0.5
)


@dataclass(frozen=True)
class ProxyPick:
    """One frozen optimizer pick: topology + segment-ordered stub params (metres)."""

    topology: TopologyFamily
    params: tuple[float, ...]
    f_s_Hz: float = _F_S_HZ
    n_cells: int = _N_CELLS
    # Recorded optimizer-pick metadata (for the bit-faithful reconstruction guard).
    summary_z_tl_ohm: float = 61.269841269841265
    summary_main_width_m: float = 0.002493590861719739
    analytical_coherent_peak_dB: float = float("nan")


# Frozen picks -- full-precision metres straight from the deterministic CLI run.
PROXY_PICKS: dict[str, ProxyPick] = {
    "proxy_n2_double_short": ProxyPick(
        topology=TopologyFamily(n_segments=2, via=ViaConfig.DOUBLE, termination=Termination.SHORT),
        params=(0.002064896048995837, 0.00692911184216039,
                0.004235218849693343, 0.013877658875017301),
        analytical_coherent_peak_dB=10.632571693309034,
    ),
    "proxy_n3_single_short": ProxyPick(
        topology=TopologyFamily(n_segments=3, via=ViaConfig.SINGLE, termination=Termination.SHORT),
        params=(0.0006742931770352891, 0.009675199418467447,
                0.0034501144509067336, 0.0036865178393537027,
                0.003613807748585201, 0.007534543842344881),
        analytical_coherent_peak_dB=10.11182951048201,
    ),
    "proxy_n4_double_short": ProxyPick(
        topology=TopologyFamily(n_segments=4, via=ViaConfig.DOUBLE, termination=Termination.SHORT),
        params=(0.004163709396593798, 0.0024170631534919463,
                0.0013860955333413996, 0.0021169648700550946,
                0.0031983256499096236, 0.008595661569788795,
                0.004706965668737859, 0.007004340577723478),
        analytical_coherent_peak_dB=10.404413404370647,
    ),
}

# Convergent, bounded-lobe operating pump per design (dBm). At each the HB
# converges and the degenerate-parametric phase lobe is bounded (finite peak,
# anti-phase deamplification trough); each is the boost-optimal convergent pump,
# one 3 dB step below where the pump-on/off boost collapses into the
# oscillation/saturation edge. Established by analyze_proxy.py (per-design
# pump-power scan + phase sweep).
OPERATING_PUMP_DBM: dict[str, float] = {
    "proxy_n2_double_short": 6.0,    # boost +2.86 dB @45 deg, abs -3.92 dB; HB non-conv at +15 dBm
    "proxy_n3_single_short": 15.0,   # boost +8.18 dB @45 deg, abs +1.18 dB (net-positive); collapse at +18
    "proxy_n4_double_short": 9.0,    # boost +2.51 dB @90 deg, abs -4.51 dB; collapse at +12
}


def _matcher(sm):
    if not sm.needed:
        return None
    return sm.single_stub if sm.recommended == "single_stub" else sm.quarter_wave


def build_proxy_spec(key, *, pump_dBm=None, n_cells=None, with_matching=True):
    """Reconstruct one frozen proxy pick as a VIAGND/MLEF self-pumped DeviceSpec.

    Mirrors the optimizer's internal candidate build: ``match_z_tl`` at its
    defaults (Z_TL 20..120 ohm) -> ``edge_impedance`` -> ``select_matching``,
    proxy varactor, accurate VIAGND short / MLEF open-end fringe. ``pump_dBm``
    defaults to the design's convergent operating point. ``with_matching=False``
    builds the same cascade with no input/output matching networks.
    """
    pick = PROXY_PICKS[key]
    nc = pick.n_cells if n_cells is None else int(n_cells)
    pump = OPERATING_PUMP_DBM[key] if pump_dBm is None else float(pump_dBm)
    dev = PROXY_VARACTOR
    cand = StubCandidate(topology=pick.topology, params=pick.params, a1_score=0.0, f0_Hz=pick.f_s_Hz)
    mtl = match_z_tl(cand, dev, substrate=SUBSTRATE, f0_Hz=pick.f_s_Hz)
    z_in, z_out = edge_impedance(mtl, dev, nc, substrate=SUBSTRATE)
    if with_matching:
        im = _matcher(select_matching(z_in, "input", pick.f_s_Hz, substrate=SUBSTRATE))
        om = _matcher(select_matching(z_out, "output", pick.f_s_Hz, substrate=SUBSTRATE))
    else:
        im = om = None
    return optimized_self_pumped_spec(
        topology=pick.topology, params=pick.params, varactor=dev,
        substrate=SUBSTRATE, n_cells=nc, main_width_m=float(mtl.w_main_m),
        f_signal_Hz=pick.f_s_Hz, f_pump_Hz=2.0 * pick.f_s_Hz,
        pump_power_dBm=pump, pump_phase_rad=0.0,
        input_match=im, output_match=om,
        via_model=ViagndVia(), open_end_model=MlefOpen())


def spec_proxy_n2_double_short(pump_dBm=None, n_cells=None) -> "object":
    """Optimizer best pick: n2 double-via SHORT, 15 cells, 2.0 GHz (generalization test)."""
    return build_proxy_spec("proxy_n2_double_short", pump_dBm=pump_dBm, n_cells=n_cells)


def spec_proxy_n3_single_short(pump_dBm=None, n_cells=None) -> "object":
    """Optimizer pick: n3 single-via SHORT, 15 cells, 2.0 GHz."""
    return build_proxy_spec("proxy_n3_single_short", pump_dBm=pump_dBm, n_cells=n_cells)


def spec_proxy_n4_double_short(pump_dBm=None, n_cells=None) -> "object":
    """Optimizer pick: n4 double-via SHORT, 15 cells, 2.0 GHz."""
    return build_proxy_spec("proxy_n4_double_short", pump_dBm=pump_dBm, n_cells=n_cells)


# (key, spec-builder, library name, top-cell name, one-line headline)
EXAMPLES_PROXY = (
    ("proxy_n2_double_short", spec_proxy_n2_double_short,
     "proxy_n2_double_short", "proxy_n2_double_short_top",
     "Optimizer-picked n2_double_short proxy amplifier -- 2-section SHORT/VIAGND "
     "(double via) + single-stub matching, 15 cells, 2.0 GHz (generalization-test design)"),
    ("proxy_n3_single_short", spec_proxy_n3_single_short,
     "proxy_n3_single_short", "proxy_n3_single_short_top",
     "Optimizer-picked n3_single_short proxy amplifier -- 3-section SHORT, single "
     "VIAGND + single-stub matching, 15 cells, 2.0 GHz"),
    ("proxy_n4_double_short", spec_proxy_n4_double_short,
     "proxy_n4_double_short", "proxy_n4_double_short_top",
     "Optimizer-picked n4_double_short proxy amplifier -- 4-section SHORT, double "
     "VIAGND + single-stub matching, 15 cells, 2.0 GHz"),
)

# Signal-frequency output-node voltage each design reproduces under HB at its
# operating point (pump = OPERATING_PUMP_DBM, phase 0), f_s in GHz alongside (the
# parity reference; see build_examples.py). Recorded by analyze_proxy.py.
REFERENCE_VOUT_SIGNAL_PROXY: dict[str, tuple[float, float]] = {
    "proxy_n2_double_short": (4.115031e-03, 2.0),
    "proxy_n3_single_short": (6.151059e-03, 2.0),
    "proxy_n4_double_short": (4.114838e-03, 2.0),
}


def resolved_spec(builder):
    """Build a spec and resolve its VAR block to literals (idempotent w/ emitters)."""
    return resolve_spec(builder())
