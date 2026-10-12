"""Selectable optimizer objectives: ``coh_pk`` and ``G_mod/in``.

The self-pumped design engine
(:func:`tvtl_ads_builder_v2.parity.self_pumped_design.design_optimized_self_pumped`)
historically had a single, implicit objective: select the first Sobol-screened
candidate whose analytical **coherent-peak gain** ``coh_pk`` (the pump-on/off
phase-sensitive boost) falls inside a target window. ``coh_pk`` is a *ratio*
metric — pump-on output over pump-off output — and it is **inflated by passive
suppression**: a design whose signal sits in a Bloch *stopband* has a tiny
pump-off denominator, so its ``coh_pk`` is large even though the absolute
output power is far below the available input. Maximising ``coh_pk`` therefore
drives the search toward stopband placement, which is the *worst* configuration
for net **transducer** gain (output power vs available source power).

This module adds a second, generically selectable objective — **G_mod/in**, the
out/in transducer gain — alongside ``coh_pk``. The two pieces:

* :func:`analytical_g_mod_in_dB` — the cheap, in-loop selection signal the engine
  maximises when ``objective="g_mod_in"``. It is the analytical out/in transducer
  gain, read straight off the Stage-4 analytical result that the walk-down
  already computes::

      G_mod/in (analytical) = abs_gain_off_dB + coherent_peak_dB
                            = passive |S21| at f_signal (pump off, matched cascade)
                              + parametric boost (coh_pk)

  This mirrors the loss-budget identity established by the net-gain
  investigation:
  ``G_mod/in = passive_transducer_dB + G_mod_unmod``. Because ``abs_gain_off_dB``
  is ``20·log10|S21[signal]|`` of the **full matched cascade** (input match +
  N cells + output match), it already carries the passband/stopband loss and the
  matching loss, so ranking by this proxy naturally favours low-loss passband
  placement over stopband-inflated ``coh_pk``. It is fully generic — computed from
  the candidate's own harmonic ABCD, no hard-coded device/topology/length.

* :func:`measure_g_mod_in` + :class:`GModInMeasurement` — the **ADS-HB realised**
  transducer gain and the **stability** classification, reusing the committed
  full-sum extraction
  (:func:`tvtl_ads_builder_v2.oa.psa_gain_artifact.run_full_sum_gain_sweep`) and
  the loss-invariant regime detector
  (:func:`tvtl_ads_builder_v2.analysis.regime_detector.classify_regime_loss_invariant`).
  A single call returns the realised ``G_mod/in``, the parametric boost, the
  passive ``|S21|``, the self-oscillation probe ``r_A``, the regime category, and
  whether the point is ``net_positive`` / ``stable``.

* :func:`classify_net_positive_frontier` + :class:`NetPositiveFrontier` — over a
  *pump ladder* of measurements at one operating point, decides whether
  net-positive transducer gain is reached **comfortably-stable** (a net-positive
  point that still has converged, non-lasing head-room at higher pump) or only at
  the **instability edge** (net-positive only at the highest converging pump,
  with the next step non-convergent / lasing). This is the honest
  "stable vs edge" verdict the net-gain frontier requires; per the framing rule,
  edge-only net-positive is reported as *near the parametric instability
  threshold*, never as a stable-amplifier claim.

The ``coh_pk`` objective and all its results are **unchanged** — this module is
purely additive. ``objective="coh_pk"`` (the default) reproduces the historical
walk-down selection byte-for-byte.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Sequence

from tvtl_ads_builder_v2.analysis.regime_detector import (
    R_A_LASER_THRESHOLD,
    classify_regime_loss_invariant,
)
from tvtl_ads_builder_v2.oa.psa_gain_artifact import (
    full_family_phasor_sum,
    run_full_sum_gain_sweep,
)

__all__ = [
    "OBJECTIVE_COH_PK",
    "OBJECTIVE_G_MOD_IN",
    "VALID_OBJECTIVES",
    "validate_objective",
    "analytical_g_mod_in_dB",
    "GModInMeasurement",
    "measure_g_mod_in",
    "NetPositiveFrontier",
    "classify_net_positive_frontier",
]

# Objective identifiers (the engine's ``objective=`` selector).
OBJECTIVE_COH_PK = "coh_pk"
OBJECTIVE_G_MOD_IN = "g_mod_in"
VALID_OBJECTIVES = (OBJECTIVE_COH_PK, OBJECTIVE_G_MOD_IN)


def validate_objective(objective: str) -> str:
    """Return ``objective`` if it is a recognised selector, else raise.

    Recognised: ``"coh_pk"`` (the historical, default objective) and
    ``"g_mod_in"`` (the out/in transducer-gain objective). A typo would
    otherwise silently fall through to the ``coh_pk`` path, so it is rejected.
    """

    if objective not in VALID_OBJECTIVES:
        raise ValueError(
            f"unknown objective {objective!r}; valid objectives are "
            f"{VALID_OBJECTIVES}."
        )
    return objective


# --------------------------------------------------------------------------- #
#  In-loop analytical objective signal                                        #
# --------------------------------------------------------------------------- #
def analytical_g_mod_in_dB(stage4_result: Mapping[str, object]) -> float:
    """Analytical out/in transducer-gain proxy in dB (the G_mod/in objective).

    Reads the two additive pieces straight off the Stage-4 analytical result
    produced by
    :func:`tvtl_ads_builder_v2.analytical.level10_realistic_param_gain`:

    * ``abs_gain_off_dB`` — ``20·log10|S21[signal]|`` of the **pump-off** matched
      cascade (input match + N cells + output match). For a matched two-port this
      equals ``10·log10(P_out_unmod / P_avs)``: the **passive transducer |S21|**
      at the signal frequency, carrying the passband/stopband loss *and* the
      matching loss.
    * ``coherent_peak_dB`` — the phase-peaked coherent (signal + degenerate idler)
      pump-on/off **boost** (the ``coh_pk`` metric).

    Their sum is the analytical out/in transducer gain, mirroring the net-gain
    identity ``G_mod/in = passive_transducer_dB + G_mod_unmod``. The
    ADS-HB full-sum boost differs from
    the 2-term analytical ``coherent_peak_dB`` by < 0.04 dB for the trustworthy
    designs, so this is a faithful selection proxy; the realised value is measured
    by :func:`measure_g_mod_in`.

    Returns ``nan`` if either piece is missing or non-finite (the candidate then
    sorts below any finite-proxy candidate in the g_mod_in walk-down).
    """

    try:
        passive = float(stage4_result["abs_gain_off_dB"])
        boost = float(stage4_result["coherent_peak_dB"])
    except (KeyError, TypeError, ValueError):
        return float("nan")
    if not (math.isfinite(passive) and math.isfinite(boost)):
        return float("nan")
    return passive + boost


# --------------------------------------------------------------------------- #
#  ADS-HB realised measurement + stability                                    #
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class GModInMeasurement:
    """One ADS-HB realised out/in transducer-gain + stability reading.

    ``stable`` is the *point* stability (the self-oscillation probe ``r_A`` is
    below the laser threshold and the HB converged); whether a *net-positive*
    point has operating head-room above it (the frontier "stable vs edge"
    distinction) is decided by :func:`classify_net_positive_frontier` over a pump
    ladder, not by a single point.
    """

    g_mod_in_dB: float          # 10·log10(Pout_mod / P_avs) — the headline out/in
    boost_dB: float             # full-sum pump-on/off (G_mod_unmod)
    passive_S21_dB: float       # 10·log10(Pout_unmod / P_avs) — pump-off transducer
    net_positive: bool          # g_mod_in_dB > 0 (output power exceeds available input)
    converged: bool             # driven HB converged (non-convergence => unstable proxy)
    r_A: float                  # |out(sig off, pump on)| / |out(sig on, pump on)| (nan if unprobed)
    regime: str                 # amplifier | passive | lasing | non-convergent
    stable: bool                # converged and r_A below the laser threshold (not lasing)
    peak_phase_deg: float       # pump phase maximising the full-sum output
    pump_power_dBm: float
    f_signal_Hz: float
    note: str = ""

    def to_dict(self) -> dict:
        return {
            "g_mod_in_dB": self.g_mod_in_dB,
            "boost_dB": self.boost_dB,
            "passive_S21_dB": self.passive_S21_dB,
            "net_positive": self.net_positive,
            "converged": self.converged,
            "r_A": self.r_A,
            "regime": self.regime,
            "stable": self.stable,
            "peak_phase_deg": self.peak_phase_deg,
            "pump_power_dBm": self.pump_power_dBm,
            "f_signal_Hz": self.f_signal_Hz,
            "note": self.note,
        }


def _spec_scalar(spec, name: str) -> float:
    return float(next(v.value for v in spec.var_block.variables if v.name == name))


def measure_g_mod_in(
    spec,
    run_dir: str | Path,
    *,
    probe_oscillation: bool = True,
    signal_off_dBm: float = -100.0,
    off_pump_power_dBm: float = -100.0,
    timeout_s: int = 1200,
    r_a_threshold: float = R_A_LASER_THRESHOLD,
) -> GModInMeasurement:
    """Measure the ADS-HB realised out/in transducer gain + stability of ``spec``.

    Reuses the committed full-sum extraction
    (:func:`tvtl_ads_builder_v2.oa.psa_gain_artifact.run_full_sum_gain_sweep`,
    which runs the pump-off reference + the per-phase pump-on degenerate
    folding-family sum) to obtain ``G_mod/in``, the boost, and the passive
    ``|S21|``. Stability is read loss-invariantly: the driven HB's convergence is
    the deep-runaway proxy, and — when ``probe_oscillation`` — a signal-off /
    pump-on run at the peak phase supplies the self-oscillation probe
    ``r_A = |out(signal off)| / |out(signal on)|``, fed to
    :func:`tvtl_ads_builder_v2.analysis.regime_detector.classify_regime_loss_invariant`.

    A pump power at or past the parametric-instability onset makes the driven HB
    non-convergent; that is **data**, returned as ``converged=False``,
    ``regime="non-convergent"`` (treated as the unstable branch), not an error.

    Requires a reachable ``hpeesofsim`` (run under the ADS Python interpreter);
    the heavy imports are deferred so importing this module never needs ADS.
    """

    from tvtl_ads_builder_v2.analysis.hb_dataset import read_hb_dataset
    from tvtl_ads_builder_v2.device_spec import with_var_overrides
    from tvtl_ads_builder_v2.netlist import emit_flat_hb_netlist
    from tvtl_ads_builder_v2.simulate.runner import run_hpeesofsim

    run_dir = Path(run_dir)
    pump_dBm = _spec_scalar(spec, "pump_power_dBm")
    f_s = _spec_scalar(spec, "f_signal")
    f_p = _spec_scalar(spec, "f_pump")

    # Driven sweep (pump on, signal on). Non-convergence => instability onset.
    try:
        sw = run_full_sum_gain_sweep(
            spec, run_dir, off_pump_power_dBm=off_pump_power_dBm, timeout_s=timeout_s,
        )
    except Exception as exc:  # noqa: BLE001 -- non-convergence is the instability proxy
        return GModInMeasurement(
            g_mod_in_dB=float("nan"), boost_dB=float("nan"),
            passive_S21_dB=float("nan"), net_positive=False, converged=False,
            r_A=float("nan"), regime="non-convergent", stable=False,
            peak_phase_deg=float("nan"), pump_power_dBm=pump_dBm, f_signal_Hz=f_s,
            note=f"driven HB non-convergent (instability onset proxy): "
                 f"{type(exc).__name__}: {exc}",
        )

    passive = (
        10.0 * math.log10(sw.pout_unmod_W / sw.p_avs_W)
        if sw.pout_unmod_W > 0 and sw.p_avs_W > 0 else float("-inf")
    )
    out_on = math.sqrt(2.0 * 50.0 * sw.pout_mod_W)  # |V_mod| at the peak phase

    r_A = float("nan")
    out_off = float("nan")
    if probe_oscillation:
        spec_off = with_var_overrides(
            spec,
            {"signal_power_dBm": float(signal_off_dBm),
             "pump_phase_rad": math.radians(sw.peak_phase_deg)},
        )
        ws = run_dir / "sigoff_pumpon"
        ws.mkdir(parents=True, exist_ok=True)
        deck = ws / "sigoff_pumpon.ckt"
        deck.write_text(emit_flat_hb_netlist(spec_off), encoding="utf-8")
        res = run_hpeesofsim(deck, timeout_s=timeout_s)
        if res["status"] == "completed":
            ds = read_hb_dataset(res["dataset_path"], (f_s, f_p))
            out_off = abs(full_family_phasor_sum(ds.node_voltage("n_output")))
            r_A = (out_off / out_on) if (out_on > 0 and math.isfinite(out_off)) else float("nan")

    g_mod_in = sw.g_mod_in_dB
    if math.isfinite(out_off):
        # Loss-invariant regime: needs the signal-off self-oscillation reading.
        reg = classify_regime_loss_invariant(
            out_signal_off=out_off, out_signal_on=out_on,
            enhancement_dB=sw.g_mod_unmod_dB, idler_over_sig=None,
            converged=True, r_a_threshold=r_a_threshold,
        )
        regime = reg.category
        stable = regime != "lasing" and (not math.isfinite(r_A) or r_A < r_a_threshold)
        note = ""
    else:
        # No (or failed) oscillation probe -> the loss-invariant detector cannot
        # run; report the gain but do NOT certify stability.
        regime = "unprobed"
        stable = False
        note = "oscillation not probed (r_A unavailable); stability not certified"
    return GModInMeasurement(
        g_mod_in_dB=float(g_mod_in), boost_dB=float(sw.g_mod_unmod_dB),
        passive_S21_dB=float(passive), net_positive=bool(g_mod_in > 0.0),
        converged=True, r_A=float(r_A), regime=regime, stable=bool(stable),
        peak_phase_deg=float(sw.peak_phase_deg), pump_power_dBm=float(pump_dBm),
        f_signal_Hz=float(f_s), note=note,
    )


# --------------------------------------------------------------------------- #
#  Net-positive frontier: stable vs instability-edge                          #
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class NetPositiveFrontier:
    """The "stable vs instability-edge" verdict over a pump ladder.

    ``verdict`` is one of:

    * ``"stable_net_positive"`` — a net-positive point exists that is itself
      stable **and** still has converged, non-lasing head-room at higher pump
      (you reach net-positive without sitting on the ragged edge). This is the
      genuine, robust win.
    * ``"edge_net_positive"`` — net-positive is reached, but only at the highest
      converging pump, with the next pump step non-convergent / lasing (no
      head-room). Per the framing rule this is *near the parametric instability
      threshold*, NOT a stable-amplifier claim.
    * ``"negative"`` — no converged point reaches net-positive.
    """

    best_g_mod_in_dB: float
    best_pump_dBm: float
    net_positive_reached: bool
    stable_net_positive: bool
    instability_edge_only: bool
    best_stable_g_mod_in_dB: float   # best G_mod/in over points stable WITH head-room
    verdict: str
    note: str

    def to_dict(self) -> dict:
        return {
            "best_g_mod_in_dB": self.best_g_mod_in_dB,
            "best_pump_dBm": self.best_pump_dBm,
            "net_positive_reached": self.net_positive_reached,
            "stable_net_positive": self.stable_net_positive,
            "instability_edge_only": self.instability_edge_only,
            "best_stable_g_mod_in_dB": self.best_stable_g_mod_in_dB,
            "verdict": self.verdict,
            "note": self.note,
        }


def classify_net_positive_frontier(
    measurements: Sequence[GModInMeasurement],
) -> NetPositiveFrontier:
    """Decide whether net-positive G_mod/in is comfortably-stable or edge-only.

    ``measurements`` is a pump ladder at one operating point (one device, one
    geometry, one signal frequency), in any pump order — it is sorted ascending
    by pump here. A net-positive converged point has **head-room** if there is a
    higher-pump point that still converged and is not lasing; a net-positive
    point that is stable *and* has head-room is a genuine stable net-positive,
    otherwise net-positive is edge-only (it lives on the
    gain-equals-loss-equals-instability boundary).

    Raises ``ValueError`` on an empty ladder.
    """

    ms = sorted(measurements, key=lambda m: m.pump_power_dBm)
    if not ms:
        raise ValueError("classify_net_positive_frontier needs >= 1 measurement.")

    converged = [m for m in ms if m.converged]
    if converged:
        best = max(converged, key=lambda m: m.g_mod_in_dB)
        best_g, best_pump = best.g_mod_in_dB, best.pump_power_dBm
    else:
        best_g, best_pump = float("nan"), float("nan")

    def _has_headroom(idx: int) -> bool:
        """A higher-pump point still converged and not lasing (operating margin)."""
        return any(
            m.converged and m.regime != "lasing"
            for m in ms[idx + 1:]
        )

    net_positive_points = [
        (i, m) for i, m in enumerate(ms) if m.converged and m.net_positive
    ]
    net_positive_reached = bool(net_positive_points)

    stable_with_headroom = [
        m for (i, m) in net_positive_points if m.stable and _has_headroom(i)
    ]
    stable_net_positive = bool(stable_with_headroom)
    best_stable_g = (
        max(m.g_mod_in_dB for m in stable_with_headroom)
        if stable_with_headroom else float("nan")
    )
    instability_edge_only = net_positive_reached and not stable_net_positive

    if stable_net_positive:
        verdict = "stable_net_positive"
        note = (
            f"net-positive reached comfortably-stable: best stable "
            f"G_mod/in={best_stable_g:+.2f} dB with converged head-room above."
        )
    elif net_positive_reached:
        verdict = "edge_net_positive"
        note = (
            "net-positive reached ONLY at the instability edge (no converged, "
            "non-lasing head-room above the net-positive pump) -- near the "
            "parametric instability threshold, not a stable amplifier."
        )
    else:
        verdict = "negative"
        note = (
            f"no converged point reaches net-positive (best "
            f"G_mod/in={best_g:+.2f} dB)."
        )

    return NetPositiveFrontier(
        best_g_mod_in_dB=float(best_g),
        best_pump_dBm=float(best_pump),
        net_positive_reached=net_positive_reached,
        stable_net_positive=stable_net_positive,
        instability_edge_only=instability_edge_only,
        best_stable_g_mod_in_dB=float(best_stable_g),
        verdict=verdict,
        note=note,
    )


