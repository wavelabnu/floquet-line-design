"""Cheap, theory-free AMPLIFICATION-vs-OSCILLATION check at an operating point.

Given any self-pumped design ``spec``
at one operating point (a baked pump power + signal frequency), this module runs
the two physical, theory-free tests that distinguish a forced **amplifier**
(output tracks the injected signal) from a self-sustained **oscillator** (output
pinned regardless of the signal), plus a documented note on the rigorous
small-signal stability factor:

* **INPUT-DEPENDENCE test (primary).** Sweep the injected signal power (default
  −60 → −20 dBm) at the pump-on PSA-peak phase and read the signal-frequency
  output ``|Σ V_(m,n)|`` (the committed degenerate folding-family sum at
  ``n_output``). Fit the slope ``d log10|out| / d log10|in|``. Slope ≈ 1 ⇒ output
  scales with input ⇒ **amplification**; slope ≈ 0 ⇒ output pinned regardless of
  input ⇒ **oscillation**. (Test B of the loss-invariant regime detector.)

* **SIGNAL-OFF test (direct oscillation probe).** Pump on, signal driven to
  ``signal_off_dBm`` (default −200 dBm ≈ off), at the same peak phase. Measure the
  residual signal-frequency output and the loss-invariant ratio
  ``r_A = |out(signal off)| / |out(signal on)|``. Output present with no input
  (``r_A → 1``, well above the HB numerical floor) ⇒ **oscillating**; output at
  the floor (``r_A → 0``) ⇒ needs input ⇒ **amplifier**. (Test A — the primary
  test of the loss-invariant detector.)

* **K / μ (Rollett) stability factor (rigorous, if accessible).** The rigorous
  check is the Rollett ``K`` / Edwards-Sinsky ``μ`` on the **pumped-linearized**
  two-port around the operating point (NOT the passive network), formed from the
  HB small-signal/conversion S-parameters. This framework's flat-deck
  ``hpeesofsim`` path exposes only node-voltage harmonic mixes, not a pumped
  conversion S-matrix (no LSSP / conversion-S analysis), so K/μ is **not cleanly
  accessible** here — :func:`pumped_linearized_stability_available` reports that,
  and the verdict rests on the input-dependence + signal-off pair, exactly the
  fallback documented for that case.

Both operational tests are fed to the committed loss-invariant detector
(:func:`tvtl_ads_builder_v2.analysis.regime_detector.classify_regime_loss_invariant`)
and the verdict is the binary AMPLIFICATION / OSCILLATION mapping of its category.

SCOPE LIMIT (state it explicitly). These cheap checks certify amplification
**at an operating point** — they do NOT replace the analytical gain theory for the
deep lasing-vs-gain question near threshold. Driven HB converges to the *driven*
(input-tracking) branch, so the signal-off probe sees the amplified residual, not
the homogeneous self-sustained solution; near-threshold lasing manifests instead
as driven-HB non-convergence (reported as ``non-convergent``). Settling the
lasing-vs-gain distinction at threshold remains follow-on analytical theory
(round-trip loop gain with end reflections).

Additive and read-only on the design path: this module reuses the committed
full-sum extraction (:func:`...oa.psa_gain_artifact.run_full_sum_gain_sweep`,
:func:`...oa.psa_gain_artifact.full_family_phasor_sum`) and the committed regime
detector; it never rebuilds or perturbs a spec's circuit, so the A.3.1 anchor
deck is byte-unchanged. The heavy ADS imports are deferred inside the runner, so
importing this module never requires ADS (the pure helpers are base-Python and
unit-tested).
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path

from tvtl_ads_builder_v2.analysis.regime_detector import (
    R_A_LASER_THRESHOLD,
    SLOPE_LASER_THRESHOLD,
    classify_regime_loss_invariant,
)
from tvtl_ads_builder_v2.oa.psa_gain_artifact import (
    DEFAULT_PHASES_DEG,
    full_family_phasor_sum,
    run_full_sum_gain_sweep,
)

__all__ = [
    "DEFAULT_INPUT_SWEEP_DBM",
    "DEFAULT_SIGNAL_REF_DBM",
    "DEFAULT_SIGNAL_OFF_DBM",
    "DEFAULT_NOISE_FLOOR_V",
    "AMPLIFICATION",
    "OSCILLATION",
    "NON_CONVERGENT",
    "INDETERMINATE",
    "log_amplitude_slope",
    "classify_amplification",
    "pumped_linearized_stability_available",
    "AmplificationCheck",
    "run_amplification_check",
]

# Input-dependence sweep grid (dBm). −60 → −20 dBm is small-signal for these
# cascades (an amplifier stays linear; an oscillator is pinned far above −20 dBm).
DEFAULT_INPUT_SWEEP_DBM: tuple[float, ...] = (-60.0, -50.0, -40.0, -30.0, -20.0)
# Reference small-signal drive for the r_A denominator (the "signal on" point).
DEFAULT_SIGNAL_REF_DBM: float = -30.0
# Signal-off level for the direct oscillation probe (≈ off; the deck still emits a
# vanishing source so the HB structure is unchanged).
DEFAULT_SIGNAL_OFF_DBM: float = -200.0
# HB numerical-floor for the signal-frequency node voltage. Provenance: the
# committed full-sum floor criterion (``oa.psa_gain_artifact`` / full_sum_defs):
# known-noise designs sit at |V|≈1.7e-17…2.3e-16 V, the lowest real design at
# 5.7e-14 V; 1e-15 V splits them cleanly. Output below this is indistinguishable
# from numerical noise (⇒ no self-oscillation present).
DEFAULT_NOISE_FLOOR_V: float = 1e-15

AMPLIFICATION = "amplification"
OSCILLATION = "oscillation"
NON_CONVERGENT = "non-convergent"
INDETERMINATE = "indeterminate"


# --------------------------------------------------------------------------- #
#  Pure helpers (base-Python, unit-tested without ADS)                         #
# --------------------------------------------------------------------------- #
def log_amplitude_slope(
    signal_dBm,
    out_V,
    *,
    floor_V: float = DEFAULT_NOISE_FLOOR_V,
) -> float:
    """Least-squares slope ``d log10|out| / d log10|in|`` of the input sweep.

    ``signal_dBm`` is the swept signal *power* in dBm; the corresponding input
    *amplitude* is ``|V_in| ∝ 10**(P/20)``, so ``log10|V_in| = P/20`` (up to an
    additive constant that drops out of the slope). ``out_V`` is the matching
    signal-frequency output amplitude per point (``None``/non-finite/at-or-below
    ``floor_V`` points are dropped — a floored output carries no slope information
    and would bias an amplifier's slope toward the oscillator value 0).

    Slope ≈ 1 ⇒ output tracks input (amplifier); slope ≈ 0 ⇒ output pinned
    (oscillator). Returns ``nan`` if fewer than two usable points remain or the
    inputs are degenerate (identical x).
    """

    xs: list[float] = []
    ys: list[float] = []
    for p, v in zip(signal_dBm, out_V):
        if v is None:
            continue
        v = float(v)
        if not math.isfinite(v) or v <= float(floor_V):
            continue
        xs.append(float(p) / 20.0)
        ys.append(math.log10(v))
    n = len(xs)
    if n < 2:
        return float("nan")
    sx = math.fsum(xs)
    sy = math.fsum(ys)
    sxx = math.fsum(x * x for x in xs)
    sxy = math.fsum(x * y for x, y in zip(xs, ys))
    denom = n * sxx - sx * sx
    if abs(denom) < 1e-30:
        return float("nan")
    return (n * sxy - sx * sy) / denom


def classify_amplification(regime: str, tests_agree: bool, *, probed: bool = True) -> str:
    """Binary AMPLIFICATION / OSCILLATION verdict from the detector category.

    * unprobed (``probed=False``) or the two operational tests disagree
      (``tests_agree=False``) ⇒ :data:`INDETERMINATE` — the cheap checks do not
      concur; do not assert a verdict.
    * ``"non-convergent"`` ⇒ :data:`NON_CONVERGENT` — the driven HB diverged
      (the near-threshold instability proxy), neither a clean amplifier nor a
      directly-observed oscillation.
    * ``"lasing"`` ⇒ :data:`OSCILLATION`.
    * ``"amplifier"`` / ``"passive"`` ⇒ :data:`AMPLIFICATION` (stable,
      input-tracking forced response; the gain magnitude is reported separately).
    """

    if not probed:
        return INDETERMINATE
    if regime == "non-convergent":
        return NON_CONVERGENT
    if not tests_agree:
        return INDETERMINATE
    if regime == "lasing":
        return OSCILLATION
    return AMPLIFICATION


def pumped_linearized_stability_available() -> tuple[bool, str]:
    """Whether the rigorous pumped-linearized K/μ stability factor is accessible.

    The Rollett ``K`` / Edwards-Sinsky ``μ`` must be formed from the
    **pumped-linearized** (large-signal small-signal) conversion S-parameters of
    the two-port around the operating point. This framework's only run path is the
    flat-deck ``hpeesofsim`` bridge, which performs the two-tone HB and exposes
    node-voltage harmonic mixes only — it has no LSSP / conversion-matrix analysis
    and emits no pumped two-port S-matrix. So K/μ is **not cleanly accessible**
    here and the verdict rests on the input-dependence + signal-off pair.

    Returns ``(available, reason)``; ``available`` is ``False`` on this framework.
    """

    return (
        False,
        "pumped-linearized conversion S-parameters are not exposed by the "
        "flat-deck hpeesofsim path (node-voltage harmonic mixes only; no "
        "LSSP/conversion-matrix analysis), so Rollett K / mu cannot be formed; "
        "verdict rests on the input-dependence + signal-off pair.",
    )


# --------------------------------------------------------------------------- #
#  Result type                                                                #
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class AmplificationCheck:
    """One operating point's AMPLIFICATION-vs-OSCILLATION verdict + evidence.

    ``classification`` is the binary verdict (:data:`AMPLIFICATION` /
    :data:`OSCILLATION` / :data:`NON_CONVERGENT` / :data:`INDETERMINATE`);
    ``regime`` is the underlying loss-invariant detector category
    (``passive | amplifier | lasing | non-convergent | unprobed``).
    """

    classification: str
    regime: str
    converged: bool
    # INPUT-DEPENDENCE test (primary).
    input_dependence_slope: float
    input_sweep_dBm: tuple[float, ...]
    input_sweep_out_V: tuple[float, ...]
    # SIGNAL-OFF test (direct oscillation probe).
    r_A: float
    out_on_ref_V: float
    signal_off_out_V: float
    noise_floor_V: float
    signal_off_above_floor: bool
    signal_off_converged: bool
    # Agreement of the two operational tests + the rigorous K/μ status.
    tests_agree: bool
    is_laser_input_independence: bool | None
    is_laser_gain_linearity: bool | None
    k_mu_available: bool
    k_mu_note: str
    # Context carried from the reused full-sum sweep (NOT a lasing decision).
    boost_dB: float
    g_mod_in_dB: float
    passive_S21_dB: float
    idler_over_sig: float
    peak_phase_deg: float
    pump_power_dBm: float
    signal_ref_dBm: float
    signal_off_dBm: float
    f_signal_Hz: float
    f_pump_Hz: float
    reason: str
    note: str = ""

    def to_dict(self) -> dict:
        return {
            "classification": self.classification,
            "regime": self.regime,
            "converged": self.converged,
            "input_dependence_slope": self.input_dependence_slope,
            "input_sweep_dBm": list(self.input_sweep_dBm),
            "input_sweep_out_V": list(self.input_sweep_out_V),
            "r_A": self.r_A,
            "out_on_ref_V": self.out_on_ref_V,
            "signal_off_out_V": self.signal_off_out_V,
            "noise_floor_V": self.noise_floor_V,
            "signal_off_above_floor": self.signal_off_above_floor,
            "signal_off_converged": self.signal_off_converged,
            "tests_agree": self.tests_agree,
            "is_laser_input_independence": self.is_laser_input_independence,
            "is_laser_gain_linearity": self.is_laser_gain_linearity,
            "k_mu_available": self.k_mu_available,
            "k_mu_note": self.k_mu_note,
            "boost_dB": self.boost_dB,
            "g_mod_in_dB": self.g_mod_in_dB,
            "passive_S21_dB": self.passive_S21_dB,
            "idler_over_sig": self.idler_over_sig,
            "peak_phase_deg": self.peak_phase_deg,
            "pump_power_dBm": self.pump_power_dBm,
            "signal_ref_dBm": self.signal_ref_dBm,
            "signal_off_dBm": self.signal_off_dBm,
            "f_signal_Hz": self.f_signal_Hz,
            "f_pump_Hz": self.f_pump_Hz,
            "reason": self.reason,
            "note": self.note,
        }


# --------------------------------------------------------------------------- #
#  Internal HB-driving helpers (deferred ADS imports)                         #
# --------------------------------------------------------------------------- #
def _spec_scalar(spec, name: str) -> float:
    return float(next(v.value for v in spec.var_block.variables if v.name == name))


def _run_point(spec, run_dir, tag, *, signal_dBm, phase_deg, f_s, f_p, timeout_s):
    """One pump-on HB run; return ``|Σ V_(m,n)|`` (folding-family sum) at n_output.

    ``signal_dBm`` overrides the injected signal power; ``phase_deg`` bakes the
    pump phase. Returns ``None`` if the HB did not complete (non-convergence /
    error is data, handled by the caller), never raises on a sim failure.
    """

    from tvtl_ads_builder_v2.analysis.hb_dataset import read_hb_dataset
    from tvtl_ads_builder_v2.device_spec import with_var_overrides
    from tvtl_ads_builder_v2.netlist import emit_flat_hb_netlist
    from tvtl_ads_builder_v2.simulate.runner import run_hpeesofsim

    s = with_var_overrides(
        spec,
        {"signal_power_dBm": float(signal_dBm), "pump_phase_rad": math.radians(float(phase_deg))},
    )
    ws = Path(run_dir) / tag
    ws.mkdir(parents=True, exist_ok=True)
    deck = ws / f"{tag}.ckt"
    deck.write_text(emit_flat_hb_netlist(s), encoding="utf-8")
    res = run_hpeesofsim(deck, timeout_s=timeout_s)
    if res["status"] != "completed":
        return None
    ds = read_hb_dataset(res["dataset_path"], (f_s, f_p))
    return abs(full_family_phasor_sum(ds.node_voltage("n_output")))


def _idler_over_sig_at_peak(sweep) -> float:
    """``|V(-1,1)| / |V(1,0)|`` at the full-sum peak phase, from the sweep rows."""

    try:
        row = next(r for r in sweep.rows if r["phase_deg"] == sweep.peak_phase_deg)
    except StopIteration:
        return float("nan")
    mixes = list(sweep.family_mixes)
    re, im = row["family_re"], row["family_im"]

    def _mag(mn):
        if tuple(mn) not in mixes:
            return None
        i = mixes.index(tuple(mn))
        return abs(complex(re[i], im[i]))

    sig = _mag((1, 0))
    idl = _mag((-1, 1))
    if sig and idl and sig > 0.0:
        return idl / sig
    return float("nan")


# --------------------------------------------------------------------------- #
#  Public runner                                                              #
# --------------------------------------------------------------------------- #
def run_amplification_check(
    spec,
    run_dir: str | Path,
    *,
    signal_ref_dBm: float = DEFAULT_SIGNAL_REF_DBM,
    input_sweep_dBm=DEFAULT_INPUT_SWEEP_DBM,
    signal_off_dBm: float = DEFAULT_SIGNAL_OFF_DBM,
    off_pump_power_dBm: float = -100.0,
    noise_floor_V: float = DEFAULT_NOISE_FLOOR_V,
    phases_deg=DEFAULT_PHASES_DEG,
    timeout_s: int = 1200,
    r_a_threshold: float = R_A_LASER_THRESHOLD,
    slope_threshold: float = SLOPE_LASER_THRESHOLD,
) -> AmplificationCheck:
    """Run the cheap AMPLIFICATION-vs-OSCILLATION checks on ``spec`` at its
    operating point (baked pump power; pump phase peaked here).

    Procedure (all via the reused committed machinery):

    1. **Reference + peak phase.** Override the signal level to ``signal_ref_dBm``
       and run :func:`...oa.psa_gain_artifact.run_full_sum_gain_sweep` (pump-off
       reference + 6-phase pump-on degenerate folding-family sweep). This supplies
       the PSA-peak pump phase, the reference "signal on" output
       ``out_on = |Σ V|(peak)``, the boost / passive ``|S21|`` / out-in gain, and
       the idler-to-signal ratio — no rebuild, reuse only. A non-convergent driven
       sweep returns a ``non-convergent`` verdict (the instability proxy).
    2. **INPUT-DEPENDENCE (primary).** Run each ``input_sweep_dBm`` point pump-on
       at the peak phase (reusing the reference point) and fit
       :func:`log_amplitude_slope`.
    3. **SIGNAL-OFF (oscillation probe).** Run pump-on / signal at
       ``signal_off_dBm`` at the peak phase; ``r_A = out_off / out_on``; compare
       ``out_off`` to ``noise_floor_V``.
    4. **Classify.** Feed ``r_A`` (test A) and the slope (test B) to the committed
       loss-invariant detector and map its category to the binary verdict
       (:func:`classify_amplification`). K/μ status from
       :func:`pumped_linearized_stability_available`.

    Requires a reachable ``hpeesofsim`` (run under the ADS Python interpreter).
    """

    run_dir = Path(run_dir)
    pump_dBm = _spec_scalar(spec, "pump_power_dBm")
    f_s = _spec_scalar(spec, "f_signal")
    f_p = _spec_scalar(spec, "f_pump")
    k_mu_available, k_mu_note = pumped_linearized_stability_available()

    from tvtl_ads_builder_v2.device_spec import with_var_overrides

    # All measurements at the reference small-signal drive (so the sweep peak
    # phase, the r_A denominator, and the input sweep share one operating level).
    spec_ref = with_var_overrides(spec, {"signal_power_dBm": float(signal_ref_dBm)})

    def _nonconv(reason: str, note: str) -> AmplificationCheck:
        return AmplificationCheck(
            classification=NON_CONVERGENT, regime="non-convergent", converged=False,
            input_dependence_slope=float("nan"),
            input_sweep_dBm=tuple(float(p) for p in input_sweep_dBm),
            input_sweep_out_V=tuple(float("nan") for _ in input_sweep_dBm),
            r_A=float("nan"), out_on_ref_V=float("nan"), signal_off_out_V=float("nan"),
            noise_floor_V=float(noise_floor_V), signal_off_above_floor=False,
            signal_off_converged=False, tests_agree=True,
            is_laser_input_independence=None, is_laser_gain_linearity=None,
            k_mu_available=k_mu_available, k_mu_note=k_mu_note,
            boost_dB=float("nan"), g_mod_in_dB=float("nan"), passive_S21_dB=float("nan"),
            idler_over_sig=float("nan"), peak_phase_deg=float("nan"),
            pump_power_dBm=pump_dBm, signal_ref_dBm=float(signal_ref_dBm),
            signal_off_dBm=float(signal_off_dBm), f_signal_Hz=f_s, f_pump_Hz=f_p,
            reason=reason, note=note,
        )

    # 1. Reference full-sum sweep (peak phase + reference "signal on" output).
    try:
        sw = run_full_sum_gain_sweep(
            spec_ref, run_dir / "sweep",
            phases_deg=phases_deg, off_pump_power_dBm=off_pump_power_dBm,
            timeout_s=timeout_s,
        )
    except Exception as exc:  # noqa: BLE001 -- non-convergence is the instability proxy
        return _nonconv(
            "driven HB non-convergent at the operating point (instability onset "
            "proxy): output diverges with the signal on",
            f"{type(exc).__name__}: {exc}",
        )

    peak_phase = float(sw.peak_phase_deg)
    out_on_ref = math.sqrt(2.0 * 50.0 * sw.pout_mod_W) if sw.pout_mod_W > 0 else 0.0
    boost_dB = float(sw.g_mod_unmod_dB)
    g_mod_in_dB = float(sw.g_mod_in_dB)
    passive_S21_dB = (
        10.0 * math.log10(sw.pout_unmod_W / sw.p_avs_W)
        if sw.pout_unmod_W > 0 and sw.p_avs_W > 0 else float("-inf")
    )
    idler_over_sig = _idler_over_sig_at_peak(sw)

    # 2. INPUT-DEPENDENCE sweep at the peak phase (reuse the reference point).
    out_sweep: list[float] = []
    for s_dBm in input_sweep_dBm:
        if abs(float(s_dBm) - float(signal_ref_dBm)) < 1e-9:
            out_sweep.append(out_on_ref)
            continue
        tag = "in_" + str(int(round(float(s_dBm)))).replace("-", "m")
        v = _run_point(spec_ref, run_dir, tag, signal_dBm=float(s_dBm),
                       phase_deg=peak_phase, f_s=f_s, f_p=f_p, timeout_s=timeout_s)
        out_sweep.append(float("nan") if v is None else float(v))
    slope = log_amplitude_slope(input_sweep_dBm, out_sweep, floor_V=noise_floor_V)

    # 3. SIGNAL-OFF oscillation probe (pump on, signal ≈ off, peak phase).
    out_off = _run_point(
        spec_ref, run_dir, "sigoff_pumpon", signal_dBm=float(signal_off_dBm),
        phase_deg=peak_phase, f_s=f_s, f_p=f_p, timeout_s=timeout_s,
    )
    signal_off_converged = out_off is not None
    signal_off_out_V = float(out_off) if signal_off_converged else float("nan")
    signal_off_above_floor = signal_off_converged and signal_off_out_V > float(noise_floor_V)
    r_A = (
        signal_off_out_V / out_on_ref
        if (signal_off_converged and out_on_ref > 0.0) else float("nan")
    )

    # 4. Classify.
    if not signal_off_converged:
        # Autonomous (no-signal) pumped HB did not converge while the driven sweep
        # did: suggestive of self-oscillation, but r_A is unmeasurable, so the
        # cheap pair cannot certify a verdict -> indeterminate (honest).
        return AmplificationCheck(
            classification=INDETERMINATE, regime="unprobed", converged=True,
            input_dependence_slope=float(slope),
            input_sweep_dBm=tuple(float(p) for p in input_sweep_dBm),
            input_sweep_out_V=tuple(out_sweep),
            r_A=float("nan"), out_on_ref_V=float(out_on_ref),
            signal_off_out_V=float("nan"), noise_floor_V=float(noise_floor_V),
            signal_off_above_floor=False, signal_off_converged=False,
            tests_agree=False, is_laser_input_independence=None,
            is_laser_gain_linearity=(None if math.isnan(slope) else slope < slope_threshold),
            k_mu_available=k_mu_available, k_mu_note=k_mu_note,
            boost_dB=boost_dB, g_mod_in_dB=g_mod_in_dB, passive_S21_dB=passive_S21_dB,
            idler_over_sig=idler_over_sig, peak_phase_deg=peak_phase,
            pump_power_dBm=pump_dBm, signal_ref_dBm=float(signal_ref_dBm),
            signal_off_dBm=float(signal_off_dBm), f_signal_Hz=f_s, f_pump_Hz=f_p,
            reason="signal-off pump-on HB did not converge (autonomous problem "
                   "unstable -- suggestive of oscillation, but r_A unmeasurable so "
                   "the cheap checks cannot certify a verdict)",
            note="signal-off run non-convergent",
        )

    reg = classify_regime_loss_invariant(
        out_signal_off=signal_off_out_V, out_signal_on=out_on_ref,
        input_scaling_slope=(None if math.isnan(slope) else slope),
        enhancement_dB=boost_dB, idler_over_sig=idler_over_sig,
        converged=True, r_a_threshold=r_a_threshold, slope_threshold=slope_threshold,
    )
    classification = classify_amplification(reg.category, reg.tests_agree, probed=True)
    return AmplificationCheck(
        classification=classification, regime=reg.category, converged=True,
        input_dependence_slope=float(slope),
        input_sweep_dBm=tuple(float(p) for p in input_sweep_dBm),
        input_sweep_out_V=tuple(out_sweep),
        r_A=float(r_A), out_on_ref_V=float(out_on_ref),
        signal_off_out_V=float(signal_off_out_V), noise_floor_V=float(noise_floor_V),
        signal_off_above_floor=bool(signal_off_above_floor),
        signal_off_converged=True, tests_agree=bool(reg.tests_agree),
        is_laser_input_independence=reg.is_laser_input_independence,
        is_laser_gain_linearity=reg.is_laser_gain_linearity,
        k_mu_available=k_mu_available, k_mu_note=k_mu_note,
        boost_dB=boost_dB, g_mod_in_dB=g_mod_in_dB, passive_S21_dB=passive_S21_dB,
        idler_over_sig=idler_over_sig, peak_phase_deg=peak_phase,
        pump_power_dBm=pump_dBm, signal_ref_dBm=float(signal_ref_dBm),
        signal_off_dBm=float(signal_off_dBm), f_signal_Hz=f_s, f_pump_Hz=f_p,
        reason=reg.reason,
    )


