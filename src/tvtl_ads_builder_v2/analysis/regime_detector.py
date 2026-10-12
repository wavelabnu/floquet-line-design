"""Loss-invariant operating-regime detector grounded in the physical definition.

The legacy PSA-ratio classifier (:mod:`tvtl.reporting.regime_category` and the
associated diagnostic scripts) decides *lasing* from the steady-state pumped/unpumped gain ratio
("PSA peak > 15-20 dB => lasing"). That ratio is **reference-relative**: its denominator
is the *unpumped* output, which collapses with device/board loss, cell count and matching.
A loss sweep measured the unpumped reference dropping ~45 dB across a
0.5 -> 5 ohm branch-series-R change at fixed 20 dBm pump, which inflated the PSA "gain"
+31 -> +83 dB and mislabelled the SAME physical regime as progressively "more lasing".
So PSA magnitude cannot separate amplification from oscillation across loss/length.

This detector classifies the **laser vs stable** distinction from the PHYSICAL definition
of oscillation, loss-INVARIANTLY (it never divides by the moving unpumped reference):

* ``lasing``    -- self-sustained oscillation: a steady output persists as the signal
                   input -> 0 (the homogeneous/input-independent solution).
* ``amplifier`` -- stable, input-proportional output WITH net parametric enhancement.
* ``passive``   -- stable, input-proportional output, no parametric enhancement.

Two INDEPENDENT operational tests feed the laser/stable decision; they must AGREE:

A. INPUT-INDEPENDENCE (primary).  ``r_A = |out(signal OFF)| / |out(signal ON)|`` with the
   pump ON in both, at the SAME loss/config. Self-sustained output -> ``r_A ~ 1`` (laser);
   output that tracks the input -> ``r_A ~ 0`` (stable). The common loss factor cancels
   between numerator and denominator -- this is the exact mechanism that broke PSA, removed.

B. GAIN-LINEARITY (cross-check).  ``slope = d log|out| / d log|in|`` over a small-signal
   input sweep at fixed pump. Output proportional to input -> ``slope ~ 1`` (stable);
   pump-set, input-independent output -> ``slope ~ 0`` (laser).

Thresholds are the physical MIDPOINTS of these definitions (``r_A`` at 0.5; ``slope`` at
0.5) -- NOT tuned to experimental or prior-label data.

The steady-state GAIN is reported SEPARATELY and is NEVER used to decide lasing (that was
the PSA error). The passive/amplifier split (only meaningful once classified stable) keeps
the existing physical amplifier definition: net parametric enhancement above the passive
band AND a parametric idler present.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

__all__ = [
    "RegimeReading",
    "classify_regime_loss_invariant",
    "R_A_LASER_THRESHOLD",
    "SLOPE_LASER_THRESHOLD",
    "PASSIVE_BAND_DB",
    "IDLER_MIN",
]

# --- thresholds: physical midpoints of the regime definitions (NOT data-tuned) --------- #
# r_A in [0, 1]: 0 = output tracks input (stable), 1 = output self-sustained (laser).
R_A_LASER_THRESHOLD = 0.5
# input-scaling slope: 1 = output proportional to input (stable), 0 = input-independent (laser).
SLOPE_LASER_THRESHOLD = 0.5
# passive/amplifier boundary (existing physical definition, kept; not the lasing decision):
# net parametric enhancement (pumped vs unpumped, at the SAME config) above this band.
PASSIVE_BAND_DB = 1.0
# a real parametric amplifier produces a degenerate idler comparable to the signal.
IDLER_MIN = 0.05


@dataclass(frozen=True)
class RegimeReading:
    """A loss-invariant regime verdict with the two operational test results.

    ``category`` is one of ``passive | amplifier | lasing | non-convergent``.
    ``tests_agree`` is whether the two independent laser/stable tests concur; a
    disagreement is surfaced (the caller's validation gate must reject it).
    """

    category: str
    reason: str
    is_laser_input_independence: bool | None   # test A verdict (None if not measured)
    is_laser_gain_linearity: bool | None       # test B verdict (None if not measured)
    tests_agree: bool
    r_A: float | None                          # |out(sig off)| / |out(sig on)|
    input_scaling_slope: float | None
    enhancement_dB: float | None               # pumped/unpumped enhancement (reported only)
    idler_over_sig: float | None


def classify_regime_loss_invariant(
    *,
    out_signal_off: float | None,
    out_signal_on: float | None,
    input_scaling_slope: float | None = None,
    enhancement_dB: float | None = None,
    idler_over_sig: float | None = None,
    converged: bool = True,
    r_a_threshold: float = R_A_LASER_THRESHOLD,
    slope_threshold: float = SLOPE_LASER_THRESHOLD,
    passive_band_dB: float = PASSIVE_BAND_DB,
    idler_min: float = IDLER_MIN,
) -> RegimeReading:
    """Classify a self-pumped design point loss-invariantly.

    Parameters
    ----------
    out_signal_off, out_signal_on :
        |output at the signal frequency| with the SIGNAL source off (-> -100 dBm) and on
        (reference small-signal drive), BOTH with the pump ON, at the same loss/config.
        Test A: ``r_A = out_signal_off / out_signal_on``.
    input_scaling_slope :
        ``d log|out| / d log|in|`` from a small-signal input sweep at fixed pump (test B).
        ``None`` runs test A only (and ``tests_agree`` is reported True vacuously with a
        note, but a validation gate should require B to be supplied).
    enhancement_dB :
        The pumped-vs-unpumped enhancement (PSA-style), REPORTED ONLY and used solely for
        the passive/amplifier boundary among already-stable points -- never for lasing.
    idler_over_sig :
        |idler| / |signal| at the reference input (degenerate idler presence).
    converged :
        False when the HB solution did not converge -- treated as non-convergent
        (deep-runaway/unstable), NOT silently mis-handled as a stable point.
    """

    # --- non-convergence: data, classified consistently (deep runaway / unstable) ------ #
    if not converged:
        return RegimeReading(
            "non-convergent",
            "HB did not converge -> deep-runaway / unstable (not a stable solution)",
            None, None, True, None, None, enhancement_dB, idler_over_sig,
        )

    if out_signal_off is None or out_signal_on is None:
        raise ValueError("out_signal_off and out_signal_on are required (test A inputs).")
    on = float(out_signal_on)
    off = float(out_signal_off)
    if not (math.isfinite(on) and math.isfinite(off)):
        raise ValueError("out_signal_on/off must be finite.")
    if on <= 0.0:
        # No output even with the signal on: degenerate/dead point -> passive.
        return RegimeReading(
            "passive", "no output with signal on (dead point)",
            None, None, True, None, input_scaling_slope, enhancement_dB, idler_over_sig,
        )

    # --- Test A: input-independence (primary, loss-invariant) -------------------------- #
    r_A = off / on
    is_laser_A = r_A > r_a_threshold

    # --- Test B: gain-linearity slope (cross-check) ------------------------------------ #
    is_laser_B: bool | None
    if input_scaling_slope is None:
        is_laser_B = None
        agree = True  # vacuous; the validation gate must require B for full trust
    else:
        slope = float(input_scaling_slope)
        is_laser_B = slope < slope_threshold
        agree = (is_laser_A == is_laser_B)

    is_laser = is_laser_A  # primary test decides; agreement is reported for the gate

    if is_laser:
        return RegimeReading(
            "lasing",
            f"input-independent output: r_A={r_A:.3f} > {r_a_threshold} (output persists as "
            f"signal -> 0)" + ("" if is_laser_B is None else
            f"; slope={input_scaling_slope:.3f} {'<' if is_laser_B else '>='} {slope_threshold} "
            f"({'agrees' if agree else 'DISAGREES'})"),
            is_laser_A, is_laser_B, agree, r_A, input_scaling_slope, enhancement_dB, idler_over_sig,
        )

    # --- stable: passive vs amplifier (existing physical definition; NOT a lasing test) - #
    enh = float(enhancement_dB) if enhancement_dB is not None else float("nan")
    ios = float(idler_over_sig) if idler_over_sig is not None else float("nan")
    is_amp = math.isfinite(enh) and enh > passive_band_dB and math.isfinite(ios) and ios > idler_min
    cat = "amplifier" if is_amp else "passive"
    reason = (
        f"stable (r_A={r_A:.3f} <= {r_a_threshold}"
        + ("" if is_laser_B is None else f", slope={input_scaling_slope:.3f} "
           f"{'>=' if not is_laser_B else '<'} {slope_threshold}, "
           f"{'agree' if agree else 'DISAGREE'}")
        + "); "
        + (f"enhancement {enh:.2f} dB > {passive_band_dB} dB and idler/sig {ios:.3f} > {idler_min} "
           "(net parametric gain)" if is_amp else
           f"enhancement {enh:.2f} dB / idler/sig {ios:.3f} below the amplifier band "
           "(no net parametric gain)")
    )
    return RegimeReading(cat, reason, is_laser_A, is_laser_B, agree, r_A,
                         input_scaling_slope, enhancement_dB, idler_over_sig)


