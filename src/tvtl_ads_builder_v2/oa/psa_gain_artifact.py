"""PSA parametric-gain measurement + OA-workspace gain artifact.

Bakes the phase-sensitive-amplifier (PSA) gain measurement into the OA-emit
path so a generated workspace carries the materialised gain-vs-phase figure
the manuscript reports, not just the schematic.

The PSA gain is the coherent node-voltage ratio at ``n_output`` of
(signal + degenerate idler) ON over signal OFF, in dB -- the exact definition
that produces the locked A-3.1 +9.7116 dB anchor
(``src/tvtl/optimization/stage4_validation.py``):

    V(1,0)   = n_output voltage at HB mix (1, 0)   = signal band (f_signal)
    V(-1,1)  = n_output voltage at HB mix (-1, 1)  = degenerate idler
               at f_pump - f_signal (= f_signal for the degenerate self-pump)
    ON  = pump at full calibrated power, swept pump phase
    OFF = pump_power_dBm = -100 (denominator; phase-independent)

    coh0(phi)  = 20*log10( |V(1,0)_on + V(-1,1)_on| / |V(1,0)_off| )   <- PSA gain
    mix10(phi) = 20*log10( |V(1,0)_on|              / |V(1,0)_off| )   <- phase-invariant

Two public entry points:

* :func:`run_psa_gain_sweep` -- run the HB phase sweep through the flat-deck
  ``hpeesofsim`` path and return the per-phase coh0/mix10 plus phasors and the
  peak. Reused by the ADS-verification driver and the anchor check.
* :func:`write_psa_gain_artifact` -- run (or accept) the sweep and write
  ``data/psa_gain.ds`` + ``psa_gain_equation.txt`` (+ a PNG when matplotlib is
  present) into an emitted OA workspace.

``keysight.ads.dataset`` and the ``hpeesofsim`` bridge are imported lazily
inside the functions, so importing this module never requires ADS (the pure
:func:`coherent_gain_dB` helper is base-Python and unit-tested).
"""

from __future__ import annotations

import cmath
import math
from dataclasses import dataclass
from pathlib import Path

__all__ = [
    "PsaGainSweep",
    "coherent_gain_dB",
    "signal_band_gain_dB",
    "run_psa_gain_sweep",
    "write_psa_gain_artifact",
    "DEFAULT_PHASES_DEG",
    # Full-harmonic-sum (degenerate folding family) additions.
    "degenerate_folding_mixes",
    "full_family_phasor_sum",
    "full_sum_gain_dB",
    "input_reflection_sq",
    "FullSumGainSweep",
    "run_full_sum_gain_sweep",
]

DEFAULT_PHASES_DEG: tuple[float, ...] = (0.0, 60.0, 120.0, 180.0, 240.0, 300.0)


def coherent_gain_dB(v_sig_on: complex, v_idler_on: complex, v_sig_off: complex) -> float:
    """PSA coherent gain in dB: 20*log10(|v_sig_on + v_idler_on| / |v_sig_off|)."""

    num = abs(complex(v_sig_on) + complex(v_idler_on))
    den = abs(complex(v_sig_off))
    return 20.0 * math.log10(num / den)


def signal_band_gain_dB(v_sig_on: complex, v_sig_off: complex) -> float:
    """Phase-invariant signal-band gain in dB: 20*log10(|v_sig_on| / |v_sig_off|)."""

    return 20.0 * math.log10(abs(complex(v_sig_on)) / abs(complex(v_sig_off)))


# --------------------------------------------------------------------------- #
#  Full degenerate folding-family sum (full-harmonic-sum gain)                 #
# --------------------------------------------------------------------------- #
# When the pump is the exact second harmonic of the signal (f_pump = 2 f_signal)
# the parametric mixing is DEGENERATE: a two-tone product (m, n) sits at
# frequency m*f_signal + n*f_pump = (m + 2n)*f_signal, so it lands on the signal
# frequency f_signal whenever
#
#       m + 2 n = 1.
#
# The family is signal (1,0), idler (-1,1), then the higher folding terms
# (3,-1), (-3,2), (5,-2), (-5,3), ... A spectrum analyser at f_signal sums the
# WHOLE family coherently. The locked 2-term coh_pk (:func:`coherent_gain_dB`,
# signal + idler only) is the leading pair of that sum. The helpers below add the
# full-family phasor sum -- the SA-comparable quantity -- WITHOUT changing the
# 2-term path: :func:`coherent_gain_dB` / :func:`signal_band_gain_dB` /
# :func:`run_psa_gain_sweep` are untouched so the +9.7116 dB anchor stays intact.


def degenerate_folding_mixes(spectrum) -> tuple[tuple[int, int], ...]:
    """The (m, n) mixing tuples present in an HB ``spectrum`` with m + 2n = 1.

    For f_pump = 2 f_signal these are exactly the products that fold onto
    f_signal. Returned ordered by ascending folding order |m| + |n| (then |n|),
    so a running phasor sum walks the family signal -> idler -> higher terms.
    Truncation-limited: only tuples the HB actually solved for are returned, so
    the list directly exposes which folding terms the HB order captured.
    """

    family: list[tuple[int, int]] = []
    for harmonic in spectrum.harmonics:
        mix = harmonic.mix
        if len(mix) == 2 and (int(mix[0]) + 2 * int(mix[1]) == 1):
            family.append((int(mix[0]), int(mix[1])))
    family.sort(key=lambda mn: (abs(mn[0]) + abs(mn[1]), abs(mn[1])))
    return tuple(family)


def full_family_phasor_sum(spectrum, mixes=None) -> complex:
    """Coherent phasor sum ``Σ V_(m,n)`` over the degenerate folding family.

    ``mixes`` defaults to :func:`degenerate_folding_mixes`; pass an explicit
    ordered subsequence to form a partial sum (a convergence study walks the
    family one term at a time). Each tuple is looked up with ``spectrum.at_mix``.
    """

    if mixes is None:
        mixes = degenerate_folding_mixes(spectrum)
    return sum((complex(spectrum.at_mix(mn)) for mn in mixes), 0j)


def full_sum_gain_dB(v_family_sum: complex, v_sig_off: complex) -> float:
    """Full-family degenerate PSA gain in dB:

        20*log10( |Σ_(m+2n=1) V_(m,n)_on| / |V_(1,0)_off| ).

    Identical convention to :func:`coherent_gain_dB` (same denominator, same
    20*log10 voltage ratio, phase-peaked downstream); the ONLY change is that the
    numerator sums the WHOLE folding family rather than just signal + idler.
    Equals :func:`coherent_gain_dB` exactly when the family is {(1,0), (-1,1)}.
    """

    return 20.0 * math.log10(abs(complex(v_family_sum)) / abs(complex(v_sig_off)))


def input_reflection_sq(
    v_in: complex,
    p_avs_W: float,
    *,
    z0_ohm: float = 50.0,
    emf_phase_rad: float = 0.0,
) -> float:
    """``|Gamma_in|^2`` seen by the signal port at its injection node.

    DIAGNOSTIC ONLY -- context for the input match / coupler injection; NEVER a
    gain denominator. Derived purely from the HB node voltage
    ``v_in = V(n_input, f_signal)`` and the source available power ``p_avs_W``.

    The signal ``Port`` is a Thevenin source of EMF ``E`` behind ``Z0`` with the
    peak-phasor convention ``P_avs = |E|^2 / (8 Z0)`` (matched-load voltage
    ``E/2`` delivers ``P_avs``), so ``E = sqrt(8 Z0 P_avs)`` at phase
    ``emf_phase_rad`` (the deck injects the signal at phase 0). The node voltage
    is ``V_in = E (1 + Gamma) / 2``, hence ``Gamma = 2 V_in / E - 1``. Evaluate at
    pump OFF for the clean small-signal input match (passive: ``|Gamma| <= 1``).
    """

    e_s = cmath.rect(math.sqrt(8.0 * float(z0_ohm) * float(p_avs_W)), float(emf_phase_rad))
    gamma = 2.0 * complex(v_in) / e_s - 1.0
    return abs(gamma) ** 2


@dataclass(frozen=True)
class PsaGainSweep:
    """Result of a PSA phase sweep."""

    phases_deg: tuple[float, ...]
    rows: tuple[dict, ...]            # per-phase: phase_deg, coh0_dB, mix10_dB, phasors
    v_sig_off: complex
    peak_phase_deg: float
    peak_coh0_dB: float
    pump_power_dBm: float
    f_signal_Hz: float
    f_pump_Hz: float

    @property
    def swing_dB(self) -> float:
        vals = [r["coh0_dB"] for r in self.rows]
        return float(max(vals) - min(vals)) if vals else float("nan")

    def to_dict(self) -> dict:
        return {
            "phases_deg": list(self.phases_deg),
            "pump_power_dBm": self.pump_power_dBm,
            "f_signal_Hz": self.f_signal_Hz,
            "f_pump_Hz": self.f_pump_Hz,
            "v_sig_off_re": self.v_sig_off.real,
            "v_sig_off_im": self.v_sig_off.imag,
            "rows": [dict(r) for r in self.rows],
            "peak_phase_deg": self.peak_phase_deg,
            "peak_coh0_dB": self.peak_coh0_dB,
            "swing_dB": self.swing_dB,
        }


def _spec_freqs(spec) -> tuple[float, float]:
    fs = float(next(v.value for v in spec.var_block.variables if v.name == "f_signal"))
    fp = float(next(v.value for v in spec.var_block.variables if v.name == "f_pump"))
    return fs, fp


def run_psa_gain_sweep(
    spec,
    run_dir: str | Path,
    *,
    phases_deg=DEFAULT_PHASES_DEG,
    timeout_s: int = 900,
    off_pump_power_dBm: float = -100.0,
) -> PsaGainSweep:
    """Run the PSA phase sweep through the flat-deck ``hpeesofsim`` path.

    For each phase ``phi`` the pump-phase-overridden spec is emitted, simulated,
    and the (signal, idler) ON phasors read at ``n_output``; one OFF run (pump
    at ``off_pump_power_dBm``) supplies the phase-independent denominator.

    Requires a reachable ``hpeesofsim`` (run under the ADS Python interpreter).
    """

    from tvtl_ads_builder_v2.analysis.hb_dataset import read_hb_dataset
    from tvtl_ads_builder_v2.device_spec import with_var_overrides
    from tvtl_ads_builder_v2.netlist import emit_flat_hb_netlist
    from tvtl_ads_builder_v2.simulate.runner import run_hpeesofsim

    run_dir = Path(run_dir)
    fs, fp = _spec_freqs(spec)
    pump_dBm = float(next(v.value for v in spec.var_block.variables if v.name == "pump_power_dBm"))

    def _mixes(s, tag):
        ws = run_dir / tag
        ws.mkdir(parents=True, exist_ok=True)
        deck = ws / f"{tag}.ckt"
        deck.write_text(emit_flat_hb_netlist(s), encoding="utf-8")
        res = run_hpeesofsim(deck, timeout_s=timeout_s)
        if res["status"] != "completed":
            raise RuntimeError(f"{tag}: hpeesofsim {res.get('error_message')}")
        ds = read_hb_dataset(res["dataset_path"], (fs, fp))
        return (
            complex(ds.node_voltage("n_output").at_mix((1, 0))),
            complex(ds.node_voltage("n_output").at_mix((-1, 1))),
        )

    spec_off = with_var_overrides(spec, {"pump_power_dBm": off_pump_power_dBm})
    v_sig_off, _ = _mixes(spec_off, "off")

    rows: list[dict] = []
    for ph in phases_deg:
        spec_on = with_var_overrides(spec, {"pump_phase_rad": math.radians(float(ph))})
        v_sig_on, v_idler_on = _mixes(spec_on, f"on_{int(round(ph))}")
        rows.append({
            "phase_deg": float(ph),
            "coh0_dB": coherent_gain_dB(v_sig_on, v_idler_on, v_sig_off),
            "mix10_dB": signal_band_gain_dB(v_sig_on, v_sig_off),
            "v_sig_on_re": v_sig_on.real, "v_sig_on_im": v_sig_on.imag,
            "v_idler_on_re": v_idler_on.real, "v_idler_on_im": v_idler_on.imag,
        })

    peak = max(rows, key=lambda r: r["coh0_dB"])
    return PsaGainSweep(
        phases_deg=tuple(float(p) for p in phases_deg),
        rows=tuple(rows), v_sig_off=v_sig_off,
        peak_phase_deg=float(peak["phase_deg"]), peak_coh0_dB=float(peak["coh0_dB"]),
        pump_power_dBm=pump_dBm, f_signal_Hz=fs, f_pump_Hz=fp,
    )


# --------------------------------------------------------------------------- #
#  Full-family (degenerate folding) gain sweep                                 #
# --------------------------------------------------------------------------- #
def _p_out_watts(v_node: complex) -> float:
    """Power into a clean 50-ohm port at one harmonic: (1/2)|V|^2 / 50."""

    return 0.5 * abs(complex(v_node)) ** 2 / 50.0


def _dbmtow(p_dbm: float) -> float:
    """dBm -> watts (dbmtow(0) = 1e-3 W)."""

    return 10.0 ** ((float(p_dbm) - 30.0) / 10.0)


@dataclass(frozen=True)
class FullSumGainSweep:
    """Result of a degenerate full-folding-family PSA phase sweep.

    Carries BOTH the new full-family metric and the legacy 2-term coh_pk so a
    design's correction (full-sum minus 2-term) is a field difference. The
    numerator of the full metric sums every m+2n=1 product the HB solved for;
    the 2-term column reuses :func:`coherent_gain_dB` verbatim, so the +9.7116 dB
    anchor reproduces in ``peak_coh_2term_dB``.
    """

    phases_deg: tuple[float, ...]
    family_mixes: tuple[tuple[int, int], ...]
    rows: tuple[dict, ...]            # per phase: phasors, partial sums, coh_2term/full
    v_sig_off: complex               # n_output (1,0), pump off
    v_in_off: complex                # n_input  (1,0), pump off (for |Gamma_in|^2)
    p_avs_W: float
    pout_unmod_W: float              # 0.5|V(1,0)off|^2/50
    pout_mod_W: float                # 0.5|Sigma V|^2/50 at the full-sum peak phase
    gamma_in_sq: float               # |Gamma_in|^2 (pump off; diagnostic only)
    peak_phase_deg: float            # phase maximising the full-family sum
    peak_coh_full_dB: float          # G_mod/unmod (full-sum pump-on/off, phase-peaked)
    peak_coh_2term_dB: float         # legacy coh_pk (signal+idler, phase-peaked)
    peak_phase_2term_deg: float
    pump_power_dBm: float
    signal_power_dBm: float
    f_signal_Hz: float
    f_pump_Hz: float

    @property
    def g_mod_unmod_dB(self) -> float:
        """Full-sum pump-on/off gain (the SA-comparable headline)."""

        return self.peak_coh_full_dB

    @property
    def g_mod_in_dB(self) -> float:
        """Full-sum absolute gain vs available input power: 10log10(Pout_mod/P_avs)."""

        return 10.0 * math.log10(self.pout_mod_W / self.p_avs_W)

    @property
    def correction_dB(self) -> float:
        """Full-sum G_mod/unmod minus the legacy 2-term coh_pk (the correction)."""

        return self.peak_coh_full_dB - self.peak_coh_2term_dB

    def to_dict(self) -> dict:
        return {
            "phases_deg": list(self.phases_deg),
            "family_mixes": [list(mn) for mn in self.family_mixes],
            "rows": [dict(r) for r in self.rows],
            "v_sig_off_re": self.v_sig_off.real, "v_sig_off_im": self.v_sig_off.imag,
            "v_in_off_re": self.v_in_off.real, "v_in_off_im": self.v_in_off.imag,
            "p_avs_W": self.p_avs_W,
            "pout_unmod_W": self.pout_unmod_W,
            "pout_mod_W": self.pout_mod_W,
            "gamma_in_sq": self.gamma_in_sq,
            "peak_phase_deg": self.peak_phase_deg,
            "G_mod_unmod_dB": self.g_mod_unmod_dB,
            "G_mod_in_dB": self.g_mod_in_dB,
            "coh_pk_2term_dB": self.peak_coh_2term_dB,
            "peak_phase_2term_deg": self.peak_phase_2term_deg,
            "correction_dB": self.correction_dB,
            "pump_power_dBm": self.pump_power_dBm,
            "signal_power_dBm": self.signal_power_dBm,
            "f_signal_Hz": self.f_signal_Hz,
            "f_pump_Hz": self.f_pump_Hz,
        }


def run_full_sum_gain_sweep(
    spec,
    run_dir: str | Path,
    *,
    phases_deg=DEFAULT_PHASES_DEG,
    timeout_s: int = 900,
    off_pump_power_dBm: float = -100.0,
) -> FullSumGainSweep:
    """Run the degenerate full-folding-family PSA phase sweep (flat-deck hpeesofsim).

    Same OFF + per-phase-ON structure and phase convention as
    :func:`run_psa_gain_sweep`; the only difference is that each ON point reads
    the WHOLE m+2n=1 family at ``n_output`` (not just signal + idler) and the
    pump-off ``n_input`` voltage for the ``|Gamma_in|^2`` diagnostic. The legacy
    2-term coh_pk is computed alongside via :func:`coherent_gain_dB` for an
    apples-to-apples correction. Requires a reachable ``hpeesofsim`` (ADSPY).
    """

    from tvtl_ads_builder_v2.analysis.hb_dataset import read_hb_dataset
    from tvtl_ads_builder_v2.device_spec import with_var_overrides
    from tvtl_ads_builder_v2.netlist import emit_flat_hb_netlist
    from tvtl_ads_builder_v2.simulate.runner import run_hpeesofsim

    run_dir = Path(run_dir)
    fs, fp = _spec_freqs(spec)
    pump_dBm = float(next(v.value for v in spec.var_block.variables if v.name == "pump_power_dBm"))
    sig_dBm = float(next(v.value for v in spec.var_block.variables if v.name == "signal_power_dBm"))
    p_avs = _dbmtow(sig_dBm)

    def _dataset(s, tag):
        ws = run_dir / tag
        ws.mkdir(parents=True, exist_ok=True)
        deck = ws / f"{tag}.ckt"
        deck.write_text(emit_flat_hb_netlist(s), encoding="utf-8")
        res = run_hpeesofsim(deck, timeout_s=timeout_s)
        if res["status"] != "completed":
            raise RuntimeError(f"{tag}: hpeesofsim {res.get('error_message')}")
        return read_hb_dataset(res["dataset_path"], (fs, fp))

    # OFF reference (denominator + Gamma_in).
    ds_off = _dataset(with_var_overrides(spec, {"pump_power_dBm": off_pump_power_dBm}), "off")
    v_sig_off = complex(ds_off.node_voltage("n_output").at_mix((1, 0)))
    v_in_off = complex(ds_off.node_voltage("n_input").at_mix((1, 0)))
    gamma_in_sq = input_reflection_sq(v_in_off, p_avs)

    family_mixes: tuple[tuple[int, int], ...] = ()
    rows: list[dict] = []
    for ph in phases_deg:
        ds = _dataset(with_var_overrides(spec, {"pump_phase_rad": math.radians(float(ph))}),
                      f"on_{int(round(ph))}")
        out = ds.node_voltage("n_output")
        if not family_mixes:
            family_mixes = degenerate_folding_mixes(out)
        phasors = [complex(out.at_mix(mn)) for mn in family_mixes]
        partial = []
        acc = 0j
        for p in phasors:
            acc += p
            partial.append(abs(acc))
        v10 = complex(out.at_mix((1, 0)))
        vm11 = complex(out.at_mix((-1, 1)))
        full_sum = acc
        rows.append({
            "phase_deg": float(ph),
            "coh_2term_dB": coherent_gain_dB(v10, vm11, v_sig_off),
            "coh_full_dB": full_sum_gain_dB(full_sum, v_sig_off),
            "mix10_dB": signal_band_gain_dB(v10, v_sig_off),
            "family_re": [p.real for p in phasors],
            "family_im": [p.imag for p in phasors],
            "partial_sum_mag": partial,
            "full_sum_re": full_sum.real, "full_sum_im": full_sum.imag,
        })

    peak_full = max(rows, key=lambda r: r["coh_full_dB"])
    peak_2term = max(rows, key=lambda r: r["coh_2term_dB"])
    pout_mod = _p_out_watts(complex(peak_full["full_sum_re"], peak_full["full_sum_im"]))
    return FullSumGainSweep(
        phases_deg=tuple(float(p) for p in phases_deg),
        family_mixes=family_mixes, rows=tuple(rows),
        v_sig_off=v_sig_off, v_in_off=v_in_off, p_avs_W=p_avs,
        pout_unmod_W=_p_out_watts(v_sig_off), pout_mod_W=pout_mod,
        gamma_in_sq=gamma_in_sq,
        peak_phase_deg=float(peak_full["phase_deg"]),
        peak_coh_full_dB=float(peak_full["coh_full_dB"]),
        peak_coh_2term_dB=float(peak_2term["coh_2term_dB"]),
        peak_phase_2term_deg=float(peak_2term["phase_deg"]),
        pump_power_dBm=pump_dBm, signal_power_dBm=sig_dBm,
        f_signal_Hz=fs, f_pump_Hz=fp,
    )


_EQUATION_TEMPLATE = """\
PSA parametric-gain -- ADS Data Display equation
================================================
Dataset: data/psa_gain.ds   (independent axis: PumpPhase_deg)
Source spec: {label}   (f_signal = {fs_ghz:.4f} GHz, f_pump = {fp_ghz:.4f} GHz,
pump_power = {pump_dBm:.4f} dBm)

Definition (src/tvtl/optimization/stage4_validation.py -- the code that
produces the locked +9.7116 dB A-3.1 figure): a coherent NODE-VOLTAGE ratio at
n_output of (signal + degenerate idler) ON over signal OFF, in dB.

  V(1,0)  = n_output voltage at HB mix (1, 0)  = signal band f_signal
  V(-1,1) = n_output voltage at HB mix (-1, 1) = degenerate idler at f_pump - f_signal
  ON  = pump at full power, swept pump phase
  OFF = pump_power_dBm = -100 (denominator; phase-independent)

The dataset stores those probe phasors so a Data Display equation recomputes
the gain identically (paste into the Eqn region of a Data Display window):

  Gain_dB = 20*log10( mag(V_out_signal_on + V_out_idler_on) / mag(V_out_signal_off) )

Phase-invariant signal-band gain (cross-check):

  SignalBand_dB = 20*log10( mag(V_out_signal_on) / mag(V_out_signal_off) )

The column Gain_dB in psa_gain.ds is this equation already evaluated, so a plot
of Gain_dB vs PumpPhase_deg shows the PSA curve directly. Measured peak:
{peak_coh0:.4f} dB at PumpPhase_deg = {peak_phase:.0f} deg (swing {swing:.4f} dB).

NOTE: the definition uses the n_output node-voltage mixes only; any I_Probe
currents present in the HB dataset are NOT part of this parametric-gain figure.
"""


def write_psa_gain_artifact(
    spec,
    workspace_dir: str | Path,
    *,
    sweep: PsaGainSweep | None = None,
    run_dir: str | Path | None = None,
    phases_deg=DEFAULT_PHASES_DEG,
    make_png: bool = True,
    timeout_s: int = 900,
) -> dict:
    """Bake the PSA gain dataset + equation (+ PNG) into an emitted OA workspace.

    Writes into ``<workspace_dir>/data/psa_gain.ds`` an ADS dataset with an
    independent ``PumpPhase_deg`` axis and dependents ``Gain_dB``,
    ``SignalBandGain_dB``, and the ON/OFF probe phasors (so a Data Display
    equation can recompute ``Gain_dB``). Also writes ``psa_gain_equation.txt``
    (the exact equation + provenance) next to the workspace and, when
    matplotlib is available, ``Gain_dB_vs_phase.png``.

    ``sweep`` may be supplied to reuse an already-run sweep; otherwise the
    sweep is run via :func:`run_psa_gain_sweep` (``run_dir`` required then).

    Requires the ADS Python interpreter (``keysight.ads.dataset``). Returns a
    dict of written paths plus the peak/swing.
    """

    import numpy as np
    import pandas as pd

    workspace_dir = Path(workspace_dir)
    if sweep is None:
        if run_dir is None:
            raise ValueError("write_psa_gain_artifact requires run_dir when sweep is not supplied.")
        sweep = run_psa_gain_sweep(spec, run_dir, phases_deg=phases_deg, timeout_s=timeout_s)

    rows = sweep.rows
    phase = np.array([r["phase_deg"] for r in rows], dtype=float)
    v_sig_on = np.array([complex(r["v_sig_on_re"], r["v_sig_on_im"]) for r in rows])
    v_idl_on = np.array([complex(r["v_idler_on_re"], r["v_idler_on_im"]) for r in rows])
    gain_db = np.array([r["coh0_dB"] for r in rows], dtype=float)
    mix10_db = np.array([r["mix10_dB"] for r in rows], dtype=float)
    v_off = sweep.v_sig_off

    df = pd.DataFrame(
        {
            "Gain_dB": gain_db,
            "SignalBandGain_dB": mix10_db,
            "V_out_signal_on": v_sig_on,
            "V_out_idler_on": v_idl_on,
            "V_out_signal_off": np.full(phase.shape, v_off, dtype=complex),
        },
        index=pd.Index(phase, name="PumpPhase_deg"),
    )

    from keysight.ads import dataset as ds_mod

    ds_out = workspace_dir / "data" / "psa_gain.ds"
    ds_out.parent.mkdir(parents=True, exist_ok=True)
    if ds_out.exists():
        ds_out.unlink()
    with ds_mod.create_dataset(str(ds_out)) as ds:
        ds.create_varblock_from_dataframe("PSA_GAIN", df)

    eqn_out = workspace_dir.parent / "psa_gain_equation.txt"
    eqn_out.write_text(
        _EQUATION_TEMPLATE.format(
            label=getattr(spec, "label", "spec"),
            fs_ghz=sweep.f_signal_Hz / 1e9, fp_ghz=sweep.f_pump_Hz / 1e9,
            pump_dBm=sweep.pump_power_dBm,
            peak_coh0=sweep.peak_coh0_dB, peak_phase=sweep.peak_phase_deg,
            swing=sweep.swing_dB,
        ),
        encoding="utf-8",
    )

    png_out = None
    if make_png:
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt

            png_out = workspace_dir.parent / "Gain_dB_vs_phase.png"
            fig, ax = plt.subplots(figsize=(6, 4))
            ax.plot(phase, gain_db, "o-", label="Gain_dB (PSA coherent)")
            ax.plot(phase, mix10_db, "s--", color="green", label="SignalBand_dB (phase-invariant)")
            ax.axvline(sweep.peak_phase_deg, ls=":", color="red", lw=0.8)
            ax.set_xlabel("Signal-pump phase (deg)")
            ax.set_ylabel("Gain (dB)")
            ax.set_title(f"{getattr(spec, 'label', 'spec')} -- PSA gain vs phase (OA workspace)")
            ax.legend(fontsize=8)
            ax.grid(True, alpha=0.3)
            fig.tight_layout()
            fig.savefig(png_out, dpi=120)
            plt.close(fig)
        except Exception as exc:  # noqa: BLE001 -- PNG is optional
            png_out = f"skipped ({exc!r})"

    return {
        "dataset_path": str(ds_out),
        "equation_path": str(eqn_out),
        "png_path": str(png_out) if png_out is not None else None,
        "peak_coh0_dB": sweep.peak_coh0_dB,
        "peak_phase_deg": sweep.peak_phase_deg,
        "swing_dB": sweep.swing_dB,
    }


