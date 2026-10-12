"""Analytical-Floquet vs ADS-HB per-varactor current spectrum cross-check.

The I_Probe machinery
exposes every varactor's complex current spectrum from the live HB
``.ds``; the :mod:`tvtl.floquet.harmonics` module independently solves
the same physics analytically -- the Valagiannopoulos elastance
tridiagonal that yields charge harmonics ``Q_n`` and current harmonics
``I_n = j·(ω_s + n·ω_m)·Q_n`` for a series RLC with a sinusoidal
elastance ``S(t) = S0 + dS·cos(ω_m·t + φ)`` driven by
``v(t) = V_0·exp(+j·ω_s·t)``.

This module joins those two paths and reports, per cell and per
harmonic, the dB ratio ``20·log10(|I_n_ADS| / |I_n_analytical|)``. The
mapping between the analytical harmonic index ``n`` and the ADS
mixing tuple is set by the pump/signal frequency ratio: for the
paper's degenerate operating point ``f_m = 2·f_s`` (the
default), analytical harmonic ``n`` lands at ``(2n + 1)·f_s`` -- the
same frequency as ADS mix tuple ``(m, l) = (1, n)`` (whose product is
``m·f_s + l·f_m = (1 + 2n)·f_s``). The general mapping for an integer
``pump_signal_ratio = r`` is ``(m, l) = (1, n)`` with the analytical
harmonic n at ``(1 + r·n)·f_s``.

Varactor model extraction
-------------------------
The analytical solver takes ``(S0, dS, φ, R, L, ω_s, ω_m, N)``. The
Level-9 ``SDD2P`` carries the charge polynomial
``Q(v) = q1·v + q2·v² + ...`` with ``q[k] = C_{k-1} / k`` (the
:class:`tvtl.design.nonlinear_capacitor.ChargePolynomialModel` source
of truth, exposed via ``charge_polynomial_coefficients``). The small-
signal capacitance is ``C(v) = ∂Q/∂v = q1 + 2·q2·v``, so

    C0 = q1 (the static capacitance at v = 0)
    C1 = 2·q2 (the linear C-V slope ∂C/∂v, in F/V; signed)

Under a sinusoidal pump voltage ``v_pump(t) = Re(V_p·exp(j·ω_m·t))``
the capacitance is modulated to first order as
``C(t) ≈ C0 + C1·|V_p|·cos(ω_m·t + arg(V_p))``; the elastance series
is then ``S(t) = 1/C(t) ≈ S0 - (C1·V_p / C0²)·cos(ω_m·t + arg(V_p))``.
Matching ``S(t) = S0 + dS·cos(ω_m·t + φ)`` yields

    dS = |C1·V_p| / C0²    (positive magnitude)
    φ  = arg(V_p)    if C1 < 0     (the SMV1245 case)
    φ  = arg(V_p) + π    if C1 > 0

The SDD2P element does not carry an in-element series R or L --
``R = 0``, ``L = 0`` for the analytical comparison. The varactor
through which the cross-check runs is the varactor probe ``IPR_VAR_C{k}``
(in series between SDD2P pin 2 and the stub MLIN); ADS reports its
current at every ADS mixing tuple in the HB result. The analytical
``V_0`` is the signal-frequency phasor at the same varactor branch top
(``n_c{k}_b`` / SDD2P pin 1) read from the live HB.

Per-cell pump voltage
---------------------
The pump voltage ``V_p`` differs cell to cell because the through line
has loss and the cells partly mismatch the input. The cross-check
therefore reads ``V_p`` per cell from the HB dataset at the varactor
branch top, mix tuple ``(0, 1)`` (the pump fundamental). This is the
honest comparison: the analytical solver predicts ``I_n`` *given*
``V_p`` at the varactor, and ADS independently computes ``I_n``
*given* the actual delivered pump and signal. Disagreement therefore
reflects model-vs-simulator gaps, not analytical guesses about pump
delivery.

Output schema
-------------
:func:`compare_varactor_harmonics` returns a dict keyed by:

* ``harmonic_index`` -- ``tuple(int)``, ordered ``n = -N, ..., +N``.
* ``frequencies_Hz`` -- ``tuple(float)``, the analytical frequencies
  ``(1 + r·n)·f_s``.
* ``mix_tuples`` -- ``tuple(tuple(int, int))``, the corresponding
  ADS mix tuples ``(1, n)``.
* ``I_n_analytical`` -- ``tuple(complex)`` per harmonic.
* ``I_n_ADS`` -- ``tuple(complex)`` per harmonic at the ADS-side
  mix tuple (``0j`` if the dataset has no harmonic at that tuple).
* ``ratio_dB`` -- ``tuple(float)`` per harmonic,
  ``20·log10(|I_ADS| / |I_analytical|)`` (NaN if either is zero).
* ``max_abs_ratio_dB`` -- ``float``, max over harmonics where both
  sides are non-zero; NaN if no harmonic has a finite ratio.
* ``within_tolerance`` -- ``bool``, every harmonic's
  ``|ratio_dB|`` lies under its per-harmonic-band tolerance.

:func:`compare_cascade_harmonics` aggregates :func:`compare_varactor_harmonics`
across all cells of a cascade.
"""

from __future__ import annotations

import math
from typing import Callable, Mapping, Sequence

import numpy as np

from ..device_spec import DeviceSpec
from .hb_dataset import HBDataset
from .per_cell import CellMap, cell_count


# Per-harmonic-band default tolerance budget (dB):
# - Dominant (|n| <= 1): 0.5 dB
# - Secondary (|n| == 2): 1.0 dB
# - Higher (|n| >= 3): 2.0 dB
DEFAULT_HARMONIC_TOLERANCE_DB: Mapping[str, float] = {
    "dominant": 0.5,
    "secondary": 1.0,
    "higher": 2.0,
}


__all__ = [
    "DEFAULT_HARMONIC_TOLERANCE_DB",
    "VaractorModelParameters",
    "compare_cascade_harmonics",
    "compare_varactor_harmonics",
    "extract_varactor_parameters",
    "harmonic_band",
    "harmonic_tolerance_dB",
]


def harmonic_band(n: int) -> str:
    """Classify analytical harmonic index ``n`` into a tolerance band.

    Returns ``"dominant"`` for ``|n| <= 1``, ``"secondary"`` for
    ``|n| == 2``, ``"higher"`` for ``|n| >= 3``.
    """

    abs_n = abs(int(n))
    if abs_n <= 1:
        return "dominant"
    if abs_n == 2:
        return "secondary"
    return "higher"


def harmonic_tolerance_dB(
    n: int,
    tolerances: Mapping[str, float] = DEFAULT_HARMONIC_TOLERANCE_DB,
) -> float:
    """Return the dB tolerance for harmonic ``n`` under ``tolerances``."""

    return float(tolerances[harmonic_band(n)])


class VaractorModelParameters:
    """Plain-data container for the analytical-stub parameter set.

    The Floquet solver consumes ``(S0, dS, phi, R, L)``. This class
    keeps the per-cell extraction step explicit -- :func:`extract_varactor_parameters`
    reads ``C0``, ``C1`` from the spec and ``V_pump_at_varactor`` from
    the HB dataset, then computes ``S0``, ``dS``, and ``phi`` from
    them. ``R`` and ``L`` are stored at ``0.0`` (the SDD2P device has
    no in-element series R/L); callers that want to add an external
    series RLC can pass them explicitly.
    """

    __slots__ = ("S0", "dS", "phi", "R", "L", "C0", "C1", "V_pump")

    def __init__(
        self,
        *,
        S0: float,
        dS: float,
        phi: float,
        R: float = 0.0,
        L: float = 0.0,
        C0: float | None = None,
        C1: float | None = None,
        V_pump: complex | None = None,
    ) -> None:
        if S0 <= 0.0:
            raise ValueError("S0 must be positive.")
        if dS < 0.0 or dS > S0:
            raise ValueError("dS must satisfy 0 <= dS <= S0.")
        if R < 0.0 or L < 0.0:
            raise ValueError("R and L must be non-negative.")
        self.S0 = float(S0)
        self.dS = float(dS)
        self.phi = float(phi)
        self.R = float(R)
        self.L = float(L)
        self.C0 = None if C0 is None else float(C0)
        self.C1 = None if C1 is None else float(C1)
        self.V_pump = None if V_pump is None else complex(V_pump)

    def __repr__(self) -> str:  # pragma: no cover -- diagnostic only
        return (
            f"VaractorModelParameters(S0={self.S0:.4e}, dS={self.dS:.4e}, "
            f"phi={self.phi:.4f}, R={self.R:.4e}, L={self.L:.4e})"
        )


def _capacitance_coefficients_from_q_poly(
    q_poly: Sequence[float],
) -> tuple[float, float]:
    """Return ``(C0, C1)`` from a charge polynomial ``Q(v)``.

    The polynomial is ``Q(v) = sum_k q[k]·v**k``; the capacitance is
    its derivative ``C(v) = ∂Q/∂v = q[1] + 2·q[2]·v + 3·q[3]·v² + ...``,
    so the static capacitance is ``C0 = q[1]`` and the small-signal
    linear C-V slope is ``C1 = 2·q[2]``. Higher-order coefficients are
    ignored at the linearised level used here.
    """

    if len(q_poly) < 2:
        raise ValueError(
            "q_poly must carry at least q[0] and q[1] (Q(0) = 0; C0 = q[1])."
        )
    c0 = float(q_poly[1])
    c1 = 2.0 * float(q_poly[2]) if len(q_poly) >= 3 else 0.0
    if c0 <= 0.0:
        raise ValueError(
            f"q_poly[1] (the static capacitance C0) must be positive; got {c0}."
        )
    return c0, c1


def _spec_varactor_q_poly(spec: DeviceSpec, cell_index: int) -> tuple[float, ...]:
    """Return the cell-k SDD2P's charge polynomial coefficients."""

    sdd_elements = [e for e in spec.elements if e.type == "SDD2P"]
    if not 0 <= cell_index < len(sdd_elements):
        raise IndexError(
            f"cell_index {cell_index} out of range for {len(sdd_elements)} "
            f"SDD2P element(s) in spec {spec.label!r}."
        )
    sdd = sdd_elements[cell_index]
    q_poly = sdd.parameters.get("q_poly_coefficients")
    if q_poly is None:
        raise KeyError(
            f"SDD2P element {sdd.id!r} (cell {cell_index}) has no "
            f"'q_poly_coefficients' parameter."
        )
    return tuple(float(value) for value in q_poly)


def extract_varactor_parameters(
    spec: DeviceSpec,
    dataset: HBDataset,
    cell_map: CellMap,
    cell_index: int,
    *,
    R: float = 0.0,
    L: float = 0.0,
    pump_mix: tuple[int, ...] = (0, 1),
) -> VaractorModelParameters:
    """Extract the analytical-stub parameters for cell ``cell_index``.

    Reads the charge polynomial from the spec's SDD2P; derives ``C0``
    and ``C1`` from the polynomial (``C0 = q[1]``, ``C1 = 2·q[2]``);
    reads the pump-frequency complex voltage at the varactor branch
    top from the HB dataset (mix tuple ``pump_mix``, default
    ``(0, 1)``); and assembles the small-signal elastance modulation
    ``S(t) = S0 + dS·cos(ω_m·t + φ)``.

    ``R`` and ``L`` default to ``0.0`` (the SDD2P element has no
    in-element series R or L). Callers that want to add an external
    series RLC -- e.g. lead inductance or parasitic resistance --
    pass them as keyword arguments.

    The ``pump_mix`` tuple defaults to ``(0, 1)`` (the pump
    fundamental at ``f_m``). A spec running with a different
    convention can override.
    """

    q_poly = _spec_varactor_q_poly(spec, cell_index)
    c0, c1 = _capacitance_coefficients_from_q_poly(q_poly)
    s0 = 1.0 / c0

    # V_0 is read at the MTEE pin-3 branch top -- the input to the local
    # RLC (R_pkg + L_pkg + time-varying C). For Level-9 the branch top
    # net and the SDD2P pin 1 net are the same physical net (no
    # L_pkg/R_pkg in the Level-9 branch chain), so this also matches
    # the Level-9 behaviour byte-for-byte. For Level-10 the two nets
    # differ -- the branch top is upstream of L_pkg+R_pkg, which the
    # analytical solver absorbs into its local RLC.
    branch_top_net = cell_map.cells[cell_index].branch_top_net
    spectrum = dataset.node_voltage(branch_top_net)
    v_pump = complex(spectrum.at_mix(pump_mix))

    abs_vp = abs(v_pump)
    if abs_vp == 0.0:
        # Static / no pump -- dS = 0, phi undefined; solver still runs.
        d_s = 0.0
        phi = 0.0
    else:
        d_s = abs(c1) * abs_vp / (c0 * c0)
        arg_vp = math.atan2(v_pump.imag, v_pump.real)
        # For C1 < 0 the elastance modulation is in-phase with the pump;
        # for C1 > 0 the in-phase elastance modulation is 180 deg
        # shifted (S decreases when C grows).
        phi = arg_vp if c1 < 0.0 else (arg_vp + math.pi)

    # Defensive: in the rare case dS overshoots S0 (a very strong pump or
    # a near-singular C0), clamp to S0 so the Floquet solver's positivity
    # constraint is preserved. We do NOT silently re-normalise: a clamp
    # happens only if dS > 0.999*S0 to leave a finite margin, and the
    # event is surfaced via an informative ValueError to make the
    # caller aware that the analytical linearisation has hit its limit.
    if d_s > 0.999 * s0:
        raise ValueError(
            f"cell {cell_index}: linearised elastance modulation "
            f"dS = {d_s:.3e} exceeds 0.999*S0 ({s0:.3e}); the small-signal "
            f"C(v) approximation has hit its limit at this pump amplitude. "
            f"Reduce pump_power_dBm or extend the analytical model to "
            f"higher-order C-V terms."
        )

    return VaractorModelParameters(
        S0=s0, dS=d_s, phi=phi, R=float(R), L=float(L),
        C0=c0, C1=c1, V_pump=v_pump,
    )


def _signal_voltage_at_branch_top(
    dataset: HBDataset,
    cell_map: CellMap,
    cell_index: int,
    signal_mix: tuple[int, ...],
) -> complex:
    """Read ``V_signal`` at cell ``cell_index``'s varactor branch top.

    This is the analytical solver's ``V_0`` -- the harmonic drive at
    ``ω_s`` against which the harmonic-balance system delta-functions
    the n = 0 row.
    """

    # V_0 is read at the MTEE pin-3 branch top -- the input to the local
    # RLC (R_pkg + L_pkg + time-varying C). For Level-9 the branch top
    # net and the SDD2P pin 1 net are the same physical net (no
    # L_pkg/R_pkg in the Level-9 branch chain), so this also matches
    # the Level-9 behaviour byte-for-byte. For Level-10 the two nets
    # differ -- the branch top is upstream of L_pkg+R_pkg, which the
    # analytical solver absorbs into its local RLC.
    branch_top_net = cell_map.cells[cell_index].branch_top_net
    spectrum = dataset.node_voltage(branch_top_net)
    return complex(spectrum.at_mix(signal_mix))


def _ratio_dB(numerator: complex, denominator: complex) -> float:
    """Return ``20·log10(|num| / |denom|)`` with ``nan`` on zero edges."""

    num_mag = abs(numerator)
    den_mag = abs(denominator)
    if num_mag == 0.0 or den_mag == 0.0:
        return float("nan")
    return 20.0 * math.log10(num_mag / den_mag)


def compare_varactor_harmonics(
    dataset: HBDataset,
    cell_map: CellMap,
    spec: DeviceSpec,
    cell_index: int,
    *,
    truncation_N: int = 10,
    pump_signal_ratio: int = 2,
    signal_mix: tuple[int, ...] = (1, 0),
    pump_mix: tuple[int, ...] = (0, 1),
    R: float = 0.0,
    L: float = 0.0,
    tolerances: Mapping[str, float] = DEFAULT_HARMONIC_TOLERANCE_DB,
    lift_stub_abcd: bool = False,
    y_stub_fn: Callable[[float], complex] | None = None,
) -> Mapping[str, object]:
    """Compare per-harmonic varactor currents (analytical vs ADS).

    Solves the analytical Floquet system for cell ``cell_index``'s
    extracted varactor parameters and the live-HB ``V_pump`` and
    ``V_signal``, then evaluates the dB ratio at every analytical
    harmonic ``n = -N, ..., +N`` against the ADS mix tuple
    ``(1, n)`` (for ``pump_signal_ratio = 2``).

    Parameters
    ----------
    dataset : HBDataset
        The live HB result with both ``V`` and ``I`` columns.
    cell_map : CellMap
        The cascade's per-cell resolution.
    spec : DeviceSpec
        The spec the dataset came from (used to read the SDD2P
        charge polynomial).
    cell_index : int
        0-based cell index.
    truncation_N : int, optional
        Floquet system order; the system size is ``2·N + 1``.
    pump_signal_ratio : int, optional
        Integer ratio ``f_m / f_s``. Defaults to 2 (the paper's
        degenerate point). The analytical harmonic ``n`` maps to ADS
        mix tuple ``(1, n)`` independent of the ratio when the spec
        runs at ``f_m = r·f_s``; the ratio only sets the analytical
        frequency grid ``ω_s + n·ω_m``.
    signal_mix : tuple of int, optional
        ADS mix tuple at which the signal-frequency drive ``V_0`` is
        read; default ``(1, 0)``.
    pump_mix : tuple of int, optional
        ADS mix tuple at which the pump-frequency drive ``V_p`` is
        read; default ``(0, 1)``.
    R, L : float, optional
        External series R and L to add to the analytical RLC. Default
        ``0.0`` (the SDD2P element has none built in).
    tolerances : Mapping[str, float], optional
        Per-harmonic-band dB budgets. Default is
        :data:`DEFAULT_HARMONIC_TOLERANCE_DB`.
    lift_stub_abcd : bool, optional
        When True (the harmonic ABCD lift), the analytical solver
        adds the per-harmonic admittance of the cell's stub MLIN
        cascade in series with the local-RLC varactor model. The stub
        geometry is extracted from the spec automatically via
        :func:`tvtl_ads_builder_v2.analytical.build_stub_y_fn_from_spec`.
        Default ``False`` preserves the historical local-RLC
        behaviour.
    y_stub_fn : Callable[[float], complex], optional
        Explicit stub-admittance callable, overriding the
        spec-extracted form when ``lift_stub_abcd=True``. The
        function maps angular frequency [rad/s] to a complex admittance
        ``Y_stub(omega) = 1 / Z_stub(omega)``. Default ``None``
        triggers spec-based extraction when the lift is enabled.

    Returns
    -------
    dict
        See module docstring. The mapping ``analytical harmonic n ->
        ADS mix (1, n)`` is set by the operating point at
        ``f_m = pump_signal_ratio · f_s``.
    """

    # Avoid the local proxy module the floquet_uc_cascade_response
    # imports -- we just need the bare tridiagonal solver and the
    # current spectrum helper.
    from tvtl.floquet.harmonics import (
        current_spectrum,
        solve_harmonic_coefficients,
    )

    if int(truncation_N) < 0:
        raise ValueError("truncation_N must be non-negative.")
    if int(pump_signal_ratio) <= 0:
        raise ValueError("pump_signal_ratio must be a positive integer.")

    params = extract_varactor_parameters(
        spec, dataset, cell_map, cell_index,
        R=R, L=L, pump_mix=pump_mix,
    )

    # Operating frequencies. ``fundamental_freqs_Hz`` is in the order
    # ``(f_signal, f_pump)`` -- see hb_dataset.py's docstring.
    f_signal = float(dataset.fundamental_freqs_Hz[0])
    f_pump = float(dataset.fundamental_freqs_Hz[1])
    ws = 2.0 * math.pi * f_signal
    wm = 2.0 * math.pi * f_pump

    v_signal = _signal_voltage_at_branch_top(
        dataset, cell_map, cell_index, signal_mix
    )

    # Construct the y_stub_fn used by the harmonic ABCD lift. When the
    # caller does not enable the lift, this remains None and the
    # Floquet solver reduces to its historical local-RLC form.
    effective_y_stub_fn = y_stub_fn
    if lift_stub_abcd and effective_y_stub_fn is None:
        from ..analytical import build_stub_y_fn_from_spec

        effective_y_stub_fn = build_stub_y_fn_from_spec(spec, int(cell_index))

    # Solve the Floquet system normalised to V_0 = 1, then rescale by
    # the live V_signal phasor (the solver is linear in V_0).
    Qn_unit = solve_harmonic_coefficients(
        params.S0, params.dS, params.phi, params.R, params.L,
        ws, wm, int(truncation_N),
        y_stub_fn=effective_y_stub_fn,
    )
    Qn = v_signal * Qn_unit
    _freq_axis, I_n_analytical = current_spectrum(
        Qn, ws, wm, int(truncation_N)
    )

    n_indices = tuple(range(-int(truncation_N), int(truncation_N) + 1))
    # The varactor probe sits in series with the SDD2P (current through
    # the varactor equals current through the probe). For
    # pump_signal_ratio = 2, analytical harmonic n -> ADS mix (1, n).
    mix_tuples = tuple((1, int(n)) for n in n_indices)
    frequencies = tuple(
        float(f_signal + int(n) * f_pump) for n in n_indices
    )

    probe_id = cell_map.cells[cell_index].varactor_probe_id
    if probe_id is None:
        raise ValueError(
            f"cell {cell_index}: no varactor I_Probe resolved -- "
            f"compare_varactor_harmonics requires Level-9+ probed cascades."
        )
    current_spectrum_ads = dataset.i_probe_current(probe_id)
    I_n_ads: list[complex] = []
    for mix in mix_tuples:
        try:
            I_n_ads.append(complex(current_spectrum_ads.at_mix(mix)))
        except KeyError:
            I_n_ads.append(0j)

    ratios_dB = tuple(
        _ratio_dB(I_n_ads[k], I_n_analytical[k])
        for k in range(len(n_indices))
    )
    finite_ratios_with_band = [
        (n_indices[k], ratios_dB[k])
        for k in range(len(n_indices))
        if math.isfinite(ratios_dB[k])
    ]
    if finite_ratios_with_band:
        max_abs_ratio_dB = max(abs(r) for _n, r in finite_ratios_with_band)
        within_tolerance = all(
            abs(r) <= harmonic_tolerance_dB(n, tolerances)
            for n, r in finite_ratios_with_band
        )
    else:
        max_abs_ratio_dB = float("nan")
        within_tolerance = False

    I_n_ana_tuple = tuple(complex(value) for value in I_n_analytical)
    I_n_ads_tuple = tuple(I_n_ads)
    I_n_ana_mag = tuple(abs(value) for value in I_n_ana_tuple)
    I_n_ads_mag = tuple(abs(value) for value in I_n_ads_tuple)

    return {
        "harmonic_index": n_indices,
        "frequencies_Hz": frequencies,
        "mix_tuples": mix_tuples,
        "I_n_analytical": I_n_ana_tuple,
        "I_n_ADS": I_n_ads_tuple,
        "I_n_analytical_mag": I_n_ana_mag,
        "I_n_ADS_mag": I_n_ads_mag,
        "ratio_dB": ratios_dB,
        "max_abs_ratio_dB": float(max_abs_ratio_dB),
        "within_tolerance": bool(within_tolerance),
        "params": params,
        "V_signal_at_branch_top": v_signal,
    }


def compare_cascade_harmonics(
    dataset: HBDataset,
    cell_map: CellMap,
    spec: DeviceSpec,
    *,
    truncation_N: int = 10,
    pump_signal_ratio: int = 2,
    signal_mix: tuple[int, ...] = (1, 0),
    pump_mix: tuple[int, ...] = (0, 1),
    R: float = 0.0,
    L: float = 0.0,
    tolerances: Mapping[str, float] = DEFAULT_HARMONIC_TOLERANCE_DB,
    lift_stub_abcd: bool = False,
    y_stub_fn: Callable[[float], complex] | None = None,
) -> Mapping[str, object]:
    """Run :func:`compare_varactor_harmonics` on every cell of a cascade.

    Returns
    -------
    dict
        Keys:

        * ``per_cell`` -- ``tuple`` of one
          :func:`compare_varactor_harmonics` result per cell.
        * ``max_abs_ratio_dB_per_cell`` -- ``tuple(float)``,
          per-cell worst-harmonic ratio in dB.
        * ``max_abs_ratio_dB`` -- ``float``, worst across the whole
          cascade.
        * ``within_tolerance`` -- ``bool``, True iff every cell's
          per-harmonic result meets its tolerance band.
    """

    n = cell_count(cell_map)
    if n == 0:
        raise ValueError("compare_cascade_harmonics: cascade has no cells.")
    per_cell = tuple(
        compare_varactor_harmonics(
            dataset, cell_map, spec, k,
            truncation_N=truncation_N,
            pump_signal_ratio=pump_signal_ratio,
            signal_mix=signal_mix,
            pump_mix=pump_mix,
            R=R, L=L,
            tolerances=tolerances,
            lift_stub_abcd=lift_stub_abcd,
            y_stub_fn=y_stub_fn,
        )
        for k in range(n)
    )
    max_per_cell = tuple(
        float(entry["max_abs_ratio_dB"]) for entry in per_cell
    )
    finite_max = [v for v in max_per_cell if not math.isnan(v)]
    max_abs_ratio_dB = max(finite_max) if finite_max else float("nan")
    within_tolerance = all(
        bool(entry["within_tolerance"]) for entry in per_cell
    )
    return {
        "per_cell": per_cell,
        "max_abs_ratio_dB_per_cell": max_per_cell,
        "max_abs_ratio_dB": float(max_abs_ratio_dB),
        "within_tolerance": bool(within_tolerance),
    }


