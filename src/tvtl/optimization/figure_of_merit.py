"""Stage-1 figure of merit (A1) and its single-cell-Floquet fallback (A2).

The figure of merit scores a per-cell stub design as a fast proxy for cascade
parametric gain. Both scorers implement one frozen interface
(:class:`FigureOfMerit`), so swapping A1 -> A2 is a one-line change of the
injected scorer; the screening / refinement / pruning code in
:mod:`tvtl.optimization.stage1_per_cell` calls only ``fom.score(...)`` and never
inspects internals (the A1/A2 interchangeability contract).

A1 (:class:`ClosedFormA1`, the default)
---------------------------------------
A closed-form product of three single-cell factors evaluated at the design
frequency ``f0`` (no cascade, no Floquet solve)::

    A1 = m_eff(f0) * kappa_branch(f0) * rho_para(f0)

* ``m_eff = 2|C1|/C0`` -- effective junction modulation depth at a standardized
  control-port drive ``Voc_std = open_circuit_pump_voltage(P_std)``
  (``P_std = -3 dBm``); from
  :meth:`RealisticVaractor.capacitance_coeffs_from_junction_vpump`.
* ``kappa_branch = |Z_ref| / |Z_ref + Z_branch(f0)|`` -- shunt-branch
  current-divider coupling fraction, ``Z_ref = 50 ohm``,
  ``Z_branch = (R_series + jw0 L_pkg) + 1/(jw0 C0) + Z_stub(f0)``.
* ``rho_para = exp(-(x - delta*)^2 / (2 sigma^2))`` with
  ``x = |Im Z_branch(f0)| / Z_ref`` -- a resonance-detuning band-pass guard
  peaking at a small non-zero residual reactance (``delta* = 0.30``,
  ``sigma = 0.50`` defaults), rewarding "near but not at" the branch resonance
  (the steep rising edge below it) and steering the screen out of
  the non-convergent oscillation band. ``rho_para in (0, 1]`` is the intrinsic
  cap -- with ``kappa <= 1`` and ``m_eff`` device-bounded, A1 cannot diverge.

A1's correlation with cascade coherent gain is validated/calibrated in the
A1-correlation calibration sweep.

A2 (:class:`SingleCellFloquetA2`, the fallback)
-----------------------------------------------
The single-cell (``n_cells = 1``) analytical coherent-peak gain via the
validated :func:`...level10_realistic_param_gain` -- still a single-cell
evaluation but a faithful 1-cell version of the 15-cell objective.
Swapped in if calibration shows A1's correlation is borderline.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Protocol

from tvtl.varactors import RealisticVaractor

from .substrate import ROGERS_4003C, SubstrateConfig
from .topology_generator import (
    Termination,
    TopologyFamily,
    VIA_RADIUS_M,
    topology_y_stub_fn,
)

__all__ = [
    "FigureOfMerit",
    "ClosedFormA1",
    "LorentzianA1",
    "ClosedFormA1SelfPumped",
    "SingleCellFloquetA2",
    "standardized_drive_voltage",
    "m_branch_from_source_pump_dBm",
    "calibrate_self_pumped_pump_dBm",
    "CalibrationError",
    "a1_factors",
    "DEFAULT_DRIVE_DBM",
    "DEFAULT_DELTA_STAR",
    "DEFAULT_SIGMA",
    "DEFAULT_X0",
    "DEFAULT_GAMMA",
]

Z_REF_OHM: float = 50.0
DEFAULT_DRIVE_DBM: float = -3.0
DEFAULT_DELTA_STAR: float = 0.30
DEFAULT_SIGMA: float = 0.50
# Calibrated Lorentzian guard optimum (x0=0.25, gamma=0.20): the
# robust full-range-Pearson best across the calibration grid (R^2 0.74).
DEFAULT_X0: float = 0.25
DEFAULT_GAMMA: float = 0.20


class FigureOfMerit(Protocol):
    """The Stage-1 scorer interface (A1 / A2 share it)."""

    def score(
        self,
        topology: TopologyFamily,
        params,
        *,
        f0_Hz: float,
        varactor: RealisticVaractor,
        substrate: SubstrateConfig = ROGERS_4003C,
        drive_dBm: float = DEFAULT_DRIVE_DBM,
    ) -> float: ...


def standardized_drive_voltage(drive_dBm: float, z_ohm: float = Z_REF_OHM) -> float:
    """Open-circuit control-port pump voltage ``Voc = 2 sqrt(2 z P_watts)`` [V].

    Mirrors :func:`...parity.level10_ideal_c.open_circuit_pump_voltage`; inlined
    so the optimization layer carries no parity import for this one scalar.
    """

    p_watts = 10.0 ** ((float(drive_dBm) - 30.0) / 10.0)
    return 2.0 * math.sqrt(2.0 * float(z_ohm) * p_watts)


def a1_factors(
    topology: TopologyFamily,
    params,
    *,
    f0_Hz: float,
    varactor: RealisticVaractor,
    substrate: SubstrateConfig = ROGERS_4003C,
    drive_dBm: float = DEFAULT_DRIVE_DBM,
    via_radius_m: float = VIA_RADIUS_M,
) -> tuple[float, float, float]:
    """Return the three A1 ingredients ``(m_eff, kappa_branch, x_reactance)``.

    Independent of the ``rho_para`` guard parameters ``(delta*, sigma)``, so the
    calibration sweep computes these once per candidate and re-applies
    only the cheap Gaussian. ``x_reactance = |Im Z_branch(f0)| / Z_ref``.
    """

    if varactor.ideal_linear:
        raise ValueError(
            "A1 requires a curved (control-port-capable) varactor; an ideal "
            "constant-C device has no pump response."
        )
    w0 = 2.0 * math.pi * float(f0_Hz)

    # Factor 1: effective modulation depth at the standardized control-port drive.
    voc = standardized_drive_voltage(drive_dBm)
    coeffs = varactor.capacitance_coeffs_from_junction_vpump(voc, phi=0.0)
    c0 = float(coeffs[0].real)
    m_eff = 2.0 * abs(complex(coeffs[1])) / c0 if c0 > 0.0 else 0.0

    # Factors 2 & 3: branch coupling + resonance detuning.
    z_stub = 1.0 / topology_y_stub_fn(
        topology, params, substrate, via_radius_m=via_radius_m
    )(w0)
    z_pkg = float(varactor.R_series_ohm) + 1j * w0 * float(varactor.L_pkg_h)
    z_cap = 1.0 / (1j * w0 * float(varactor.C0_f))
    z_branch = z_pkg + z_cap + z_stub
    kappa = Z_REF_OHM / abs(Z_REF_OHM + z_branch)
    x_reactance = abs(z_branch.imag) / Z_REF_OHM
    return float(m_eff), float(kappa), float(x_reactance)


def _rho_para(x_reactance: float, delta_star: float, sigma: float) -> float:
    """Gaussian resonance-detuning guard ``exp(-(x - delta*)^2 / (2 sigma^2))``."""

    return math.exp(-((float(x_reactance) - float(delta_star)) ** 2) / (2.0 * float(sigma) ** 2))


def _rho_lorentzian(x_reactance: float, x0: float, gamma: float) -> float:
    """Lorentzian resonance-detuning guard ``gamma^2 / ((x - x0)^2 + gamma^2)``.

    Same "peak at a small non-zero residual reactance" hypothesis as the
    Gaussian, but with **heavy tails**: an off-resonance candidate keeps a
    polynomially-small (not exponentially-vanishing) ``rho``, so A1 does not
    underflow to denormal values for the far tail -- which keeps a full-range
    correlation metric meaningful (the guard-form comparison). FWHM ~ 2*gamma.
    """

    g = float(gamma)
    return g * g / ((float(x_reactance) - float(x0)) ** 2 + g * g)


@dataclass(frozen=True)
class ClosedFormA1:
    """The default A1 scorer: ``m_eff * kappa_branch * rho_para``.

    The guard parameters ``(delta_star, sigma)`` are the calibration
    knobs; the defaults are the calibrated values.
    """

    delta_star: float = DEFAULT_DELTA_STAR
    sigma: float = DEFAULT_SIGMA
    via_radius_m: float = VIA_RADIUS_M

    def score(
        self,
        topology: TopologyFamily,
        params,
        *,
        f0_Hz: float,
        varactor: RealisticVaractor,
        substrate: SubstrateConfig = ROGERS_4003C,
        drive_dBm: float = DEFAULT_DRIVE_DBM,
    ) -> float:
        m_eff, kappa, x = a1_factors(
            topology, params, f0_Hz=f0_Hz, varactor=varactor,
            substrate=substrate, drive_dBm=drive_dBm, via_radius_m=self.via_radius_m,
        )
        return m_eff * kappa * _rho_para(x, self.delta_star, self.sigma)


@dataclass(frozen=True)
class LorentzianA1:
    """Alternative A1 scorer: ``m_eff * kappa_branch * rho_lorentzian``.

    Identical to :class:`ClosedFormA1` except the resonance-detuning guard is a
    Lorentzian ``gamma^2 / ((x - x0)^2 + gamma^2)`` (heavy-tailed) rather than a
    Gaussian. The heavy tail avoids the Gaussian's denormal underflow for
    far-off-resonance candidates, so a full-range gain-vs-log10(A1) correlation
    stays well-conditioned. Compared against the Gaussian during guard-form
    calibration.
    """

    x0: float = DEFAULT_X0
    gamma: float = DEFAULT_GAMMA
    via_radius_m: float = VIA_RADIUS_M

    def score(
        self,
        topology: TopologyFamily,
        params,
        *,
        f0_Hz: float,
        varactor: RealisticVaractor,
        substrate: SubstrateConfig = ROGERS_4003C,
        drive_dBm: float = DEFAULT_DRIVE_DBM,
    ) -> float:
        m_eff, kappa, x = a1_factors(
            topology, params, f0_Hz=f0_Hz, varactor=varactor,
            substrate=substrate, drive_dBm=drive_dBm, via_radius_m=self.via_radius_m,
        )
        return m_eff * kappa * _rho_lorentzian(x, self.x0, self.gamma)


def m_branch_from_source_pump_dBm(
    pump_power_dBm: float,
    varactor: RealisticVaractor,
    *,
    z_source_ohm: float = Z_REF_OHM,
) -> float:
    """Provisional self-pumped ``m_branch_top`` from source pump power [dBm].

    Maps the input pump power applied to a 50 ohm-matched through-line to the
    framework's branch-top modulation-depth convention:

        V_branch_top = sqrt(2 * Z_source * P_pump_watts)        # matched-load peak
        m_branch_top = |dC/dV(V_bias)| * V_branch_top / C0      # depth proxy

    The analytical leg's self-pumped path (``level10_realistic_param_gain``
    via ``RealisticVaractor.effective_capacitance_coeffs``) applies the
    ``L_pkg`` / ``R_series`` divider internally; ``m_branch_top`` is the
    input the user supplies, *not* the post-divider junction depth.

    Convention status: the mapping is provisional pending co-author
    input on self-pumped operating points (open question). Stage 1
    ranking is convention-
    invariant (any scalar mapping preserves the same ordering), but
    Stages 4+ depend on the absolute m_branch value passed to the
    analytical leg.
    """

    p_watts = 10.0 ** ((float(pump_power_dBm) - 30.0) / 10.0)
    v_branch_top = math.sqrt(2.0 * float(z_source_ohm) * p_watts)
    if varactor.C0_f <= 0.0:
        return 0.0
    return abs(varactor.dC_dV_f_per_v) * v_branch_top / varactor.C0_f


class CalibrationError(ValueError):
    """Raised when :func:`calibrate_self_pumped_pump_dBm` cannot satisfy
    its target."""


def _junction_m_eff_at_source_pump_dBm(
    pump_power_dBm: float,
    varactor: RealisticVaractor,
    f_pump_Hz: float,
    *,
    z_source_ohm: float = Z_REF_OHM,
) -> float:
    """Return junction ``m_eff = 2|C_1|/C_0`` at the given source pump power.

    Mirrors the helper used by :class:`ClosedFormA1SelfPumped`. The
    ``effective_capacitance_coeffs`` call applies the L_pkg / R_series
    pump divider internally.
    """

    m_branch = m_branch_from_source_pump_dBm(
        pump_power_dBm, varactor, z_source_ohm=z_source_ohm,
    )
    coeffs = varactor.effective_capacitance_coeffs(
        float(f_pump_Hz), m_branch, phi=0.0,
    )
    c0 = float(coeffs[0].real)
    if c0 <= 0.0:
        return 0.0
    return 2.0 * abs(complex(coeffs[1])) / c0


def calibrate_self_pumped_pump_dBm(
    *,
    varactor: RealisticVaractor,
    f_pump_Hz: float,
    m_eff_target: float = 0.22,
    tolerance: float = 0.005,
    z_source_ohm: float = Z_REF_OHM,
    bracket_dBm: tuple[float, float] = (-30.0, 30.0),
    max_iter: int = 80,
) -> float:
    """Bisect on source ``pump_dBm`` so junction ``m_eff`` matches ``m_eff_target``.

    Returns the source pump power (dBm) that produces the requested junction
    modulation depth ``m_eff = 2|C_1|/C_0`` from
    :meth:`RealisticVaractor.effective_capacitance_coeffs` at ``f_pump_Hz``.
    The mapping from source power to branch-top depth uses
    :func:`m_branch_from_source_pump_dBm` (the matched-load convention).
    The L_pkg / R_series divider attenuates / enhances the branch-top depth
    inside ``effective_capacitance_coeffs``, so the calibration is on the
    *junction* m_eff, not the branch-top depth.

    Parameters
    ----------
    varactor : RealisticVaractor
        Curved realistic varactor; must respond to a pump (``ideal_linear``
        devices have no pump response).
    f_pump_Hz : float
        Pump frequency [Hz]; sets the package divider.
    m_eff_target : float, optional
        Target junction m_eff. Default 0.22 — small-signal regime matching
        the control-port reference at -3 dBm
        (``m_eff(control-port, -3 dBm) = 0.218``).
    tolerance : float, optional
        Acceptable absolute deviation of the returned m_eff from
        ``m_eff_target``. Default 0.005.
    z_source_ohm : float, optional
        Source impedance for the source-dBm -> V_branch_top conversion.
        Default :data:`Z_REF_OHM`.
    bracket_dBm : (float, float), optional
        Search bracket for the bisection in dBm. Default (-30, +30) dBm
        spans the operating range of interest.
    max_iter : int, optional
        Maximum bisection iterations. Default 80 (well above what's needed
        for double-precision convergence over a 60 dB bracket).

    Returns
    -------
    float
        Calibrated source ``pump_dBm`` (in dBm).

    Raises
    ------
    CalibrationError
        If ``m_eff_target`` is not bracketed by the m_eff values at
        ``bracket_dBm[0]`` and ``bracket_dBm[1]``, or if convergence
        fails within ``max_iter``.

    Example
    -------
    For Varactor A ``(CJ0=0.3 pF, VJ=3.5, M=1.7, L_pkg=0.3 nH,
    R_series=0.5 Ω)`` at f_pump=6.2 GHz with the default target
    m_eff=0.22, the calibrator returns ~+1.67 dBm
    (the A.3.1 locked operating point).
    """

    lo, hi = float(bracket_dBm[0]), float(bracket_dBm[1])
    if lo >= hi:
        raise CalibrationError(
            f"bracket_dBm must be strictly ordered, got {bracket_dBm}."
        )
    m_lo = _junction_m_eff_at_source_pump_dBm(
        lo, varactor, f_pump_Hz, z_source_ohm=z_source_ohm,
    )
    m_hi = _junction_m_eff_at_source_pump_dBm(
        hi, varactor, f_pump_Hz, z_source_ohm=z_source_ohm,
    )
    target = float(m_eff_target)
    if not (m_lo <= target <= m_hi):
        raise CalibrationError(
            f"m_eff_target={target} not bracketed by bracket_dBm={bracket_dBm} "
            f"dBm; m_eff({lo:+.2f})={m_lo:.4f}, m_eff({hi:+.2f})={m_hi:.4f}. "
            f"If the target m_eff is achievable outside the current bracket, "
            f"consider widening to e.g. [-40, +40] dBm via the `bracket_dBm` "
            f"parameter."
        )
    # Always run full bisection (no early exit) so the result converges to
    # machine precision and is deterministic. With a 60 dB bracket and
    # max_iter=80, the bracket width shrinks to 60 / 2^80 ~ 5e-23 dB --
    # well below double-precision noise. The `tolerance` parameter is used
    # only at the end as a sanity gate.
    for _ in range(int(max_iter)):
        mid = 0.5 * (lo + hi)
        m_mid = _junction_m_eff_at_source_pump_dBm(
            mid, varactor, f_pump_Hz, z_source_ohm=z_source_ohm,
        )
        if m_mid < target:
            lo = mid
        else:
            hi = mid
    final = 0.5 * (lo + hi)
    final_m = _junction_m_eff_at_source_pump_dBm(
        final, varactor, f_pump_Hz, z_source_ohm=z_source_ohm,
    )
    if abs(final_m - target) > float(tolerance):
        raise CalibrationError(
            f"bisection did not converge within {max_iter} iterations; "
            f"final pump_dBm={final:+.4f}, m_eff={final_m:.4f}, "
            f"residual={abs(final_m - target):.4f}, tolerance={tolerance}."
        )
    return final


@dataclass(frozen=True)
class ClosedFormA1SelfPumped:
    """Self-pumped sibling of :class:`LorentzianA1`.

    ``A1_sp = m_eff_self_pumped * kappa_branch * rho_lorentzian``

    ``kappa_branch`` and ``rho_lorentzian`` are reused unchanged from
    :class:`LorentzianA1` -- both depend only on the signal-path branch
    impedance ``Z_branch(f0_Hz)`` and the stub geometry, neither of which
    depends on the pump regime.

    ``m_eff_self_pumped`` is computed by mapping the source ``pump_power_dBm``
    to a branch-top depth via :func:`m_branch_from_source_pump_dBm` and then
    calling :meth:`RealisticVaractor.effective_capacitance_coeffs`, which
    applies the ``L_pkg`` / ``R_series`` divider internally to yield the
    junction's harmonic capacitance spectrum. ``m_eff = 2*|C_1|/C_0`` uses
    the same definition as the control-port A1 (cf.
    :func:`a1_factors`).

    Rank-equivalence to control-port :class:`LorentzianA1`
    ------------------------------------------------------
    At fixed varactor and ``f0_Hz``, the ``m_eff`` factor is a *scalar
    invariant* across any stub-geometry sweep: it depends on neither
    width nor length. Since the same is true for control-port
    ``LorentzianA1``, the two scorers produce **identically-ranked**
    top-K candidates with absolute scores differing only by the
    constant ratio ``m_eff_sp / m_eff_cp``. Verified empirically at
    A.3.1 (Spearman = 1.000000 over 2157 Sobol candidates; ratio
    constant at 9.32194x to floating-point precision). The
    rank-equivalence is locked by a regression test
    (``tests/test_figure_of_merit.py::
    test_self_pumped_rank_equivalence_to_lorentzian``).

    Consequence: a Sobol top-K selection driven by self-pumped A1 picks
    the same candidates control-port A1 would have. The self-pumped
    scorer's value is providing an interpretable self-pumped ``m_eff``
    for downstream stages, not changing which candidates the screen
    selects.

    Construction parameters
    -----------------------
    f_pump_Hz : float
        Pump frequency [Hz]. Sets the package divider used internally
        by :meth:`RealisticVaractor.effective_capacitance_coeffs`.
    pump_power_dBm : float
        Self-pumped source pump power [dBm] applied to the
        through-line; maps to ``m_branch_top`` via
        :func:`m_branch_from_source_pump_dBm`. Default 16 dBm matches
        the self-pumped canonical
        (``level10_realistic_c.py:225``); pump-amplitude calibration
        for the analytical-proxy-valid small-signal regime is an open
        question.
    x0, gamma : float
        Lorentzian rho parameters; defaults match
        :class:`LorentzianA1` (calibrated).
    via_radius_m : float
        Via-radius for ``topology_y_stub_fn``; default
        :data:`VIA_RADIUS_M`.
    z_source_ohm : float
        Source impedance used to convert ``pump_power_dBm`` to
        ``V_branch_top``; default :data:`Z_REF_OHM` = 50 ohm.

    Notes on the ``drive_dBm`` argument
    -----------------------------------
    The :class:`FigureOfMerit` protocol requires a ``drive_dBm`` keyword
    in ``score()``; for self-pumped scoring it is **immaterial**.
    Internally :func:`a1_factors` is called with it to compute the
    (discarded) control-port ``m_eff`` -- the kappa and x_reactance
    returned by :func:`a1_factors` do not depend on ``drive_dBm`` and
    are the only outputs the self-pumped scorer consumes. Passing any
    value (the default ``DEFAULT_DRIVE_DBM`` is fine) is harmless.
    """

    f_pump_Hz: float
    pump_power_dBm: float = 16.0
    x0: float = DEFAULT_X0
    gamma: float = DEFAULT_GAMMA
    via_radius_m: float = VIA_RADIUS_M
    z_source_ohm: float = Z_REF_OHM

    def score(
        self,
        topology: TopologyFamily,
        params,
        *,
        f0_Hz: float,
        varactor: RealisticVaractor,
        substrate: SubstrateConfig = ROGERS_4003C,
        drive_dBm: float = DEFAULT_DRIVE_DBM,
    ) -> float:
        # Self-pumped m_eff: source -> V_branch_top -> framework depth ->
        # `effective_capacitance_coeffs` (which applies the L_pkg/R_series
        # divider) -> 2|C_1|/C_0 at the junction.
        m_branch = m_branch_from_source_pump_dBm(
            self.pump_power_dBm, varactor, z_source_ohm=self.z_source_ohm,
        )
        coeffs = varactor.effective_capacitance_coeffs(
            float(self.f_pump_Hz), m_branch, phi=0.0,
        )
        c0 = float(coeffs[0].real)
        if c0 <= 0.0:
            return 0.0
        m_eff = 2.0 * abs(complex(coeffs[1])) / c0
        # kappa_branch + x_reactance: signal-path-only, regime-agnostic
        # (the control-port m_eff returned by a1_factors is discarded).
        _m_eff_cp, kappa, x = a1_factors(
            topology, params, f0_Hz=f0_Hz, varactor=varactor,
            substrate=substrate, drive_dBm=drive_dBm,
            via_radius_m=self.via_radius_m,
        )
        return m_eff * kappa * _rho_lorentzian(x, self.x0, self.gamma)


@dataclass(frozen=True)
class SingleCellFloquetA2:
    """Fallback A2 scorer: single-cell analytical coherent-peak gain (dB).

    A faithful 1-cell version of the 15-cell coherent objective, via the
    validated :func:`...level10_realistic_param_gain` on a 1-cell control-port
    spec. Linear (OPEN/SHORT) families only -- COMBINED is A1-screened.
    Slower than A1 (~1 matrix solve) but a true gain proxy.
    """

    truncation_N: int = 6
    via_radius_m: float = VIA_RADIUS_M

    def score(
        self,
        topology: TopologyFamily,
        params,
        *,
        f0_Hz: float,
        varactor: RealisticVaractor,
        substrate: SubstrateConfig = ROGERS_4003C,
        drive_dBm: float = DEFAULT_DRIVE_DBM,
    ) -> float:
        # Imported lazily: A2 reaches up into the v2 builder (spec generator +
        # analytical leg); keeping the import local avoids loading it for the
        # default A1 path and documents the layering.
        from tvtl_ads_builder_v2.analytical import level10_realistic_param_gain
        from tvtl_ads_builder_v2.parity.optimized_design import (
            optimized_control_port_spec,
        )

        if topology.termination is Termination.COMBINED:
            raise NotImplementedError(
                "A2 scores linear (OPEN/SHORT) families only; COMBINED is "
                "A1-screened only."
            )
        spec = optimized_control_port_spec(
            topology=topology, params=params, varactor=varactor, substrate=substrate,
            n_cells=1, f_signal_Hz=float(f0_Hz), f_pump_Hz=2.0 * float(f0_Hz),
            pump_power_dBm=float(drive_dBm), via_radius_m=self.via_radius_m,
        )
        voc = standardized_drive_voltage(drive_dBm)
        res = level10_realistic_param_gain(
            spec, varactor, control_port_vpump_v=voc, truncation_N=self.truncation_N,
        )
        return float(res["coherent_peak_dB"])


