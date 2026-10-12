"""Length (N) + pump as search variables, with an in-loop stability constraint.

The structural search expansion. The production
self-pumped engine
(:func:`tvtl_ads_builder_v2.parity.self_pumped_design.design_optimized_self_pumped`)
searches geometry only: cell count ``N`` is fixed (15) and pump power is
calibrated once to a target junction ``m_eff`` (see
``docs/methodology/OPTIMIZER_METHODOLOGY.md``). This module adds the two missing
search dimensions **opt-in and backward-compatibly** -- the default engine path
is untouched and byte-identical (the A.3.1 anchor still reproduces D135A120 /
+9.71 dB) -- and wraps them in the stability constraint the physics demands.

The crux (why stability must be IN-LOOP, not a post-hoc filter)
--------------------------------------------------------------
With pump a free variable and ``G_mod/in`` the objective, the search *will* drive
toward the parametric-instability edge: that is exactly where net-positive
transducer gain lives (the net-gain investigation found net-positive only
appears as the design approaches oscillation, r_A 0 -> 0.187, the next pump step
HB-non-convergent). The cheap analytical ``g_mod_in`` proxy rises monotonically
with pump and **cannot see oscillation** (it has no loss-invariant
self-oscillation probe). So a search that maximised the analytical proxy alone
would happily return a *lasing* design as the "optimum". The stability check
must therefore be **inside the selection loop**: each promising candidate is
regime-classified (amplifier / boundary / oscillator), and the optimum is the
best ``G_mod/in`` **subject to** stable-or-boundary, with oscillator-regime
candidates rejected outright.

Constraint formulation (the selection rule)
-------------------------------------------
    maximise  realised G_mod/in
    over      (geometry, N, pump)
    s.t.      regime(candidate) in {amplifier, boundary}
    reject    regime in {oscillator (lasing), non-convergent (past onset),
                          indeterminate (stability uncertifiable)}

``boundary`` = the band-edge point where the two cheap operational tests
disagree (the loss-invariant detector's INDETERMINATE-while-probed verdict): the
VarA anchor-cell net-positive +0.55 dB point is a BOUNDARY, and per the framing
rule it is reported as *near the parametric instability threshold*, never as a
comfortably-stable amplifier. The amplifier-only optimum (the genuinely stable
point) is reported separately as ``best_stable``.

Cost discipline (analytical screen -> ADS finalists)
---------------------------------------------------
Regime classification is an ADS-HB measurement (``run_amplification_check`` runs
~7 HB solves per point) and must not run on every grid point. The division of
labour:

* **Analytical screen (cheap, every grid point).** For each ``(N, pump)`` the
  fully-analytical engine selects its best geometry and reports the analytical
  ``g_mod_in`` proxy. All grid points are ranked by this proxy.
* **ADS stability check (expensive, finalists only).** Only the top
  ``n_finalists`` grid points (highest analytical proxy) are regime-classified
  by the injected ``regime_check`` callback; the optimum is the best realised
  ``G_mod/in`` among the admissible (stable-or-boundary) finalists.

The ``regime_check`` callback is injected so the pure-Python search structure
(grid construction, ranking, the constraint) is unit-testable without ADS;
:func:`ads_regime_check` is the production callback (wraps
:func:`tvtl_ads_builder_v2.analysis.amplification_check.run_amplification_check`).

Backward compatibility
----------------------
With ``search_n_cells=False`` and ``search_pump=False`` (the defaults) the grid
collapses to the single fixed operating point and the wrapped engine call is the
ordinary :func:`design_optimized_self_pumped` -- nothing in the default path
changes. The expansion is enabled only by an explicit ``search_*=True``.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Mapping
from dataclasses import dataclass, replace
from pathlib import Path

from tvtl.microstrip.models import hammerstad_kirschning
from tvtl.optimization.figure_of_merit import calibrate_self_pumped_pump_dBm
from tvtl.optimization.substrate import (
    ROGERS_4003C,
    SubstrateConfig,
    microstrip_width_for_z0,
)
from tvtl.varactors import RealisticVaractor

from ..analysis.amplification_check import (
    AMPLIFICATION,
    INDETERMINATE,
    NON_CONVERGENT,
    OSCILLATION,
    run_amplification_check,
)
from .gain_objective import OBJECTIVE_G_MOD_IN
from .self_pumped_design import (
    NoSelfPumpedCandidateError,
    SelfPumpedDesignResult,
    design_optimized_self_pumped,
    main_period_anchor_at_3p1GHz,
)

__all__ = [
    "SPEED_OF_LIGHT_M_PER_S",
    "REGIME_AMPLIFIER",
    "REGIME_BOUNDARY",
    "REGIME_OSCILLATOR",
    "REGIME_NON_CONVERGENT",
    "REGIME_INDETERMINATE",
    "STABLE_OR_BOUNDARY",
    "REJECTED_REGIMES",
    "regime_label_from_check",
    "main_line_guided_wavelength_m",
    "n_cells_grid_for_length",
    "default_pump_ladder_dBm",
    "ads_regime_check",
    "GridCandidate",
    "ExpandedSearchResult",
    "design_optimized_self_pumped_search",
]

SPEED_OF_LIGHT_M_PER_S = 299792458.0

# --------------------------------------------------------------------------- #
#  Regime labels + the stability-constraint admissibility set                 #
# --------------------------------------------------------------------------- #
REGIME_AMPLIFIER = "amplifier"          # stable, input-tracking (allowed)
REGIME_BOUNDARY = "boundary"            # band-edge, tests disagree (allowed, flagged)
REGIME_OSCILLATOR = "oscillator"        # self-oscillation / lasing (REJECTED)
REGIME_NON_CONVERGENT = "non-convergent"  # past the instability onset (REJECTED)
REGIME_INDETERMINATE = "indeterminate"  # stability uncertifiable (REJECTED)
REGIME_UNCHECKED = "<unchecked>"        # not yet ADS-classified

# The in-loop constraint: stable-or-boundary candidates are admissible optima;
# everything else (including "cannot certify") is rejected so the search can
# never return an oscillator-regime design.
STABLE_OR_BOUNDARY = frozenset({REGIME_AMPLIFIER, REGIME_BOUNDARY})
REJECTED_REGIMES = frozenset(
    {REGIME_OSCILLATOR, REGIME_NON_CONVERGENT, REGIME_INDETERMINATE}
)

CONSTRAINT_DESCRIPTION = (
    "maximise realised G_mod/in over (geometry x N x pump) SUBJECT TO "
    "regime in {amplifier (stable), boundary (band-edge)}; REJECT regime in "
    "{oscillator (lasing), non-convergent (past instability onset), "
    "indeterminate (stability uncertifiable)}. The analytical g_mod_in proxy "
    "ranks all grid points; the ADS regime check classifies only the top "
    "finalists; the optimum is the highest realised G_mod/in among admissible "
    "finalists (best_stable reports the amplifier-only optimum separately)."
)


def regime_label_from_check(check) -> str:
    """Map an :class:`AmplificationCheck` verdict to a constraint regime label.

    * ``AMPLIFICATION``  -> ``amplifier`` (stable; both cheap tests agree it
      tracks the input).
    * ``OSCILLATION``    -> ``oscillator`` (lasing; rejected).
    * ``NON_CONVERGENT`` -> ``non-convergent`` (driven HB diverged = past the
      instability onset; rejected).
    * ``INDETERMINATE`` splits: a **probed** point whose two operational tests
      disagree is the band-edge ``boundary`` (allowed, flagged); an
      **unprobed** point (signal-off HB unmeasurable) is ``indeterminate``
      (cannot certify stability -> rejected).
    """

    cls = check.classification
    if cls == NON_CONVERGENT:
        return REGIME_NON_CONVERGENT
    if cls == OSCILLATION:
        return REGIME_OSCILLATOR
    if cls == AMPLIFICATION:
        return REGIME_AMPLIFIER
    if cls == INDETERMINATE:
        # Tests disagree at a probed point = straddling the gain/oscillation
        # boundary; an unprobed point cannot be certified either way.
        if bool(getattr(check, "signal_off_converged", False)) and not bool(
            getattr(check, "tests_agree", True)
        ):
            return REGIME_BOUNDARY
        return REGIME_INDETERMINATE
    return REGIME_INDETERMINATE


# --------------------------------------------------------------------------- #
#  STEP 1 -- length (N) grid bounded by the distributed-gain physics          #
# --------------------------------------------------------------------------- #
def main_line_guided_wavelength_m(
    f_signal_Hz: float,
    substrate: SubstrateConfig = ROGERS_4003C,
    *,
    z0_ohm: float = 50.0,
) -> float:
    """Guided wavelength [m] of a ``z0_ohm`` microstrip main line at ``f_signal``.

    ``lambda_g = c / (f * sqrt(eps_eff))`` with ``eps_eff`` from the
    Hammerstad-Kirschning model
    (:func:`tvtl.microstrip.models.hammerstad_kirschning`) at the width that
    realises ``z0_ohm`` on ``substrate``. The 50 ohm line is the natural
    reference because Stage 2 matches the periodic main line to a 50 ohm Bloch
    impedance and the edges are matched to 50 ohm.
    """

    w_m = microstrip_width_for_z0(z0_ohm, substrate, f_signal_Hz)
    m = substrate.stub_substrate_mapping()
    _, eps_eff = hammerstad_kirschning(
        float(w_m), m["t_m"], m["h_m"], m["eps_r"], [float(f_signal_Hz)]
    )
    return SPEED_OF_LIGHT_M_PER_S / (float(f_signal_Hz) * math.sqrt(float(eps_eff[0])))


def n_cells_grid_for_length(
    f_signal_Hz: float,
    *,
    main_period_m: float | None = None,
    substrate: SubstrateConfig = ROGERS_4003C,
    target_lambda: tuple[float, float] = (1.0, 4.0),
    n_points: int = 5,
    n_abs_bounds: tuple[int, int] = (4, 60),
) -> tuple[int, ...]:
    """Integer ``N`` grid spanning a target cascade length in guided wavelengths.

    Distributed parametric gain (Koutserimpas) needs the cascade to be **several
    wavelengths long** at the operating frequency for the per-cell boost to
    accumulate. The per-cell main-line period sets the per-cell electrical
    length, so ``N`` maps to a physical length ``L = N * period`` and to an
    electrical length ``L / lambda_g``. The grid spans
    ``target_lambda[0] .. target_lambda[1]`` guided wavelengths.

    Bounds (per the net-gain investigation):

    * **lower (default 1 lambda).** Below ~1 wavelength the cascade is
      electrically short, modulation is weak and there is no room for
      distributed gain to develop.
    * **upper (default 4 lambda).** The net-gain length sweep (N = 8 -> 55)
      showed the distributed boost **saturates by ~4 wavelengths (~N=40)** while
      the per-cell cascade loss keeps accruing, so longer cascades buy boost
      that no longer outruns the added loss.

    ``n_abs_bounds`` is a hard integer clamp (a sanity backstop). With the
    default anchored period the per-cell electrical length is
    frequency-invariant, so the wavelength-targeted grid is essentially
    frequency-independent (a custom ``main_period_m`` adapts it). At
    f_signal = 3.1 GHz the anchor's N=15 (~1.55 lambda) falls inside the default
    span. Returns a sorted de-duplicated tuple of cell counts.
    """

    if main_period_m is None:
        main_period_m = main_period_anchor_at_3p1GHz(f_signal_Hz)
    if float(main_period_m) <= 0.0:
        raise ValueError("main_period_m must be positive.")
    lo_lambda, hi_lambda = float(target_lambda[0]), float(target_lambda[1])
    if not (0.0 < lo_lambda <= hi_lambda):
        raise ValueError(
            f"target_lambda must satisfy 0 < lo <= hi, got {target_lambda}."
        )
    lam_g = main_line_guided_wavelength_m(f_signal_Hz, substrate)
    cells_per_lambda = lam_g / float(main_period_m)

    n_lo_abs, n_hi_abs = int(n_abs_bounds[0]), int(n_abs_bounds[1])
    lo = max(n_lo_abs, int(round(lo_lambda * cells_per_lambda)))
    hi = min(n_hi_abs, int(round(hi_lambda * cells_per_lambda)))
    if hi < lo:
        hi = lo
    if n_points <= 1 or hi == lo:
        return (lo,)
    step = (hi - lo) / (n_points - 1)
    grid = sorted({int(round(lo + i * step)) for i in range(n_points)})
    return tuple(grid)


# --------------------------------------------------------------------------- #
#  STEP 2 -- pump ladder + the ADS regime-check callback                       #
# --------------------------------------------------------------------------- #
def default_pump_ladder_dBm(
    calibrated_dBm: float | None = None,
    *,
    ladder: tuple[float, ...] = (5.0, 9.0, 13.0, 17.0),
) -> tuple[float, ...]:
    """Default source-pump search ladder [dBm]: small-signal up to the edge.

    The net-gain pump-ladder methodology
    probed +5 / +9 / +13 / +17 dBm and found net-positive transducer gain only
    near the +13..+17 dBm instability onset, with the comfortably-stable
    interior net-negative. This default ladder spans that stable -> edge range.
    When ``calibrated_dBm`` is given (the small-signal m_eff calibration point)
    it is folded in as the stable low anchor of the ladder.
    """

    pts = list(ladder)
    if calibrated_dBm is not None:
        pts.append(round(float(calibrated_dBm), 6))
    return tuple(sorted(set(pts)))


def ads_regime_check(spec, run_dir: str | Path, **kwargs):
    """Production regime-check callback: ADS amplification-vs-oscillation verdict.

    Thin wrapper over
    :func:`tvtl_ads_builder_v2.analysis.amplification_check.run_amplification_check`
    (input-dependence slope + signal-off ``r_A`` -> loss-invariant regime). The
    heavy ADS imports are deferred inside the runner, so importing this module
    never requires ADS; only *calling* this callback does.
    """

    return run_amplification_check(spec, run_dir, **kwargs)


# --------------------------------------------------------------------------- #
#  Result types                                                                #
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class GridCandidate:
    """One (N, pump) grid point: its analytical-best design + regime verdict.

    ``regime`` is ``"<unchecked>"`` until the ADS stability check runs on it (it
    only runs on the finalists). ``admissible`` is whether ``regime`` is in
    :data:`STABLE_OR_BOUNDARY`. ``realized_g_mod_in_dB`` is the ADS-measured
    out/in gain (nan when unchecked or non-convergent).
    """

    design: SelfPumpedDesignResult
    n_cells: int
    pump_dBm: float
    analytical_g_mod_in_dB: float
    regime: str = REGIME_UNCHECKED
    admissible: bool = False
    realized_g_mod_in_dB: float = float("nan")
    check: object | None = None
    note: str = ""

    @property
    def topology_key(self) -> str:
        return self.design.topology.key

    def to_dict(self) -> dict:
        return {
            "topology_key": self.topology_key,
            "n_cells": self.n_cells,
            "pump_dBm": self.pump_dBm,
            "analytical_g_mod_in_dB": self.analytical_g_mod_in_dB,
            "regime": self.regime,
            "admissible": self.admissible,
            "realized_g_mod_in_dB": self.realized_g_mod_in_dB,
            "note": self.note,
        }


@dataclass(frozen=True)
class ExpandedSearchResult:
    """Outcome of the stability-constrained N + pump search.

    ``selected`` is the optimum under the constraint (best realised G_mod/in
    among stable-or-boundary finalists) or ``None`` if no finalist is admissible
    (a legitimate "no stable optimum in this catalog" answer -- e.g. a device
    that lases across the whole ladder). ``best_stable`` is the amplifier-only
    optimum (the comfortably-stable point), reported separately so the
    stable-vs-edge distinction is explicit.
    """

    selected: GridCandidate | None
    best_stable: GridCandidate | None
    n_cells_grid: tuple[int, ...]
    pump_grid_dBm: tuple[float | None, ...]
    target_lambda: tuple[float, float]
    finalists: tuple[GridCandidate, ...]
    all_grid_points: tuple[GridCandidate, ...]
    constraint: str = CONSTRAINT_DESCRIPTION
    note: str = ""

    def to_dict(self) -> dict:
        return {
            "selected": self.selected.to_dict() if self.selected else None,
            "best_stable": self.best_stable.to_dict() if self.best_stable else None,
            "n_cells_grid": list(self.n_cells_grid),
            "pump_grid_dBm": list(self.pump_grid_dBm),
            "target_lambda": list(self.target_lambda),
            "finalists": [c.to_dict() for c in self.finalists],
            "all_grid_points": [c.to_dict() for c in self.all_grid_points],
            "constraint": self.constraint,
            "note": self.note,
        }


# --------------------------------------------------------------------------- #
#  Selection helper (pure; the stability constraint applied to checked rows)   #
# --------------------------------------------------------------------------- #
def _score(candidate: GridCandidate, *, realised: bool) -> float:
    v = candidate.realized_g_mod_in_dB if realised else candidate.analytical_g_mod_in_dB
    return v if math.isfinite(v) else float("-inf")


# --------------------------------------------------------------------------- #
#  The expanded search                                                         #
# --------------------------------------------------------------------------- #
def design_optimized_self_pumped_search(
    *,
    varactor: RealisticVaractor,
    f_signal_Hz: float,
    search_n_cells: bool = False,
    search_pump: bool = False,
    n_cells_grid: tuple[int, ...] | None = None,
    target_lambda: tuple[float, float] = (1.0, 4.0),
    n_cells_points: int = 5,
    pump_grid_dBm: tuple[float, ...] | None = None,
    regime_check: Callable[..., object] | None = None,
    regime_check_kwargs: Mapping[str, object] | None = None,
    n_finalists: int = 6,
    run_dir: str | Path | None = None,
    require_regime_check: bool = True,
    **design_kwargs,
) -> ExpandedSearchResult:
    """Stability-constrained search over geometry x N x pump.

    Adds length and pump as opt-in search dimensions on top of
    :func:`design_optimized_self_pumped`, enforcing the in-loop stability
    constraint (module docstring). ``**design_kwargs`` are forwarded verbatim to
    the wrapped engine for every grid point (e.g. ``f_pump_Hz``, ``substrate``,
    ``walk_down_top_k``, ``search_combined``). ``n_cells`` may be passed there to
    set the fixed length when ``search_n_cells=False``; ``pump_dBm_override`` is
    controlled by the pump search and must not be passed.

    Geometry screen fix. The engine's ``reject_signal_stopband``
    and ``refine`` flags also forward through ``**design_kwargs``. Because the
    expanded search re-optimises geometry at the *operating* frequency -- where the
    Stage-1 A1 ranking is stopband-biased and the analytical ``g_mod_in`` proxy
    would otherwise pick a Bloch-stopband cell with poor realized gain (-35 dB
    observed) -- pass ``reject_signal_stopband=True`` (and usually ``refine=True``)
    so every grid point selects a passband design. The stability constraint here is
    orthogonal to and complements that fix (it rejects the oscillator-regime
    finalists the high-gain passband optima approach). Both default OFF, so the
    existing pure-Python search tests are byte-identical.

    Parameters
    ----------
    search_n_cells, search_pump :
        Enable the length / pump dimensions. Both default ``False`` -> the grid
        is the single fixed operating point and the call reduces to the ordinary
        engine (no behaviour change).
    n_cells_grid, target_lambda, n_cells_points :
        The N grid. An explicit ``n_cells_grid`` overrides the physics helper;
        otherwise :func:`n_cells_grid_for_length` derives it from
        ``target_lambda`` (default 1..4 guided wavelengths) with ``n_cells_points``
        samples.
    pump_grid_dBm :
        The source-pump ladder [dBm]. ``None`` -> :func:`default_pump_ladder_dBm`
        (small-signal calibration folded into the +5..+17 dBm net-gain ladder).
    regime_check :
        In-loop stability classifier ``regime_check(spec, run_dir, **kwargs) ->
        AmplificationCheck-like``. ``None`` + an enabled expansion +
        ``require_regime_check`` (the default) raises -- a pump/N search ranked by
        the analytical proxy cannot certify stability and must not silently
        return a possible oscillator. :func:`ads_regime_check` is the production
        callback. Set ``require_regime_check=False`` only for an explicitly
        analytical-only (unconstrained) ranking.
    n_finalists :
        How many top-analytical grid points are ADS regime-checked (cost guard).
    run_dir :
        Working directory for the ADS checks (one subdir per finalist).

    Returns
    -------
    ExpandedSearchResult
        The constrained optimum (``selected``), the stable-only optimum
        (``best_stable``), the full grid, and the ADS-checked finalists.
    """

    if design_kwargs.get("objective", OBJECTIVE_G_MOD_IN) != OBJECTIVE_G_MOD_IN:
        raise ValueError(
            "the expanded search ranks by the analytical g_mod_in proxy; pass "
            "objective='g_mod_in' (the default) -- coh_pk does not populate it."
        )
    if "pump_dBm_override" in design_kwargs:
        raise ValueError(
            "pump_dBm_override is controlled by the pump search; pass the ladder "
            "via pump_grid_dBm / search_pump instead."
        )
    if search_n_cells and n_cells_grid is None and "n_cells" in design_kwargs:
        raise ValueError(
            "with search_n_cells=True the length is the searched grid; pass it "
            "via n_cells_grid / target_lambda, not n_cells."
        )

    design_kwargs.setdefault("objective", OBJECTIVE_G_MOD_IN)
    substrate = design_kwargs.get("substrate", ROGERS_4003C)
    fixed_n_cells = int(design_kwargs.pop("n_cells", 15))
    expansion_on = bool(search_n_cells or search_pump)

    # Crux guard: an expanded search ranked by the analytical proxy will climb
    # toward the instability edge, so refuse to run it without an in-loop
    # stability check rather than risk returning an oscillator-regime optimum.
    if expansion_on and require_regime_check and regime_check is None:
        raise ValueError(
            "expanded search (search_pump / search_n_cells) requires a "
            "regime_check callback for the in-loop stability constraint "
            "(use ads_regime_check, or pass require_regime_check=False for an "
            "explicitly unconstrained analytical-only ranking)."
        )

    # --- 1. length grid ---------------------------------------------------- #
    main_period_scaling_fn = design_kwargs.get(
        "main_period_scaling_fn", main_period_anchor_at_3p1GHz
    )
    main_period_m = float(main_period_scaling_fn(f_signal_Hz))
    if search_n_cells:
        n_grid = (
            tuple(int(n) for n in n_cells_grid)
            if n_cells_grid is not None
            else n_cells_grid_for_length(
                f_signal_Hz,
                main_period_m=main_period_m,
                substrate=substrate,
                target_lambda=target_lambda,
                n_points=n_cells_points,
            )
        )
    else:
        n_grid = (fixed_n_cells,)

    # --- 2. pump grid ------------------------------------------------------ #
    if search_pump:
        if pump_grid_dBm is not None:
            pump_grid: tuple[float | None, ...] = tuple(
                float(p) for p in pump_grid_dBm
            )
        else:
            f_pump = design_kwargs.get("f_pump_Hz", None)
            f_pump = float(f_pump) if f_pump is not None else 2.0 * float(f_signal_Hz)
            calib = calibrate_self_pumped_pump_dBm(
                varactor=varactor,
                f_pump_Hz=f_pump,
                m_eff_target=design_kwargs.get("m_eff_target", 0.22),
            )
            pump_grid = default_pump_ladder_dBm(calib)
    else:
        # None => design_optimized_self_pumped calibrates as usual (byte-identical).
        pump_grid = (None,)

    # --- 3. analytical screen over the grid -------------------------------- #
    grid_points: list[GridCandidate] = []
    for n_cells in n_grid:
        for pump in pump_grid:
            kw = dict(design_kwargs)
            kw["n_cells"] = int(n_cells)
            if pump is not None:
                kw["pump_dBm_override"] = float(pump)
            try:
                res = design_optimized_self_pumped(
                    varactor=varactor, f_signal_Hz=f_signal_Hz, **kw
                )
            except NoSelfPumpedCandidateError:
                # No candidate at this (N, pump) -- recorded as a skip, not fatal.
                continue
            grid_points.append(
                GridCandidate(
                    design=res,
                    n_cells=int(n_cells),
                    pump_dBm=float(res.calibrated_pump_dBm),
                    analytical_g_mod_in_dB=float(res.analytical_g_mod_in_dB),
                )
            )

    grid_points.sort(key=lambda c: _score(c, realised=False), reverse=True)

    # --- 4. ADS stability check on the finalists --------------------------- #
    finalists: list[GridCandidate] = []
    if regime_check is not None and grid_points:
        rk_kwargs = dict(regime_check_kwargs or {})
        base_dir = Path(run_dir) if run_dir is not None else None
        for i, cand in enumerate(grid_points[: max(0, int(n_finalists))]):
            sub = (base_dir / f"finalist_{i:02d}") if base_dir is not None else None
            if sub is not None:
                sub.mkdir(parents=True, exist_ok=True)
            check = regime_check(cand.design.spec, sub, **rk_kwargs)
            regime = regime_label_from_check(check)
            realized = float(getattr(check, "g_mod_in_dB", float("nan")))
            finalists.append(
                replace(
                    cand,
                    regime=regime,
                    admissible=regime in STABLE_OR_BOUNDARY,
                    realized_g_mod_in_dB=realized,
                    check=check,
                )
            )
    else:
        # Unconstrained analytical-only ranking (require_regime_check=False).
        finalists = list(grid_points[: max(0, int(n_finalists))])

    # --- 5. apply the constraint ------------------------------------------- #
    admissible = [c for c in finalists if c.admissible]
    selected = (
        max(admissible, key=lambda c: _score(c, realised=True)) if admissible else None
    )
    stable_only = [c for c in finalists if c.regime == REGIME_AMPLIFIER]
    best_stable = (
        max(stable_only, key=lambda c: _score(c, realised=True))
        if stable_only
        else None
    )

    if not grid_points:
        note = "no candidate found at any (N, pump) grid point."
    elif regime_check is None:
        note = (
            "analytical-only ranking (no regime_check); finalists are NOT "
            "stability-certified -- selected/best_stable left unset."
        )
        selected = None
        best_stable = None
    elif selected is None:
        note = (
            "no stable-or-boundary optimum among the checked finalists "
            "(all rejected: oscillator / non-convergent / indeterminate). "
            "This is a 'no stable optimum in this catalog' result, not an error."
        )
    elif selected.regime == REGIME_BOUNDARY:
        note = (
            "optimum is a BOUNDARY (band-edge) point -- near the parametric "
            "instability threshold, not a comfortably-stable amplifier; "
            "best_stable is the amplifier-only optimum."
        )
    else:
        note = "optimum is a stable amplifier."

    return ExpandedSearchResult(
        selected=selected,
        best_stable=best_stable,
        n_cells_grid=tuple(n_grid),
        pump_grid_dBm=tuple(pump_grid),
        target_lambda=(float(target_lambda[0]), float(target_lambda[1])),
        finalists=tuple(finalists),
        all_grid_points=tuple(grid_points),
        note=note,
    )


