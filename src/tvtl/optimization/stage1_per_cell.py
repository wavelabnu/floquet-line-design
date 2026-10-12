"""Stage 1 -- per-cell stub optimization (coarse screen + refinement).

Implements hybrid pruning -- a coarse Sobol screen followed by continuous
refinement -- over the generated topology families, scored by a
:class:`~tvtl.optimization.figure_of_merit.FigureOfMerit` (the calibrated
Lorentzian-guard A1 by default; Gaussian and A2 selectable). The
stage is decoupled: it consumes only a topology, a parameter vector, the design
frequency, and the varactor/substrate config -- no cascade context.

Pipeline (per design frequency):

* :func:`coarse_screen` -- Sobol ``N = 300``/family over all 20 families; rank
  the 12 *linear* families' candidates -> ``top_k = 10`` for refinement; rank
  the 8 *COMBINED* families separately and flag any within 20 % of the top
  linear A1 (analytical-only; flagged for future ADS cross-validation).
* :func:`refine_topk` -- ``scipy.optimize.minimize`` (Nelder-Mead, box bounds)
  on each top-K candidate's continuous parameters -> ``top_n = 5``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from functools import partial

import numpy as np
from scipy.optimize import Bounds, minimize

from tvtl.varactors import RealisticVaractor

from .figure_of_merit import DEFAULT_DRIVE_DBM, FigureOfMerit, LorentzianA1
from .substrate import ROGERS_4003C, SubstrateConfig
from .topology_generator import (
    TopologyFamily,
    enumerate_families,
    is_valid,
    param_bounds,
    sample_family,
)

__all__ = [
    "StubCandidate",
    "CoarseScreenResult",
    "coarse_screen",
    "refine_topk",
    "COMBINED_FLAG_FRACTION",
]

# A COMBINED candidate is flagged (analytical-screening-only family) if its A1
# is within this fraction of the top *linear* A1.
COMBINED_FLAG_FRACTION: float = 0.80


@dataclass(frozen=True)
class StubCandidate:
    """A scored per-cell stub design (topology + parameter vector + A1)."""

    topology: TopologyFamily
    params: tuple[float, ...]
    a1_score: float
    f0_Hz: float

    @property
    def params_array(self) -> np.ndarray:
        return np.asarray(self.params, dtype=float)


@dataclass
class CoarseScreenResult:
    """Outcome of :func:`coarse_screen`."""

    scored: list[StubCandidate]            # every scored candidate, desc by A1
    linear_topk: list[StubCandidate]       # top-K linear (the refinement/ADS path)
    combined_flagged: list[StubCandidate]  # COMBINED within 20% of top linear A1
    n_samples_per_family: int
    top_linear_score: float = field(default=0.0)
    feasibility_records: list[dict] = field(default_factory=list)


def coarse_screen(
    f0_Hz: float,
    varactor: RealisticVaractor,
    *,
    fom: FigureOfMerit | None = None,
    substrate: SubstrateConfig = ROGERS_4003C,
    families: list[TopologyFamily] | None = None,
    n_samples: int = 300,
    seed: int = 0,
    drive_dBm: float = DEFAULT_DRIVE_DBM,
    top_k: int = 10,
    gate: str = "none",
    cell_factory=None,
    gate_options: dict | None = None,
) -> CoarseScreenResult:
    """Phase A: Sobol coarse screen over the topology families.

    Parameters
    ----------
    f0_Hz : float
        Design (signal) frequency.
    varactor : RealisticVaractor
        Curved realistic varactor.
    fom : FigureOfMerit, optional
        Scorer (default :class:`ClosedFormA1`).
    substrate : SubstrateConfig
        Substrate configuration.
    families : list[TopologyFamily], optional
        Families to screen (default all 20).
    n_samples : int
        Sobol budget per family (default 300).
    seed : int
        Sobol seed (per-family seed is ``seed + family_index`` for independence).
    drive_dBm : float
        Standardized control-port drive for A1 (default -3 dBm).
    top_k : int
        Number of top *linear* candidates to keep for refinement.

    Returns
    -------
    CoarseScreenResult
    """

    if gate not in ("none", "stage0", "cascade"):
        raise ValueError("gate must be none, stage0 or cascade")
    if gate != "none" and cell_factory is None:
        raise ValueError(f"{gate} requires cell_factory(topology, params) for the actual through-line cell")
    if gate == "cascade":
        # the stage-0 gates plus the cascade phase margin 0.5 m_del - m_EP >= 0 and the delivered-depth
        # ceiling at the varactor's bias point (optional gate option v_max_V, the part's reverse ceiling)
        gate_options = dict(gate_options or {}, phase_margin=True)
        v_max = gate_options.pop("v_max_V", None)
        if "depth_ceiling" not in gate_options:
            if getattr(varactor, "V_bias_v", None) is None:
                raise ValueError("cascade gate needs the varactor's bias or an explicit depth_ceiling")
            gate_options["depth_ceiling"] = {"bias_V": float(varactor.V_bias_v), "v_max_V": v_max}
    gate_records = []
    if fom is None:
        fom = LorentzianA1()  # calibrated default (x0=0.25, gamma=0.20)
    if families is None:
        families = enumerate_families()

    scored: list[StubCandidate] = []
    for idx, fam in enumerate(families):
        for p in sample_family(
            fam, n_samples, substrate=substrate, seed=seed + idx, f_ref_hz=f0_Hz
        ):
            if gate != "none":
                from tvtl.design.feasibility import feasibility
                try:
                    check = feasibility(cell_factory(fam, p), f0_Hz, **(gate_options or {}))
                except (ValueError, np.linalg.LinAlgError) as exc:
                    # refusals that carry a reason (e.g. no_dc_return_for_varactor) record it verbatim
                    reason = getattr(exc, "reason", None)
                    gate_records.append(dict(topology=fam.key, params=list(map(float,p)),
                                             feasible=False, reasons=[reason or str(exc)],
                                             **({"message": str(exc)} if reason else {})))
                    continue
                gate_records.append(dict(topology=fam.key, params=list(map(float,p)), **check))
                if not check["feasible"]:
                    continue
            s = fom.score(
                fam, p, f0_Hz=f0_Hz, varactor=varactor,
                substrate=substrate, drive_dBm=drive_dBm,
            )
            if np.isfinite(s):
                scored.append(
                    StubCandidate(
                        topology=fam, params=tuple(float(v) for v in p),
                        a1_score=float(s), f0_Hz=float(f0_Hz),
                    )
                )

    scored.sort(key=lambda c: c.a1_score, reverse=True)
    linear = [c for c in scored if c.topology.is_linear]
    combined = [c for c in scored if not c.topology.is_linear]
    linear_topk = linear[:top_k]
    top_lin = linear_topk[0].a1_score if linear_topk else 0.0
    flagged = [
        c for c in combined if c.a1_score >= COMBINED_FLAG_FRACTION * top_lin
    ] if top_lin > 0.0 else []

    return CoarseScreenResult(
        scored=scored, linear_topk=linear_topk, combined_flagged=flagged,
        n_samples_per_family=int(n_samples), top_linear_score=float(top_lin),
        feasibility_records=gate_records,
    )


def _neg_log_score(
    x: np.ndarray,
    *,
    topology: TopologyFamily,
    fom: FigureOfMerit,
    f0_Hz: float,
    varactor: RealisticVaractor,
    substrate: SubstrateConfig,
    drive_dBm: float,
) -> float:
    """Refinement objective: ``-log10(A1)`` with an out-of-box penalty.

    A1 spans many decades through the Gaussian guard, so the log makes the
    landscape smooth/quadratic for Nelder-Mead. Invalid points (outside the
    width/length box or the Z0 realizability band) return a large finite
    penalty rather than +inf so the simplex stays well-behaved.
    """

    if not is_valid(topology, x, substrate=substrate, f_ref_hz=f0_Hz):
        return 1.0e3
    s = fom.score(
        topology, x, f0_Hz=f0_Hz, varactor=varactor,
        substrate=substrate, drive_dBm=drive_dBm,
    )
    if not np.isfinite(s) or s <= 0.0:
        return 1.0e3
    return -float(np.log10(s))


def refine_topk(
    candidates: list[StubCandidate],
    f0_Hz: float,
    varactor: RealisticVaractor,
    *,
    fom: FigureOfMerit | None = None,
    substrate: SubstrateConfig = ROGERS_4003C,
    drive_dBm: float = DEFAULT_DRIVE_DBM,
    top_n: int = 5,
    maxiter: int = 200,
) -> list[StubCandidate]:
    """Phase B: continuous Nelder-Mead refinement of each top-K candidate.

    Each candidate's continuous ``(width, length)`` vector is refined to
    maximize A1 (within the family's box bounds), starting from the coarse
    sample. Returns the refined candidates re-scored and sorted desc, truncated
    to ``top_n``. A refinement that fails to improve keeps the coarse vector.
    """

    if fom is None:
        fom = LorentzianA1()  # calibrated default (x0=0.25, gamma=0.20)
    refined: list[StubCandidate] = []
    for cand in candidates:
        topo = cand.topology
        x0 = cand.params_array
        lo, hi = param_bounds(topo)
        objective = partial(
            _neg_log_score, topology=topo, fom=fom, f0_Hz=f0_Hz,
            varactor=varactor, substrate=substrate, drive_dBm=drive_dBm,
        )
        result = minimize(
            objective, x0, method="Nelder-Mead",
            bounds=Bounds(lo, hi),
            options={"maxiter": int(maxiter), "xatol": 1e-7, "fatol": 1e-4},
        )
        x_best = np.asarray(result.x, dtype=float)
        # Keep the refined point only if valid and not worse than the start.
        s_start = fom.score(topo, x0, f0_Hz=f0_Hz, varactor=varactor,
                            substrate=substrate, drive_dBm=drive_dBm)
        if is_valid(topo, x_best, substrate=substrate, f_ref_hz=f0_Hz):
            s_best = fom.score(topo, x_best, f0_Hz=f0_Hz, varactor=varactor,
                               substrate=substrate, drive_dBm=drive_dBm)
        else:
            s_best = -np.inf
        if np.isfinite(s_best) and s_best >= s_start:
            x_keep, s_keep = x_best, s_best
        else:
            x_keep, s_keep = x0, s_start
        refined.append(
            StubCandidate(
                topology=topo, params=tuple(float(v) for v in x_keep),
                a1_score=float(s_keep), f0_Hz=float(f0_Hz),
            )
        )

    refined.sort(key=lambda c: c.a1_score, reverse=True)
    return refined[:top_n]


