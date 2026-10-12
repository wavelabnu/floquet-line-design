"""Top-level decoupled TVTL design-workflow orchestrator.

Runs the four decoupled stages in sequence for one ``(design frequency,
varactor)`` cell and returns cross-validated design examples:

1. Stage 1 -- coarse Sobol screen + Nelder-Mead refine (per-cell stub, A1);
2. **top-M diversification** (greedy: different topology family, or same family
   with normalized parameter distance above a threshold) so the examples are
   genuinely different designs, not parameter-noise variants of one design
   (addressing the top-K degeneracy finding);
3. Stage 2 -- Bloch-impedance match of the main line (realizable candidates);
4. Stage 3 -- per-side distributed matching network;
5. Stage 4 -- coherent-gain / PSA / landscape characterization (+ optional ADS).

The decoupling is preserved: each stage consumes the previous stage's output and
a fixed config; there is no unified cross-stage objective.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from tvtl.varactors import RealisticVaractor

from .stage1_per_cell import StubCandidate, coarse_screen, refine_topk
from .stage2_main_tl import MainTLDesign, match_z_tl
from .stage3_matching import edge_impedance, select_matching
from .stage4_validation import (
    CompleteDesign,
    CrossValReport,
    analytical_characterization,
    cross_validate,
)
from .substrate import ROGERS_4003C, SubstrateConfig
from .topology_generator import (
    LENGTH_MAX_M,
    LENGTH_MIN_M,
    WIDTH_MAX_M,
    WIDTH_MIN_M,
    TopologyFamily,
    linear_families,
)

__all__ = [
    "WorkflowConfig",
    "WorkflowResult",
    "param_distance",
    "select_diverse_top_m",
    "decoupled_design_workflow",
]


@dataclass(frozen=True)
class WorkflowConfig:
    """Configuration for one workflow run."""

    substrate: SubstrateConfig = ROGERS_4003C
    order_by: str = "absolute_outin_dB"
    gate: str = "none"
    gate_options: dict | None = None
    n_samples: int = 300         # Sobol budget per family
    seed: int = 0
    top_k: int = 10              # refined candidates considered
    top_m: int = 3               # diversified design examples
    n_cells: int = 15
    drive_dBm: float = -3.0
    refine_maxiter: int = 200
    param_distance_threshold: float = 0.30  # normalized RMS diversification threshold
    min_coherent_dB: float = 0.0  # drop deep-loss "A1-ceiling" candidates
    # Recommended Stage-4 selection criterion: a moderate-gain
    # target window [lo, hi] dB on the analytical coherent peak, instead of
    # gain-maximization. Empirically, gain-max drives designs into
    # parametric oscillation regardless of termination; the cross-validatable
    # regime is moderate gain. None -> gain-max behavior.
    coherent_window_dB: tuple[float, float] | None = None
    # In window mode, walk down the A1-ranked screen (not the refined top-K,
    # which refine pulls to the A1 ceiling = oscillation) probing each
    # candidate's Stage-4 gain; cap the walk for compute.
    walk_cap: int = 1200
    pumps_dBm: tuple[float, ...] = (-6.0, -3.0, 0.0)
    phases_deg: tuple[float, ...] = (0.0, 72.0, 144.0, 216.0, 288.0)
    run_ads: bool = False
    ads_tmp_dir: str | None = None
    families: tuple[TopologyFamily, ...] | None = None  # default: 12 linear


@dataclass
class WorkflowResult:
    """All design examples for one (frequency, varactor) cell + metadata."""

    f0_Hz: float
    reports: list[CrossValReport]
    n_screened: int
    n_realizable: int
    n_positive_gain: int
    n_diverse: int
    note: str = ""


def param_distance(a: StubCandidate, b: StubCandidate) -> float:
    """Normalized RMS parameter distance in ``[0, 1]`` (``inf`` if families differ).

    Each (width, length) coordinate is normalized to its box range; the distance
    is the root-mean-square of the normalized coordinate differences, so it is
    comparable across topologies of the same arity and reads as a "fraction of
    parameter range".
    """

    if a.topology != b.topology:
        return float("inf")
    pa = a.params_array
    pb = b.params_array
    span = np.empty_like(pa)
    span[0::2] = WIDTH_MAX_M - WIDTH_MIN_M
    span[1::2] = LENGTH_MAX_M - LENGTH_MIN_M
    norm = (pa - pb) / span
    return float(np.sqrt(np.mean(norm ** 2)))


def select_diverse_top_m(
    candidates: list[StubCandidate],
    m: int = 3,
    param_distance_threshold: float = 0.30,
) -> list[StubCandidate]:
    """Greedy diversified top-M selection (Addition 1).

    Always takes the highest-ranked candidate; adds a subsequent candidate only
    if it differs from every already-selected one by either a different topology
    family or a normalized parameter distance above the threshold. Input is
    assumed ordered best-first.
    """

    if not candidates:
        return []
    selected: list[StubCandidate] = [candidates[0]]
    for cand in candidates[1:]:
        if len(selected) >= m:
            break
        if all(
            cand.topology != s.topology
            or param_distance(cand, s) > param_distance_threshold
            for s in selected
        ):
            selected.append(cand)
    return selected


def _build_and_probe(cand, varactor, cfg, f0_Hz):
    """Stage 2 (match) + Stage 3 (matching) + Stage-4 analytical gain probe.

    Returns ``(cand, mtl, input_match, output_match, coh_peak_dB)`` or ``None``
    if the candidate is not Z_TL-realizable (Stage 2). ``coh_peak_dB`` is the
    analytical coherent peak at the design pump -- the faithful selection metric.
    """

    mtl = match_z_tl(cand, varactor, substrate=cfg.substrate, f0_Hz=f0_Hz)
    if not mtl.realizable:
        return None
    z_in, z_out = edge_impedance(mtl, varactor, cfg.n_cells, substrate=cfg.substrate)
    sm_in = select_matching(z_in, "input", f0_Hz, substrate=cfg.substrate)
    sm_out = select_matching(z_out, "output", f0_Hz, substrate=cfg.substrate)
    im = None if not sm_in.needed else (
        sm_in.single_stub if sm_in.recommended == "single_stub" else sm_in.quarter_wave
    )
    om = None if not sm_out.needed else (
        sm_out.single_stub if sm_out.recommended == "single_stub" else sm_out.quarter_wave
    )
    design = CompleteDesign(
        stub=cand, main_tl=mtl, input_match=im, output_match=om,
        varactor=varactor, n_cells=cfg.n_cells, f0_Hz=f0_Hz, substrate=cfg.substrate,
    )
    probe = analytical_characterization(
        design, pumps_dBm=(cfg.drive_dBm,), phases_deg=(0.0,),
        landscape_GHz=(round(f0_Hz / 1e9, 3),),
    )
    return (cand, mtl, im, om, float(probe.analytical[float(cfg.drive_dBm)][cfg.order_by]))


def decoupled_design_workflow(
    f0_Hz: float,
    varactor: RealisticVaractor,
    cfg: WorkflowConfig = WorkflowConfig(),
) -> WorkflowResult:
    """Run Stages 1-4 for one (frequency, varactor) cell (decoupled).

    Two selection modes (both keep Stage 1 optimizing A1; the orchestrator
    selects by Stage-4 analytical gain):

    * **gain-max** (``coherent_window_dB=None``): refine the A1 top-K, probe,
      drop deep-loss, gain-rank, diversify.
    * **gain-target-window walk-down** (``coherent_window_dB=(lo, hi)``): refine
      maximizes A1 (correlated with oscillation), so instead **walk down the
      A1-ranked screen** probing each candidate's Stage-4 gain, selecting the
      first ``top_m`` diverse candidates with gain in ``[lo, hi]`` dB -- the
      moderate, cross-validatable regime the screen identifies but refine skips.
    """

    if cfg.order_by not in ("absolute_outin_dB", "coherent_peak"):
        raise ValueError("order_by must be absolute_outin_dB or coherent_peak")
    families = list(cfg.families) if cfg.families is not None else linear_families()
    factory = None
    gate_options = dict(cfg.gate_options or {})
    bias_return_ohm = gate_options.pop("bias_return_ohm", None)   # a cell option, not a gate threshold
    if cfg.gate in ("stage0", "cascade"):
        from tvtl.design.feasibility import candidate_cell_factory
        factory = candidate_cell_factory(varactor=varactor, substrate=cfg.substrate,
                                         f_s=f0_Hz, N=cfg.n_cells, bias_return_ohm=bias_return_ohm)
    screen = coarse_screen(
        f0_Hz, varactor, substrate=cfg.substrate, families=families,
        n_samples=cfg.n_samples, seed=cfg.seed, top_k=cfg.top_k, drive_dBm=cfg.drive_dBm,
        gate=cfg.gate, cell_factory=factory, gate_options=dict(N=cfg.n_cells, **gate_options),
    )
    n_screened_linear = len([c for c in screen.scored if c.topology.is_linear])

    if cfg.coherent_window_dB is not None:
        # WALK-DOWN: probe the A1-ranked screen, select in-window + diverse.
        lo, hi = cfg.coherent_window_dB
        ranked = [c for c in screen.scored if c.topology.is_linear]  # A1-descending
        diverse: list = []
        diverse_cands: list[StubCandidate] = []
        n_realizable = 0
        for cand in ranked[: cfg.walk_cap]:
            ev = _build_and_probe(cand, varactor, cfg, f0_Hz)
            if ev is None:
                continue
            n_realizable += 1
            coh = ev[4]
            if not (np.isfinite(coh) and lo <= coh <= hi):
                continue
            if all(
                cand.topology != s.topology
                or param_distance(cand, s) > cfg.param_distance_threshold
                for s in diverse_cands
            ):
                diverse.append(ev)
                diverse_cands.append(cand)
                if len(diverse) >= cfg.top_m:
                    break
        n_selected = len(diverse)
        sel_desc = f"window {cfg.coherent_window_dB} dB (walk-down; {n_realizable} probed-realizable)"
    else:
        # GAIN-MAX: refine the A1 top-K, probe, drop deep-loss, gain-rank, diversify.
        refined = refine_topk(
            screen.linear_topk, f0_Hz, varactor, substrate=cfg.substrate,
            top_n=cfg.top_k, maxiter=cfg.refine_maxiter, drive_dBm=cfg.drive_dBm,
        )
        built = [_build_and_probe(c, varactor, cfg, f0_Hz) for c in refined]
        built = [b for b in built if b is not None]
        n_realizable = len(built)
        selected = [b for b in built if np.isfinite(b[4]) and b[4] > cfg.min_coherent_dB]
        selected.sort(key=lambda b: b[4], reverse=True)
        diverse_cands = select_diverse_top_m(
            [b[0] for b in selected], m=cfg.top_m,
            param_distance_threshold=cfg.param_distance_threshold,
        )
        by_cand = {id(b[0]): b for b in selected}
        diverse = [by_cand[id(c)] for c in diverse_cands]
        n_selected = len(selected)
        sel_desc = f"min_coherent={cfg.min_coherent_dB:g} dB"

    # Full cross-validation (analytical + optional ADS) on the selected examples. Before any HB job, a
    # candidate whose delivered depth at the largest requested pump exceeds the C(V) ceiling at the
    # varactor's bias is refused (delivered_depth_above_ceiling) and keeps its analytical report only.
    reports: list[CrossValReport] = []
    depth_factory = None
    if cfg.run_ads:
        from tvtl.design.feasibility import candidate_cell_factory
        depth_factory = factory or candidate_cell_factory(
            varactor=varactor, substrate=cfg.substrate, f_s=f0_Hz, N=cfg.n_cells,
            bias_return_ohm=bias_return_ohm)
    for cand, mtl, im, om, _coh in diverse:
        design = CompleteDesign(
            stub=cand, main_tl=mtl, input_match=im, output_match=om,
            varactor=varactor, n_cells=cfg.n_cells, f0_Hz=f0_Hz, substrate=cfg.substrate,
        )
        refusal = None
        if cfg.run_ads:
            from tvtl.design.depth_ceiling import depth_ceiling_check
            check = depth_ceiling_check(
                depth_factory(cand.topology, cand.params), f0_Hz, N=cfg.n_cells,
                P_pump_dBm=max(cfg.pumps_dBm), bias_V=float(varactor.V_bias_v),
                v_max_V=(cfg.gate_options or {}).get("v_max_V"))
            refusal = None if check["ok"] else check
        report = cross_validate(
            design, pumps_dBm=cfg.pumps_dBm, phases_deg=cfg.phases_deg,
            gated_pump_dBm=cfg.drive_dBm, run_ads=cfg.run_ads and refusal is None,
            ads_tmp_dir=cfg.ads_tmp_dir,
        )
        if refusal is not None:
            report.refusal = refusal
            report.passed = False
            report.note = (f"refused before HB: {refusal['reason']} (max junction pump "
                           f"{refusal['max_junction_pump_V']:.3g} V > swing limit {refusal['swing_limit_V']:.3g} V)")
        reports.append(report)

    return WorkflowResult(
        f0_Hz=float(f0_Hz), reports=reports, n_screened=n_screened_linear,
        n_realizable=n_realizable, n_positive_gain=n_selected, n_diverse=len(diverse),
        note=f"{len(reports)} design example(s) from {n_selected} selected ({sel_desc}).",
    )


