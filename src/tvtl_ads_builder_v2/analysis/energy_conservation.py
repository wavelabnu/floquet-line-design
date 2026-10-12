"""Per-cell energy-conservation cross-check on an HB dataset.

Consumes the power helpers in
:mod:`.power` and the per-cell map in :mod:`.per_cell` and
exposes a single physical conservation law:

    ΔP_signal(k) ≡ P_boundary_out(k) - P_boundary_in(k) ≈ -P_var(k)

where ``P_boundary_in(k)`` and ``P_boundary_out(k)`` are the time-averaged
real powers flowing into and out of cell ``k`` on the through-line at the
signal mixing tuple, and ``P_var(k)`` is the time-averaged real power
flowing from the through-line into the cell's shunt branch (varactor +
stub series chain). Energy balance at the cell's branch tap yields
``P_in - P_out = P_var + P_dissipated_in_through_line``; in the
low-through-line-loss limit the dissipation term is small and the cell
sees ``ΔP_signal ≈ -P_var``.

For a passive cell ``P_var > 0`` (the branch absorbs from the through
line) and ``ΔP_signal < 0`` (loss). For a parametrically-amplifying cell
the modulated varactor sources power back to the through line via the
mixing products, so ``P_var < 0`` at the signal harmonic and ``ΔP_signal
> 0`` (gain). The conservation law is signed so both cases pass the same
check; the residual ``r(k) = ΔP_signal(k) + P_var(k)`` should be small
relative to the input power.

Why this is a useful cross-check
--------------------------------
The per-cell power helpers (``cell_boundary_power``, ``varactor_power``)
are independent reads of the HB dataset: boundary power couples the
through-line I_Probe current to the cell-facing node voltage, while
varactor power couples the in-series varactor probe to the branch-top
voltage. Their agreement under the cell-level energy balance is a
physical sanity check on (i) the peak-phasor (``1/2``) convention, (ii)
the I_Probe current orientation (pin-1 -> pin-2 direction), (iii) the
V-node choice at each measurement, and (iv) the ``Re(V * conj(I))``
formula consistency. A failure here therefore localises the bug
without ambiguity to one of those four conventions.

API
---
Two functions:

* :func:`cell_energy_residual` -- one cell, one mix; returns the
  residual ``r(k)`` and its dB-relative-to-input-power magnitude.
* :func:`cascade_energy_residuals` -- every cell, summary statistics.

Both accept an optional ``tolerance_dB_rel_input`` parameter; the
default 1% (``-20 dB rel input``) reflects the originally-set
budget.

Sign convention
---------------
``ΔP_signal(k)`` is computed as ``P_out - P_in`` (positive gain). The
conservation law adds ``+P_var(k)`` (positive into the branch), so the
residual is::

    r(k) = ΔP_signal(k) + P_var(k)

which should be near zero. ``residual_dB_rel_input`` is
``20 * log10(|r(k)| / |P_input|)`` with ``P_input`` the design-input
boundary power (``cell_boundary_power(boundary_index=0)``) -- a
reference magnitude every cell shares, so the per-cell residuals are
comparable across the cascade.
"""

from __future__ import annotations

import math
from typing import Mapping

from .hb_dataset import HBDataset, SIGNAL_MIX_2TONE
from .per_cell import CellMap, cell_count
from .power import cell_boundary_power, varactor_power

__all__ = [
    "DEFAULT_ENERGY_TOLERANCE_DB_REL_INPUT",
    "cascade_energy_residuals",
    "cell_energy_residual",
]


# Default tolerance: 1% of the design-input power. In dB,
# 20 * log10(0.01) = -40 dB. (The residual is referenced to a power
# magnitude, so a "1% of input power" budget means |r| / |P_in| < 0.01,
# i.e. residual_dB_rel_input < -40 dB.)
DEFAULT_ENERGY_TOLERANCE_DB_REL_INPUT: float = -40.0


def _residual_dB(residual_W: float, p_reference_W: float) -> float:
    """Return ``20 * log10(|r| / |P_ref|)`` with safe handling of zeros.

    ``20 * log10`` (not ``10 * log10``) is used because the residual and
    the reference are both real-power magnitudes; we report the *power-
    magnitude ratio* in dB on a 20-log-power-fraction convention so a
    "1% of input power" budget corresponds to -40 dB. The convention is
    intentional: a 10-log convention would compare the residual on a
    power-flux dB scale and ``1% of input power`` would map to -20 dB
    -- which obscures the fact that ``|r|`` is a power, not a power
    *ratio*. The 20-log convention treats the residual as a fractional
    correction to ``P_input`` and reads in dB the same way a voltage-
    ratio gain does.

    Returns ``-inf`` if the residual is exactly zero (perfect
    conservation), and ``+inf`` if the reference is zero but the
    residual is not (singular).
    """

    r = abs(float(residual_W))
    pref = abs(float(p_reference_W))
    if r == 0.0:
        return float("-inf")
    if pref == 0.0:
        return float("+inf")
    return 20.0 * math.log10(r / pref)


def cell_energy_residual(
    dataset: HBDataset,
    cell_map: CellMap,
    cell_index: int,
    mix: tuple[int, ...] = SIGNAL_MIX_2TONE,
    *,
    tolerance_dB_rel_input: float = DEFAULT_ENERGY_TOLERANCE_DB_REL_INPUT,
) -> Mapping[str, float | bool]:
    """Return cell ``cell_index``'s energy-conservation residual at ``mix``.

    Computes
    ``ΔP_signal(k) = P_boundary_out(k) - P_boundary_in(k)``,
    ``P_var(k)`` from :func:`.power.varactor_power`, and the residual
    ``r(k) = ΔP_signal(k) + P_var(k)``. The residual is reported
    absolutely (W), in dB relative to the design-input boundary power,
    and as a boolean ``within_tolerance`` flag against the supplied
    ``tolerance_dB_rel_input`` budget.

    Parameters
    ----------
    dataset : HBDataset
        The HB result, peak-phasor indexed by mixing tuple.
    cell_map : CellMap
        The resolved cascade (must carry boundary and varactor I_Probe
        ids -- Level 9+).
    cell_index : int
        0-based cell position in the cascade.
    mix : tuple of int, optional
        Mixing tuple at which to evaluate the balance. Defaults to the
        signal product ``(1, 0)``.
    tolerance_dB_rel_input : float, optional
        Budget in dB on ``|r| / |P_input|``. Default is
        :data:`DEFAULT_ENERGY_TOLERANCE_DB_REL_INPUT` (-40 dB = 1%).

    Returns
    -------
    dict
        Keys:

        * ``residual`` (float, W) -- ``ΔP_signal + P_var``.
        * ``residual_dB_rel_input`` (float, dB) --
          ``20 * log10(|r| / |P_input|)`` with safe -inf/+inf handling.
        * ``P_boundary_in`` (float, W) -- ``P_boundary_power(k)``.
        * ``P_boundary_out`` (float, W) -- ``P_boundary_power(k + 1)``.
        * ``P_var`` (float, W) -- :func:`.power.varactor_power`.
        * ``delta_P_signal`` (float, W) -- ``P_out - P_in``.
        * ``P_input_reference`` (float, W) -- design-input boundary
          power (``P_boundary_power(0)``).
        * ``within_tolerance`` (bool).
    """

    n_cells = cell_count(cell_map)
    if not 0 <= cell_index < n_cells:
        raise IndexError(
            f"cell_index {cell_index} out of range for a "
            f"{n_cells}-cell cascade."
        )

    p_in = cell_boundary_power(dataset, cell_map, cell_index, mix)
    p_out = cell_boundary_power(dataset, cell_map, cell_index + 1, mix)
    p_var = varactor_power(dataset, cell_map, cell_index, mix)
    delta_p = p_out - p_in
    residual = delta_p + p_var

    p_input_reference = cell_boundary_power(dataset, cell_map, 0, mix)
    residual_dB = _residual_dB(residual, p_input_reference)
    within_tolerance = residual_dB <= float(tolerance_dB_rel_input)

    return {
        "residual": float(residual),
        "residual_dB_rel_input": float(residual_dB),
        "P_boundary_in": float(p_in),
        "P_boundary_out": float(p_out),
        "P_var": float(p_var),
        "delta_P_signal": float(delta_p),
        "P_input_reference": float(p_input_reference),
        "within_tolerance": bool(within_tolerance),
    }


def cascade_energy_residuals(
    dataset: HBDataset,
    cell_map: CellMap,
    mix: tuple[int, ...] = SIGNAL_MIX_2TONE,
    *,
    tolerance_dB_rel_input: float = DEFAULT_ENERGY_TOLERANCE_DB_REL_INPUT,
) -> Mapping[str, object]:
    """Return per-cell + cascade-summary energy residuals at ``mix``.

    Evaluates :func:`cell_energy_residual` at every cell and aggregates.

    Returns
    -------
    dict
        Keys:

        * ``per_cell`` (tuple of dict) -- one
          :func:`cell_energy_residual` result per cell, in cell-index
          order.
        * ``cascade_total`` (float, W) -- ``P_design_out -
          P_design_in - sum_k P_var(k)``. By the same balance applied
          cell by cell this should be small. Equivalent to
          ``sum_k r(k)`` modulo intra-cell through-line dissipation.
        * ``cascade_total_dB_rel_input`` (float, dB) --
          ``20 * log10(|cascade_total| / |P_input|)``.
        * ``max_abs_residual`` (float, W).
        * ``max_residual_dB_rel_input`` (float, dB) -- worst per-cell
          ``residual_dB_rel_input``.
        * ``within_tolerance`` (bool) -- True iff every cell's residual
          satisfies the budget.
    """

    n_cells = cell_count(cell_map)
    if n_cells == 0:
        raise ValueError("cascade_energy_residuals: cascade has no cells.")

    per_cell = tuple(
        cell_energy_residual(
            dataset,
            cell_map,
            k,
            mix,
            tolerance_dB_rel_input=tolerance_dB_rel_input,
        )
        for k in range(n_cells)
    )

    p_design_in = cell_boundary_power(dataset, cell_map, 0, mix)
    p_design_out = cell_boundary_power(dataset, cell_map, n_cells, mix)
    sum_p_var = sum(float(entry["P_var"]) for entry in per_cell)
    cascade_total = (p_design_out - p_design_in) + sum_p_var
    cascade_total_dB = _residual_dB(cascade_total, p_design_in)

    abs_residuals = tuple(abs(float(entry["residual"])) for entry in per_cell)
    max_abs_residual = max(abs_residuals) if abs_residuals else 0.0
    max_residual_dB = max(
        float(entry["residual_dB_rel_input"]) for entry in per_cell
    )
    within_tolerance = all(bool(entry["within_tolerance"]) for entry in per_cell)

    return {
        "per_cell": per_cell,
        "cascade_total": float(cascade_total),
        "cascade_total_dB_rel_input": float(cascade_total_dB),
        "max_abs_residual": float(max_abs_residual),
        "max_residual_dB_rel_input": float(max_residual_dB),
        "within_tolerance": bool(within_tolerance),
    }


