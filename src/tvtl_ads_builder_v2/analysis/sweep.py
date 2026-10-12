"""Aggregate a :class:`SweepResult` into the Figure-6 gain curves.

Two 1-D curves, each available from two sources:

* ``source="ads"`` -- end-to-end cascade gain read from each point's live
  HB dataset. On the paper Level-10 the per-cell through nets are not
  NAMEd (the generator exposes only ``n_input`` / ``n_c{k}_b`` /
  ``n_c{k}_v`` / ``n_output``), so the existing
  :func:`cascade_power_gain_dB` -- which reads the un-NAMEd cell input /
  output nets -- cannot run. This module instead computes the end-to-end
  gain from the NAMEd design-boundary nets and the boundary I_Probes:

      P_in  = 0.5 * Re(V(n_input)  * conj(I(IPR_IN)))
      P_out = 0.5 * Re(V(n_output) * conj(I(IPR_OUT)))

  ``V(n_input) == V(n_pin)`` and ``V(n_output) == V(n_c{N}_out)`` because
  the boundary I_Probes are 0-ohm shorts, so this is electrically the
  same end-to-end gain, computed only from exported quantities.

* ``source="analytical"`` -- the harmonic-ABCD-with-lift signal proxy
  (Approach A, pump-informed) from
  :func:`tvtl_ads_builder_v2.analytical.level10_signal_gain_proxy_dB`,
  reading ``V_pump`` per cell from the same dataset.

``cell_index`` defaults to ``None`` (end-to-end). A non-``None``
``cell_index`` requests per-cell gain, which needs the cell's through
nets to be NAMEd -- true for Level-9 fully-NAMEd cascades, NOT for the
paper Level-10 -- so it raises a clear error there.
"""

from __future__ import annotations

import math

import numpy as np

from ..device_spec import DeviceSpec
from .hb_dataset import HBDataset
from .per_cell import (
    CellMap,
    build_cell_map,
    cell_boundary_current,
    cell_count,
)
from .power import cell_power_gain_dB, cell_voltage_ratio_gain_dB

__all__ = [
    "gain_vs_signal_frequency",
    "gain_vs_pump_power",
]

_GAIN_CONVENTIONS = frozenset({"power", "voltage"})
_SOURCES = frozenset({"ads", "analytical"})


def _boundary_voltage_net(
    spec: DeviceSpec, dataset: HBDataset, probe_id: str
) -> str:
    """Return the boundary I_Probe's net that the dataset exports a voltage for.

    A boundary I_Probe is a 0-ohm short spanning two nets (e.g. IPR_IN
    spans ``n_input`` and ``n_pin``); the two carry identical voltage.
    Only the NAMEd one (``n_input`` / ``n_output`` on Level-10) appears in
    the dataset's node_voltages -- this returns it.
    """

    element = next((e for e in spec.elements if e.id == probe_id), None)
    if element is None:
        raise KeyError(f"boundary probe {probe_id!r} not found in spec.")
    candidate_nets = [pin.node for pin in element.pins]
    for net in candidate_nets:
        if net in dataset.node_voltages:
            return net
    raise KeyError(
        f"boundary probe {probe_id!r} touches nets {candidate_nets}, none of "
        f"which has an exported node voltage; available NAMEd nets: "
        f"{sorted(dataset.node_voltages)}."
    )


def _ads_end_to_end_gain_dB(
    spec: DeviceSpec,
    dataset: HBDataset,
    cell_map: CellMap,
    mix: tuple[int, ...],
    gain_convention: str,
) -> float:
    """End-to-end cascade gain from the NAMEd boundary nets + boundary probes."""

    n_cells = cell_count(cell_map)
    in_probe = cell_map.cells[0].boundary_in_probe_id
    out_probe = cell_map.cells[-1].boundary_out_probe_id
    if in_probe is None or out_probe is None:
        raise KeyError(
            "cascade is missing a design-boundary I_Probe; end-to-end gain "
            "needs boundary probes (Level-9+ probed cascades)."
        )
    v_in_net = _boundary_voltage_net(spec, dataset, in_probe)
    v_out_net = _boundary_voltage_net(spec, dataset, out_probe)
    v_in = complex(dataset.node_voltage(v_in_net).at_mix(mix))
    v_out = complex(dataset.node_voltage(v_out_net).at_mix(mix))

    if gain_convention == "voltage":
        if abs(v_in) == 0.0:
            return float("nan")
        return 20.0 * math.log10(abs(v_out) / abs(v_in))

    # power convention
    i_in = complex(cell_boundary_current(dataset, cell_map, 0, mix))
    i_out = complex(cell_boundary_current(dataset, cell_map, n_cells, mix))
    p_in = 0.5 * (v_in * i_in.conjugate()).real
    p_out = 0.5 * (v_out * i_out.conjugate()).real
    if p_in <= 0.0 or p_out <= 0.0:
        return float("nan")
    return 10.0 * math.log10(p_out / p_in)


def _per_cell_gain_dB(
    dataset: HBDataset,
    cell_map: CellMap,
    cell_index: int,
    mix: tuple[int, ...],
    gain_convention: str,
) -> float:
    """Per-cell gain via the existing power.py helpers (Level-9 NAMEd nets).

    On the paper Level-10 the cell through nets are not NAMEd, so the
    underlying lookups KeyError; we re-raise that as a clear, actionable
    message rather than letting a raw net-missing KeyError surface.
    """

    try:
        if gain_convention == "voltage":
            return float(
                cell_voltage_ratio_gain_dB(dataset, cell_map, cell_index, mix)
            )
        return float(cell_power_gain_dB(dataset, cell_map, cell_index, mix))
    except KeyError as exc:
        raise ValueError(
            f"per-cell gain at cell_index={cell_index} needs the cell's "
            f"through nets to be exported (NAMEd). The paper Level-10 "
            f"generator NAMEs only the design boundaries and per-cell "
            f"branch/varactor nets, so per-cell gain is unavailable there; "
            f"use cell_index=None (end-to-end). Original lookup error: {exc}"
        ) from exc


def _gain_for_point(
    spec: DeviceSpec,
    dataset: HBDataset,
    cell_map: CellMap,
    *,
    cell_index: int | None,
    mix: tuple[int, ...],
    gain_convention: str,
    source: str,
    truncation_N: int,
) -> float:
    """Return the gain in dB for one completed sweep point."""

    if source == "analytical":
        if cell_index is not None:
            raise ValueError(
                "analytical gain is an end-to-end cascade proxy; per-cell "
                "(cell_index != None) analytical gain is not defined. Use "
                "cell_index=None."
            )
        # Lazy import: keep analysis import-time free of the analytical
        # package (mirrors floquet_comparison's lazy analytical import).
        from ..analytical.parametric_gain import level10_signal_gain_proxy_dB

        try:
            return float(
                level10_signal_gain_proxy_dB(
                    spec, dataset, cell_map, truncation_N=int(truncation_N)
                )
            )
        except (ValueError, FloatingPointError):
            # Strong-pump small-signal-elastance limit or a non-finite
            # solve: the analytical linearisation does not apply here.
            return float("nan")

    # source == "ads"
    if cell_index is None:
        return _ads_end_to_end_gain_dB(
            spec, dataset, cell_map, mix, gain_convention
        )
    return _per_cell_gain_dB(dataset, cell_map, cell_index, mix, gain_convention)


def _gain_curve(
    sweep_result,
    *,
    cell_index: int | None,
    mix: tuple[int, ...],
    gain_convention: str,
    source: str,
    truncation_N: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Shared driver: (sweep value, gain dB) over completed points, sorted by x."""

    if gain_convention not in _GAIN_CONVENTIONS:
        raise ValueError(
            f"gain_convention must be one of {sorted(_GAIN_CONVENTIONS)} "
            f"(got {gain_convention!r})."
        )
    if source not in _SOURCES:
        raise ValueError(
            f"source must be one of {sorted(_SOURCES)} (got {source!r})."
        )

    spec = sweep_result.sweep_spec.canonical
    cell_map = build_cell_map(spec)

    xs: list[float] = []
    ys: list[float] = []
    for point in sweep_result.completed_points():
        xs.append(float(point.value))
        ys.append(
            _gain_for_point(
                spec, point.dataset, cell_map,
                cell_index=cell_index, mix=tuple(int(m) for m in mix),
                gain_convention=gain_convention, source=source,
                truncation_N=truncation_N,
            )
        )
    order = np.argsort(np.asarray(xs, dtype=float))
    x_arr = np.asarray(xs, dtype=float)[order]
    y_arr = np.asarray(ys, dtype=float)[order]
    return x_arr, y_arr


def gain_vs_signal_frequency(
    sweep_result,
    *,
    cell_index: int | None = None,
    mix: tuple[int, int] = (1, 0),
    gain_convention: str = "power",
    source: str = "ads",
    truncation_N: int = 6,
) -> tuple[np.ndarray, np.ndarray]:
    """Return ``(f_signal_Hz, gain_dB)`` for the Figure-6b reproduction.

    The X-axis is the swept signal frequency (each point's
    ``SweepPointResult.value``); the curve is sorted ascending in
    frequency. By default this is the cascade end-to-end gain from each
    point's live HB dataset at the signal mix ``(1, 0)``.

    Parameters
    ----------
    sweep_result : SweepResult
        The result of an ``f_signal`` sweep (see
        :class:`tvtl_ads_builder_v2.simulate.SweepRunner`). Only
        completed points contribute.
    cell_index : int, optional
        ``None`` (default) gives end-to-end cascade gain. A non-``None``
        index requests per-cell gain, available only on fully-NAMEd
        (Level-9-style) cascades; it raises on the paper Level-10.
    mix : tuple of int, optional
        Harmonic mix tuple. Default ``(1, 0)`` (the signal).
    gain_convention : {"power", "voltage"}, optional
        ``"power"`` -> ``10*log10(P_out/P_in)``; ``"voltage"`` ->
        ``20*log10(|V_out|/|V_in|)``. Default ``"power"``. (The
        analytical proxy ``20*log10|S21[0,0]|`` is convention-agnostic.)
    source : {"ads", "analytical"}, optional
        ``"ads"`` (default) reads the live-HB boundary gain;
        ``"analytical"`` returns the harmonic-ABCD-with-lift signal proxy
        (pump-informed). Analytical points beyond the small-signal
        elastance limit come back ``NaN``.
    truncation_N : int, optional
        Floquet truncation for the analytical source (ignored for
        ``"ads"``). Default 6.

    Returns
    -------
    (numpy.ndarray, numpy.ndarray)
        ``f_signal`` in Hz and gain in dB, ascending in frequency.
    """

    return _gain_curve(
        sweep_result, cell_index=cell_index, mix=mix,
        gain_convention=gain_convention, source=source,
        truncation_N=truncation_N,
    )


def gain_vs_pump_power(
    sweep_result,
    *,
    cell_index: int | None = None,
    mix: tuple[int, int] = (1, 0),
    gain_convention: str = "power",
    source: str = "ads",
    truncation_N: int = 6,
) -> tuple[np.ndarray, np.ndarray]:
    """Return ``(P_pump_dBm, gain_dB)`` for the Figure-6d reproduction.

    The X-axis is the swept pump power in dBm (each point's
    ``SweepPointResult.value``); the curve is sorted ascending in pump
    power. Parameters mirror :func:`gain_vs_signal_frequency`.

    Returns
    -------
    (numpy.ndarray, numpy.ndarray)
        ``P_pump`` in dBm and gain in dB, ascending in pump power.
    """

    return _gain_curve(
        sweep_result, cell_index=cell_index, mix=mix,
        gain_convention=gain_convention, source=source,
        truncation_N=truncation_N,
    )


