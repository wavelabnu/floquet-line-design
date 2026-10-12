"""Per-cell power and gain computation for an N-cell TVTL cascade.

Consumes the HB dataset reader and per-cell
query API (:mod:`.hb_dataset`, :mod:`.per_cell`)
and exposes two complementary gain conventions:

* **Power gain** -- ``G_dB = 10 * log10(P_out / P_in)``, with per-mix
  real power computed as ``P = (1/2) * Re(V * conj(I))`` from the
  peak-phasor convention ADS uses for HB output. This is the
  physically correct gain definition under active power flow (the
  parametric-amplification case where the impedance seen by the signal
  is *not* uniform across the cell). It matches the paper's SI nodal-
  power formulation.
* **Voltage-ratio gain** -- ``G_dB = 20 * log10(|V_out| / |V_in|)``.
  Voltage-magnitude ratio in dB; equals power gain in dB only when the
  characteristic impedance between the two reference points is
  uniform, which holds for a passive Bloch eigenmode in a periodic
  cascade but breaks under parametric power injection. Provided for
  backward compatibility with the v1 result parser's gain extraction.

The two conventions agree numerically in the passive uniform-impedance
case and diverge in the active case. The energy-conservation
cross-check quantifies the divergence empirically against the
analytical Floquet result; this module just exposes the two observables.

Peak-phasor convention
----------------------
ADS HB writes complex amplitudes as peak phasors -- i.e. the
time-domain waveform is ``v(t) = Re(V_phasor * exp(j*omega*t))`` with
``|V_phasor|`` the peak value, not the RMS value. The time-averaged
real power at one harmonic of a single-port quantity is then::

    P_avg = (1/2) * Re(V_phasor * conj(I_phasor))

The ``1/2`` is the peak-to-RMS conversion built into the formula. If
ADS used RMS phasors the formula would be ``Re(V * conj(I))`` without
the ``1/2`` factor; it does not. See the dataset reader's docstring
(``hb_dataset.py``) for the cross-reference to the live probe that
established the column-naming convention these functions consume.

Topology-driven nodes and probes
--------------------------------
All four per-cell functions take an :class:`HBDataset` and a
:class:`CellMap` -- no DeviceSpec parameter, no naming-template
assumption. The cell-map walk resolved which I_Probe id
belongs to each boundary and each varactor by net topology; this
module looks them up through the existing :func:`cell_node_voltage`,
:func:`cell_boundary_current` and :func:`cell_varactor_current`
helpers. A spec built from a different topology rule will analyse
correctly as long as :func:`build_cell_map` accepts it.

Node convention for power
-------------------------
* **Boundary k** (``0 <= k <= N``): V is taken on the cell-facing side
  of the through-line I_Probe -- ``cells[k].input_net`` for an interior
  or input boundary, ``cells[N-1].output_net`` for the output
  boundary. The I_Probe is electrically a 0-ohm short, so the V on
  either side of it is identical; the cell-facing choice keeps the
  convention consistent. I is in the probe's pin-1 -> pin-2 direction
  (upstream -> downstream); positive ``P`` therefore means "power
  flowing downstream at this boundary."
* **Varactor in cell k**: V is taken at ``cells[k].varactor_signal_net``
  -- the SDD2P pin-1 net, which is the branch-top (MTEE pin 3) node in
  the Level-9 probed cascade. I is the in-series varactor probe
  current in its native pin-1 -> pin-2 direction (which by KCL is the
  same as the current flowing into the varactor at the branch-top
  terminal). Positive ``P`` therefore means "power flowing from the
  through-line into the shunt branch at this cell." This deviates
  slightly from an alternative "V at n_c{k}_v" convention (the floating
  node between SDD2P and the varactor probe): that node is not exposed
  by the cell map. Both choices measure the same in-series current
  through a passive observable; the branch-top choice gives "branch
  input power" (power entering the varactor + stub series chain from
  the through-line), which is the energy-balance-relevant quantity for
  the conservation check. The docstring of
  :func:`varactor_power` cross-references this decision.
"""

from __future__ import annotations

import math

from .hb_dataset import HBDataset
from .per_cell import (
    CellMap,
    cell_boundary_current,
    cell_count,
    cell_node_voltage,
    cell_varactor_current,
)

__all__ = [
    "cell_boundary_power",
    "varactor_power",
    "cell_voltage_ratio_gain_dB",
    "cell_power_gain_dB",
    "cascade_voltage_ratio_gain_dB",
    "cascade_power_gain_dB",
]


def _time_avg_real_power(voltage: complex, current: complex) -> float:
    """Return the peak-phasor time-averaged real power ``(1/2)*Re(V*conj(I))``.

    Centralised so every public function in this module is provably
    using the same formula -- a refactor here changes both boundary and
    varactor power consistently.
    """

    return 0.5 * (voltage * current.conjugate()).real


def cell_boundary_power(
    dataset: HBDataset,
    cell_map: CellMap,
    boundary_index: int,
    mix: tuple[int, ...],
) -> float:
    """Return the time-averaged real power at boundary ``boundary_index``.

    For an ``N``-cell cascade the boundary indices run from 0 to ``N``
    inclusive (mirroring :func:`cell_boundary_current`)::

        boundary 0  -- design input  (upstream of cell 0)
        boundary k  -- between cells k - 1 and k       (1 <= k < N)
        boundary N  -- design output (downstream of cell N - 1)

    Computed as ``P = (1/2) * Re(V * conj(I))`` with peak-phasor V and
    I. V is read from the cell-facing side of the boundary (the
    cell-input net for ``boundary_index < N``, the last cell's output
    net for ``boundary_index == N``). I is the through-line I_Probe
    current in the upstream -> downstream direction, so positive ``P``
    means "power flowing downstream into the next cell (or into the
    load at boundary N)."

    Parameters
    ----------
    dataset : HBDataset
        The HB result (peak phasors, indexed by mixing tuple).
    cell_map : CellMap
        The resolved cascade. Must carry boundary I_Probe ids on every
        boundary (Level 9+). Unprobed cascades raise
        :class:`.per_cell.CellResolutionError` via the underlying
        :func:`cell_boundary_current` call.
    boundary_index : int
        Boundary position, ``0 .. N`` inclusive.
    mix : tuple of int
        The harmonic mixing tuple ``(m, n, ...)``.

    Returns
    -------
    float
        Time-averaged real power in watts (peak-phasor convention).

    Raises
    ------
    IndexError
        If ``boundary_index`` is outside ``[0, N]``.
    KeyError
        If the dataset has no spectrum for the required net or I_Probe,
        or no harmonic at ``mix``.
    """

    n_cells = cell_count(cell_map)
    if not 0 <= boundary_index <= n_cells:
        raise IndexError(
            f"boundary_index {boundary_index} out of range for a "
            f"{n_cells}-cell cascade (valid: 0 .. {n_cells})."
        )
    if boundary_index < n_cells:
        voltage = cell_node_voltage(
            dataset, cell_map, boundary_index, "input", mix
        )
    else:
        voltage = cell_node_voltage(
            dataset, cell_map, n_cells - 1, "output", mix
        )
    current = cell_boundary_current(dataset, cell_map, boundary_index, mix)
    return _time_avg_real_power(voltage, current)


def varactor_power(
    dataset: HBDataset,
    cell_map: CellMap,
    cell_index: int,
    mix: tuple[int, ...],
) -> float:
    """Return the time-averaged real power into cell ``cell_index``'s branch.

    The shunt branch on a probed cascade is a single series chain
    (varactor + I_Probe + ... + stub-to-ground); the in-series probe
    current is the branch input current by KCL. This function pairs the
    branch-top voltage (cell-map's ``varactor_signal_net`` -- SDD2P
    pin 1, which on the Level-9 canonical is the MTEE pin 3 node) with
    the varactor probe current and returns
    ``P = (1/2) * Re(V * conj(I))``. The sign convention is "positive
    P = power flowing from the through-line into the shunt branch at
    this cell"; for a low-loss stub this approximates "power dissipated
    by the varactor (or absorbed from the pump in the case of a
    parametric mixer)."

    Implementation note
    -------------------
    An alternative convention measures V at ``n_c{k}_v`` (the SDD2P
    pin-2 net, between the varactor and the in-series I_Probe).
    That net is not exposed by the cell map.
    The branch-top measurement here is the cell-map-native
    equivalent: same in-series current, same peak-phasor formula, with
    V taken at the only varactor-adjacent node the cell map exposes
    (varactor_signal_net = SDD2P pin 1 = branch top on Level 9). The
    energy-conservation check is unaffected by
    the choice of V node along the in-series chain -- both define
    "power crossing into the branch from the through-line."

    Returns
    -------
    float
        Time-averaged real power in watts.

    Raises
    ------
    IndexError
        If ``cell_index`` is out of range.
    KeyError
        If the dataset is missing the required net voltage or probe
        current, or has no harmonic at ``mix``.
    .per_cell.CellResolutionError
        If the cell has no resolved varactor probe (unprobed cascade).
    """

    voltage = cell_node_voltage(
        dataset, cell_map, cell_index, "varactor_signal", mix
    )
    current = cell_varactor_current(dataset, cell_map, cell_index, mix)
    return _time_avg_real_power(voltage, current)


def cell_voltage_ratio_gain_dB(
    dataset: HBDataset,
    cell_map: CellMap,
    cell_index: int,
    mix: tuple[int, ...],
) -> float:
    """Return the voltage-ratio gain across cell ``cell_index``, in dB.

    ``G_dB = 20 * log10(|V_out| / |V_in|)`` where ``V_in`` is the cell's
    input-net voltage and ``V_out`` is the cell's output-net voltage at
    the requested mix. This is the voltage-magnitude ratio in dB and is
    the v1 result parser's gain definition; it equals the power-gain in
    dB only when the characteristic impedance seen by the harmonic is
    the same on both sides of the cell -- a property of passive Bloch
    eigenmodes that does *not* hold under active power injection at
    parametric gain. Use :func:`cell_power_gain_dB` for the
    physically-correct definition under parametric amplification, and
    use this one when comparing against v1-style gain reports.

    Returns
    -------
    float
        Gain in dB. Returns ``-inf`` if ``|V_in| == 0`` (the harmonic
        does not exist on the input -- no finite-dB ratio can be
        defined).
    """

    v_in = cell_node_voltage(dataset, cell_map, cell_index, "input", mix)
    v_out = cell_node_voltage(dataset, cell_map, cell_index, "output", mix)
    mag_in = abs(v_in)
    if mag_in == 0.0:
        return float("-inf")
    return 20.0 * math.log10(abs(v_out) / mag_in)


def cell_power_gain_dB(
    dataset: HBDataset,
    cell_map: CellMap,
    cell_index: int,
    mix: tuple[int, ...],
) -> float:
    """Return the power gain across cell ``cell_index``, in dB.

    ``G_dB = 10 * log10(P_out / P_in)`` where ``P_in`` is the boundary
    power flowing into cell k (``cell_boundary_power(boundary_index=k)``)
    and ``P_out`` is the boundary power flowing out of cell k
    (``cell_boundary_power(boundary_index=k + 1)``). Uses the
    peak-phasor ``(1/2) Re(V * conj(I))`` convention via
    :func:`cell_boundary_power`. This is the physically-correct gain
    definition under active power flow (parametric amplification): a
    purely-real-power ratio across the cell, blind to the local
    characteristic impedance.

    Returns
    -------
    float
        Gain in dB. Returns ``nan`` if ``P_in <= 0`` or ``P_out <= 0``
        (negative real power at a port means power is flowing in the
        *opposite* direction to the assumed convention, so a real-valued
        dB ratio is not meaningful at that operating point).
    """

    p_in = cell_boundary_power(dataset, cell_map, cell_index, mix)
    p_out = cell_boundary_power(dataset, cell_map, cell_index + 1, mix)
    if p_in <= 0.0 or p_out <= 0.0:
        return float("nan")
    return 10.0 * math.log10(p_out / p_in)


def cascade_voltage_ratio_gain_dB(
    dataset: HBDataset,
    cell_map: CellMap,
    mix: tuple[int, ...],
) -> float:
    """End-to-end voltage-ratio gain across the whole cascade, in dB.

    ``G_dB = 20 * log10(|V_design_output| / |V_design_input|)`` at the
    requested mix. By the log-identity it equals the sum of
    :func:`cell_voltage_ratio_gain_dB` across cells **when each cell's
    output net is electrically the same as the next cell's input net**
    -- i.e. the inter-cell I_Probe is treated as the 0-ohm short it is.
    The Level-9 probed cascade satisfies this in the .ds dataset: ADS
    reports identical complex amplitudes on the two sides of each
    inter-cell probe (verified by integration test). A
    small numerical residual is normal; tests assert
    ``abs(sum_k - cascade) < 1e-6 dB``.

    Returns
    -------
    float
        End-to-end voltage-ratio gain in dB. ``-inf`` if the input
        magnitude is zero.
    """

    n_cells = cell_count(cell_map)
    if n_cells == 0:
        return float("nan")
    v_in = cell_node_voltage(dataset, cell_map, 0, "input", mix)
    v_out = cell_node_voltage(dataset, cell_map, n_cells - 1, "output", mix)
    mag_in = abs(v_in)
    if mag_in == 0.0:
        return float("-inf")
    return 20.0 * math.log10(abs(v_out) / mag_in)


def cascade_power_gain_dB(
    dataset: HBDataset,
    cell_map: CellMap,
    mix: tuple[int, ...],
) -> float:
    """End-to-end power gain across the whole cascade, in dB.

    ``G_dB = 10 * log10(P_design_output / P_design_input)`` at the
    requested mix, with ``P_design_input = cell_boundary_power(0)`` and
    ``P_design_output = cell_boundary_power(N)``. By the log-identity it
    equals ``sum_k cell_power_gain_dB(k)`` to within numerical noise --
    this is the self-consistency check the integration test performs.

    Returns
    -------
    float
        End-to-end power gain in dB. ``nan`` if either ``P_in`` or
        ``P_out`` is non-positive (no real-valued dB ratio at that
        operating point).
    """

    n_cells = cell_count(cell_map)
    if n_cells == 0:
        return float("nan")
    p_in = cell_boundary_power(dataset, cell_map, 0, mix)
    p_out = cell_boundary_power(dataset, cell_map, n_cells, mix)
    if p_in <= 0.0 or p_out <= 0.0:
        return float("nan")
    return 10.0 * math.log10(p_out / p_in)


