"""Structure an ADS Harmonic-Balance ``.ds`` dataset for per-cell analysis.

A raw ADS HB dataset (read by
:func:`tvtl_ads_builder_v2.simulate.parser.read_raw_dataset`) is a set
of varblocks whose columns are node voltages and branch currents, and
whose rows are the harmonic mixing products. This module turns that flat
table into an :class:`HBDataset`: every measured quantity becomes an
:class:`HBSpectrum` -- a set of :class:`HBHarmonic` components indexed by
the integer mixing tuple.

Harmonic indexing
-----------------
A two-tone HB analysis produces a mixing product for every integer pair
``(m, n)``: the component at frequency ``m*f_1 + n*f_2``. ADS records
these per row, with ``Mix[1]`` / ``Mix[2]`` columns carrying ``(m, n)``
and a ``freq`` column carrying the resulting frequency. This module
indexes harmonics by the ``(m, n)`` tuple -- exact even when two mixing
products land on the same frequency (the ``f_2 = 2*f_1`` degeneracy).
A single-tone analysis uses a 1-tuple ``(m,)``.

With the fundamentals ordered ``(f_signal, f_pump)`` the named products
are: signal ``(1, 0)``, pump ``(0, 1)``, idler ``(-1, 1)`` -- the idler
at ``f_pump - f_signal``.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Mapping, Sequence

from ..simulate.parser import RawDataset, RawVarBlock, read_raw_dataset

__all__ = [
    "HBHarmonic",
    "HBSpectrum",
    "HBDataset",
    "HBDatasetError",
    "hb_dataset_from_raw",
    "read_hb_dataset",
    "SIGNAL_MIX_2TONE",
    "PUMP_MIX_2TONE",
    "IDLER_MIX_2TONE",
]

# Named two-tone mixing products, with the fundamentals ordered
# (f_signal, f_pump). The idler is the (signal -1, pump +1) product at
# f_pump - f_signal. The gain analysis uses these.
SIGNAL_MIX_2TONE: tuple[int, int] = (1, 0)
PUMP_MIX_2TONE: tuple[int, int] = (0, 1)
IDLER_MIX_2TONE: tuple[int, int] = (-1, 1)

# Column-name patterns. A varblock column is a mixing-index axis when it
# matches ``Mix`` optionally followed by an index in [], () or bare; a
# frequency axis when it is freq / frequency (case-insensitive).
_MIX_COLUMN_RE = re.compile(r"Mix[\[\(]?(\d*)[\]\)]?\Z", re.IGNORECASE)
_FREQ_COLUMN_NAMES = frozenset({"freq", "frequency"})


class HBDatasetError(ValueError):
    """Raised when a raw dataset cannot be structured as an HB result."""


@dataclass(frozen=True)
class HBHarmonic:
    """One harmonic component of one measured quantity.

    ``mix`` is the integer mixing tuple ``(m, n, ...)`` -- one entry per
    fundamental tone. ``frequency_Hz`` is the resulting mixing-product
    frequency. ``amplitude`` is the complex phasor at that harmonic.
    """

    mix: tuple[int, ...]
    frequency_Hz: float
    amplitude: complex


@dataclass(frozen=True)
class HBSpectrum:
    """The full harmonic spectrum of one quantity (a node V or a current).

    ``quantity`` is the dataset column name (a net name for a node
    voltage, an element/probe id for a branch current). ``harmonics`` is
    the set of components, one per mixing tuple.
    """

    quantity: str
    harmonics: tuple[HBHarmonic, ...]

    def at_mix(self, mix: Sequence[int]) -> complex:
        """Return the complex amplitude at mixing tuple ``mix``.

        Raises
        ------
        KeyError
            If no harmonic with that mixing tuple is present -- the
            message lists the mixing tuples the spectrum does carry.
        """

        target = tuple(int(value) for value in mix)
        for harmonic in self.harmonics:
            if harmonic.mix == target:
                return harmonic.amplitude
        available = sorted(h.mix for h in self.harmonics)
        raise KeyError(
            f"quantity {self.quantity!r} has no harmonic at mix {target}; "
            f"available mixes: {available}"
        )

    def at_frequency(
        self, frequency_Hz: float, *, sum_collisions: bool = True
    ) -> complex:
        """Return the complex amplitude at ``frequency_Hz``.

        ADS HB rows are indexed by mixing tuple, and two tuples can land
        on one physical frequency (e.g. ``3*f1`` and ``f1 + f2`` when
        ``f2 = 2*f1``). With ``sum_collisions`` the colliding phasors are
        summed -- the physically correct total at that frequency; with it
        off the first match is returned. Returns ``0j`` if no harmonic is
        within tolerance.
        """

        tolerance = max(1.0e3, 1.0e-6 * abs(float(frequency_Hz)))
        matches = [
            h.amplitude
            for h in self.harmonics
            if abs(h.frequency_Hz - float(frequency_Hz)) <= tolerance
        ]
        if not matches:
            return 0j
        return sum(matches) if sum_collisions else matches[0]


@dataclass(frozen=True)
class HBDataset:
    """An ADS HB simulation result, structured for per-cell analysis.

    ``node_voltages`` maps a net name to its harmonic spectrum;
    ``branch_currents`` maps an element/probe id to its current spectrum.
    ``fundamental_freqs_Hz`` is the ordered tone list ``(f_signal,
    f_pump)``. ``sweep_point`` carries any outer-sweep coordinates (e.g.
    ``{"pump_power_dBm": 5.0}``); empty for a single-point analysis.
    """

    fundamental_freqs_Hz: tuple[float, ...]
    node_voltages: Mapping[str, HBSpectrum]
    branch_currents: Mapping[str, HBSpectrum]
    sweep_point: Mapping[str, float] = field(default_factory=dict)

    def node_voltage(self, net: str) -> HBSpectrum:
        """Return the spectrum of node ``net``, or raise :class:`KeyError`."""

        spectrum = self.node_voltages.get(net)
        if spectrum is None:
            raise KeyError(
                f"no node-voltage spectrum for net {net!r}; "
                f"available: {sorted(self.node_voltages)}"
            )
        return spectrum

    def branch_current(self, quantity: str) -> HBSpectrum:
        """Return the current spectrum of ``quantity``, or raise ``KeyError``."""

        spectrum = self.branch_currents.get(quantity)
        if spectrum is None:
            raise KeyError(
                f"no branch-current spectrum for {quantity!r}; "
                f"available: {sorted(self.branch_currents)}"
            )
        return spectrum

    def i_probe_current(self, probe_id: str) -> HBSpectrum:
        """Return the spectrum of an I_Probe by its element id.

        The native ADS I_Probe element exposes its current in the HB
        ``.ds`` as ``<probe_id>.i`` (live-probe finding,
        confirmed across both Level-9 canonicals). This accessor takes
        the bare element id and resolves the ``.i`` suffix internally so
        callers do not need to know the spelling -- mirrors the
        :meth:`node_voltage` accessor's net-name interface.

        Raises
        ------
        KeyError
            If no ``<probe_id>.i`` column is present in the dataset. The
            message lists the available I_Probe ids stripped of the
            ``.i`` suffix so the caller can see which probes the dataset
            actually carries.
        """

        column = f"{probe_id}.i"
        spectrum = self.branch_currents.get(column)
        if spectrum is None:
            available = sorted(
                name[:-2] for name in self.branch_currents if name.endswith(".i")
            )
            raise KeyError(
                f"no I_Probe current spectrum for probe id {probe_id!r} "
                f"(looked up column {column!r}); available I_Probe ids: "
                f"{available}"
            )
        return spectrum


def _looks_like_current(column_name: str) -> bool:
    """Heuristic: is a varblock column a branch current rather than a voltage?

    ADS records node voltages under the net name and branch currents
    under a current-flavoured name. The canonical current-column form
    in an ADS HB ``.ds`` is ``<I_Probe element id>.i`` -- confirmed
    against a live HB run (an I_Probe named
    ``IPR_IN`` produces the column ``IPR_IN.i`` in the ``HB1.HB``
    varblock). The other branches of this heuristic (``I_``-prefixed,
    ``I[`` / ``I(`` / ``I:`` spellings, ``current`` in the name) are
    retained for tolerance to alternative ADS naming conventions and
    for RSENSE-style proxy spellings that pre-date the native I_Probe
    element.
    """

    name = column_name.strip()
    lower = name.lower()
    return (
        name.startswith("I_")
        or name.startswith("I[")
        or name.startswith("I(")
        or name.startswith("I:")
        or lower.startswith("i_")
        or lower.endswith(".i")
        or "current" in lower
    )


def _find_hb_block(raw: RawDataset) -> tuple[RawVarBlock, tuple[str, ...], str | None]:
    """Locate the HB varblock and its mixing-index / frequency columns.

    Returns ``(block, mix_columns, freq_column)``. The HB block is the
    first varblock carrying at least one ``Mix`` column; ``mix_columns``
    is ordered by mixing index, ``freq_column`` is the frequency axis
    (``None`` when the block has none -- then frequency is derived from
    the mixing tuple and the fundamentals).
    """

    for block in raw.blocks:
        mix_pairs: list[tuple[int, str]] = []
        freq_column: str | None = None
        for name in block.columns:
            mix_match = _MIX_COLUMN_RE.fullmatch(name.strip())
            if mix_match is not None:
                index = int(mix_match.group(1)) if mix_match.group(1) else 1
                mix_pairs.append((index, name))
            elif name.strip().lower() in _FREQ_COLUMN_NAMES:
                freq_column = name
        if mix_pairs:
            mix_columns = tuple(name for _index, name in sorted(mix_pairs))
            return block, mix_columns, freq_column
    raise HBDatasetError(
        "no Harmonic-Balance varblock found -- none of the dataset's "
        f"varblocks {raw.block_names()} carries a 'Mix' mixing-index column."
    )


def hb_dataset_from_raw(
    raw: RawDataset, fundamental_freqs_Hz: Sequence[float]
) -> HBDataset:
    """Structure a :class:`RawDataset` as an :class:`HBDataset`.

    Parameters
    ----------
    raw : RawDataset
        The analysis-agnostic dataset (every varblock, every column).
    fundamental_freqs_Hz : sequence of float
        The HB fundamentals in order -- ``(f_signal, f_pump)`` for a
        two-tone analysis. Supplied by the caller because ADS records
        mixing tuples and frequencies but not which tone is the signal.

    Raises
    ------
    HBDatasetError
        If no HB varblock is present, or the mixing-index columns are
        malformed.
    """

    fundamentals = tuple(float(value) for value in fundamental_freqs_Hz)
    if not fundamentals:
        raise HBDatasetError("fundamental_freqs_Hz must list at least one tone.")
    block, mix_columns, freq_column = _find_hb_block(raw)
    n_rows = len(next(iter(block.columns.values()))) if block.columns else 0

    mixes: list[tuple[int, ...]] = []
    for row in range(n_rows):
        mixes.append(
            tuple(int(round(block.columns[col][row].real)) for col in mix_columns)
        )
    if freq_column is not None:
        freqs = [block.columns[freq_column][row].real for row in range(n_rows)]
    else:
        freqs = [
            sum(m * f for m, f in zip(mix, fundamentals)) for mix in mixes
        ]

    index_columns = set(mix_columns) | (
        {freq_column} if freq_column is not None else set()
    )
    node_voltages: dict[str, HBSpectrum] = {}
    branch_currents: dict[str, HBSpectrum] = {}
    for name, values in block.columns.items():
        if name in index_columns:
            continue
        harmonics = tuple(
            HBHarmonic(mix=mixes[row], frequency_Hz=freqs[row], amplitude=values[row])
            for row in range(n_rows)
        )
        spectrum = HBSpectrum(quantity=name, harmonics=harmonics)
        if _looks_like_current(name):
            branch_currents[name] = spectrum
        else:
            node_voltages[name] = spectrum

    return HBDataset(
        fundamental_freqs_Hz=fundamentals,
        node_voltages=node_voltages,
        branch_currents=branch_currents,
        sweep_point={},
    )


def read_hb_dataset(
    dataset_path: str | Path, fundamental_freqs_Hz: Sequence[float]
) -> HBDataset:
    """Read a ``.ds`` file and structure it as an :class:`HBDataset`.

    The file-path entry point: it reads the raw dataset through the
    in-process / ADS-Python-subprocess bridge and structures it. ADS-free
    callers (and tests) use :func:`hb_dataset_from_raw` on an in-memory
    :class:`RawDataset` directly.
    """

    return hb_dataset_from_raw(
        read_raw_dataset(dataset_path), fundamental_freqs_Hz
    )


