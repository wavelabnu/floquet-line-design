"""A 1-D operating-point sweep over a canonical's VAR-block scalars.

The Level-10 paper canonical carries a five-scalar VAR block
(``f_signal``, ``f_pump``, ``signal_power_dBm``, ``pump_power_dBm``,
``pump_phase_rad``); its ``P1ToneSource`` / ``HBTwoToneSpec`` reference
those scalars *by name*. Re-pointing the sweep at a new operating point
is therefore a VAR-block swap, not a topology rebuild:
:func:`with_var_overrides` returns the same spec with replaced
``VarEntry`` values, and the var-resolver substitutes the new literals at
emit time.

ADS's own VAR / VarEqn is non-functional on this install
(``docs/LIMITATIONS.md``), so an embedded HB-controller ParamSweep is not
available -- the sweep is realised as one resolved spec (one ADS run) per
point by :class:`tvtl_ads_builder_v2.simulate.sweep_runner.SweepRunner`.

This module is the *specification* half: it describes what is being swept
(:class:`SweepSpec`) and resolves a sweep value into the concrete VAR
overrides for that point. It performs no I/O and needs no ADS runtime, so
it is unit-testable without a simulator.

Scope: 1-D sweeps (Figure 6b sweeps ``f_signal`` at fixed pump power;
Figure 6d sweeps ``pump_power_dBm`` at fixed ``f_signal``). The
``point_overrides`` callable lets one swept axis drive several VAR
scalars at once -- Figure 6b sets ``f_pump = 2 * f_signal`` per point.
An N-D grid is a planned extension; the value axis would generalise to a
sequence of value tuples, but it is not implemented here.
"""

from __future__ import annotations

import dataclasses
import math
from dataclasses import dataclass, field
from typing import Callable, Mapping

from .schema import DeviceSpec, VarBlockSpec, VarEntry
from .validator import DeviceSpecError, validate_var_block

__all__ = [
    "SweepSpec",
    "with_var_overrides",
]


def with_var_overrides(
    spec: DeviceSpec, overrides: Mapping[str, float]
) -> DeviceSpec:
    """Return ``spec`` with its VAR-block values replaced per ``overrides``.

    Only the ``VarEntry`` values named in ``overrides`` change; every
    other scalar keeps its current value, and the block name, the
    variable order, and the rest of the spec are untouched. Because the
    Level-10 sources / HB controller reference the VAR scalars by name,
    the returned spec resolves to the new operating point at emit time
    with no topology rebuild.

    Parameters
    ----------
    spec : DeviceSpec
        A spec carrying a VAR block (e.g. a Level-10 cascade). A spec
        with no VAR block is an error -- there is nothing to override.
    overrides : Mapping[str, float]
        ``{variable name: new value}``. Every key must name a variable
        the block already declares; an unknown name is rejected (a typo
        would otherwise silently no-op).

    Returns
    -------
    DeviceSpec
        A new frozen :class:`DeviceSpec` with the swapped VAR block.

    Raises
    ------
    DeviceSpecError
        If ``spec`` has no VAR block, or ``overrides`` names a variable
        the block does not declare, or a value is not finite.
    """

    if spec.var_block is None:
        raise DeviceSpecError(
            "with_var_overrides requires a spec with a VAR block; this spec "
            "carries none (nothing to override)."
        )
    declared = {entry.name for entry in spec.var_block.variables}
    unknown = set(overrides) - declared
    if unknown:
        raise DeviceSpecError(
            f"with_var_overrides: variable(s) {sorted(unknown)} are not "
            f"declared in the VAR block (declared: {sorted(declared)})."
        )
    for name, value in overrides.items():
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise DeviceSpecError(
                f"with_var_overrides: value for {name!r} must be a real "
                f"number (got {type(value).__name__})."
            )
        if not math.isfinite(float(value)):
            raise DeviceSpecError(
                f"with_var_overrides: value for {name!r} must be finite "
                f"(got {value})."
            )
    new_vars = tuple(
        VarEntry(
            name=entry.name,
            value=float(overrides.get(entry.name, entry.value)),
        )
        for entry in spec.var_block.variables
    )
    new_block = VarBlockSpec(name=spec.var_block.name, variables=new_vars)
    return dataclasses.replace(spec, var_block=new_block)


def _default_point_overrides(variable: str) -> Callable[[float], dict[str, float]]:
    """Return the trivial ``value -> {variable: value}`` override map."""

    def _fn(value: float) -> dict[str, float]:
        return {variable: float(value)}

    return _fn


@dataclass(frozen=True)
class SweepSpec:
    """A 1-D operating-point sweep over a canonical's VAR scalars.

    Parameters
    ----------
    canonical : DeviceSpec
        The spec to drive (must carry a VAR block). It is never mutated;
        each sweep point produces a derived spec via
        :func:`with_var_overrides`.
    sweep_variable : str
        The canonical VAR-block name being swept -- the X-axis label of
        the resulting curve (e.g. ``"f_signal"`` for Figure 6b,
        ``"pump_power_dBm"`` for Figure 6d). Must be a declared variable.
    sweep_values : tuple of float
        The 1-D axis values. Non-empty; every entry finite.
    point_overrides : callable, optional
        ``value -> {variable name: value}`` giving the full set of VAR
        overrides for one point. This is where coupled scalars are
        expressed: Figure 6b passes
        ``lambda fs: {"f_signal": fs, "f_pump": 2 * fs}`` so the pump
        tracks the signal at the 2:1 parametric ratio. Default
        (``None``) is the identity ``value -> {sweep_variable: value}``.
        Every key the callable returns must name a declared VAR scalar.
    fixed_overrides : Mapping[str, float], optional
        Constant VAR overrides applied at *every* point (e.g. pinning
        ``pump_power_dBm`` while sweeping ``f_signal``). Keys must be
        declared VAR scalars. ``point_overrides`` wins on key collision.
    label : str, optional
        Free-text tag carried into the runner's per-point workspace
        directory names and the result. Defaults to ``sweep_variable``.

    Notes
    -----
    1-D only for now. An N-D grid (heatmap) is a planned extension:
    ``sweep_values`` would carry value tuples and ``sweep_variable`` a
    tuple of names; the rest of the API is unchanged.
    """

    canonical: DeviceSpec
    sweep_variable: str
    sweep_values: tuple[float, ...]
    point_overrides: Callable[[float], dict[str, float]] | None = None
    fixed_overrides: Mapping[str, float] = field(default_factory=dict)
    label: str = ""

    def __post_init__(self) -> None:
        if self.canonical.var_block is None:
            raise DeviceSpecError(
                "SweepSpec.canonical must carry a VAR block to sweep; this "
                "spec has none."
            )
        # The block itself must be well-formed (unique identifier names,
        # finite values) -- reuse the device-spec validator so a sweep
        # cannot be built on a malformed block.
        validate_var_block(self.canonical.var_block)
        declared = {entry.name for entry in self.canonical.var_block.variables}

        if self.sweep_variable not in declared:
            raise DeviceSpecError(
                f"SweepSpec.sweep_variable {self.sweep_variable!r} is not a "
                f"declared VAR scalar (declared: {sorted(declared)})."
            )

        values = tuple(float(v) for v in self.sweep_values)
        if not values:
            raise DeviceSpecError("SweepSpec.sweep_values must be non-empty.")
        if any(not math.isfinite(v) for v in values):
            raise DeviceSpecError(
                "SweepSpec.sweep_values entries must all be finite."
            )

        fixed = {str(k): float(v) for k, v in dict(self.fixed_overrides).items()}
        unknown_fixed = set(fixed) - declared
        if unknown_fixed:
            raise DeviceSpecError(
                f"SweepSpec.fixed_overrides names undeclared VAR scalar(s) "
                f"{sorted(unknown_fixed)} (declared: {sorted(declared)})."
            )

        # Normalise the frozen fields in place.
        object.__setattr__(self, "sweep_values", values)
        object.__setattr__(self, "fixed_overrides", fixed)
        if not self.label:
            object.__setattr__(self, "label", str(self.sweep_variable))

        # Dry-run the override callable on the first value to surface a
        # bad key (or a non-numeric return) before any ADS run.
        sample = self.resolved_overrides(values[0])
        del sample  # validation side effect only

    # -- resolution ------------------------------------------------------

    @property
    def var_names(self) -> frozenset[str]:
        """The set of VAR scalar names the canonical declares."""

        assert self.canonical.var_block is not None  # checked in __post_init__
        return frozenset(
            entry.name for entry in self.canonical.var_block.variables
        )

    def resolved_overrides(self, value: float) -> dict[str, float]:
        """Return the merged VAR overrides for one sweep ``value``.

        ``fixed_overrides`` first, then the (point or default) override
        callable -- so a per-point override wins over a fixed one on the
        same key. Every resulting key is validated against the declared
        VAR scalars; a non-numeric or non-finite value is rejected.
        """

        override_fn = self.point_overrides or _default_point_overrides(
            self.sweep_variable
        )
        produced = override_fn(float(value))
        if not isinstance(produced, Mapping):
            raise DeviceSpecError(
                f"SweepSpec.point_overrides must return a mapping; got "
                f"{type(produced).__name__} for value {value!r}."
            )
        merged: dict[str, float] = dict(self.fixed_overrides)
        merged.update({str(k): v for k, v in produced.items()})

        declared = self.var_names
        unknown = set(merged) - declared
        if unknown:
            raise DeviceSpecError(
                f"SweepSpec override for value {value!r} names undeclared VAR "
                f"scalar(s) {sorted(unknown)} (declared: {sorted(declared)})."
            )
        out: dict[str, float] = {}
        for name, raw in merged.items():
            if isinstance(raw, bool) or not isinstance(raw, (int, float)):
                raise DeviceSpecError(
                    f"SweepSpec override for value {value!r}: {name!r} must be "
                    f"a real number (got {type(raw).__name__})."
                )
            number = float(raw)
            if not math.isfinite(number):
                raise DeviceSpecError(
                    f"SweepSpec override for value {value!r}: {name!r} must be "
                    f"finite (got {raw})."
                )
            out[name] = number
        return out

    def spec_for(self, value: float) -> DeviceSpec:
        """Return the canonical spec re-pointed to sweep ``value``."""

        return with_var_overrides(self.canonical, self.resolved_overrides(value))


