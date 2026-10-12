"""Drive a :class:`SweepSpec` through ADS HB, one run per sweep point.

ADS VAR / VarEqn is non-functional on this install (see
``docs/LIMITATIONS.md``), so an embedded HB-controller ParamSweep is not
available. The sweep is therefore realised as N independent runs: for
each sweep value the canonical spec is re-pointed to that operating point
(:func:`tvtl_ads_builder_v2.device_spec.with_var_overrides` -- a VAR-block
swap, no topology rebuild), an OA workspace is emitted, ``hpeesofsim``
runs the two-tone HB, and the resulting ``.ds`` is read back as an
:class:`HBDataset` tagged with that point's ``sweep_point`` coordinates.

This mirrors the proven Floquet / Figure-4 pipeline
(``emit_workspace_from_spec`` -> ``run_hpeesofsim`` -> ``read_hb_dataset``)
and inherits its status-dict discipline: a single failing point records an
error and does not abort the sweep. When the ADS Python bridge
(``keysight.ads.de``) or the simulator binary is unreachable the runner
returns every point with status ``"blocked_missing_ads_runtime"`` instead
of raising -- so an ADS-free host (and the unit tests) skips cleanly.
"""

from __future__ import annotations

import dataclasses
import os
import shutil
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Mapping

from ..device_spec import DeviceSpec, SweepSpec
from .runner import run_hpeesofsim

if TYPE_CHECKING:  # avoid an import cycle: analysis.hb_dataset -> simulate
    from ..analysis.hb_dataset import HBDataset

__all__ = [
    "SweepPointResult",
    "SweepResult",
    "SweepRunner",
    "ads_runtime_available",
]


# Sentinel statuses a point can carry. "completed" means the .ds was read
# into an HBDataset; the rest mirror run_hpeesofsim's vocabulary plus an
# "emit_error" for an OA-emit failure that happens before the simulator.
_STATUS_COMPLETED = "completed"
_STATUS_BLOCKED = "blocked_missing_ads_runtime"


def ads_runtime_available() -> bool:
    """Return True iff both the ADS Python bridge and simulator are reachable.

    The OA emit step needs ``keysight.ads.de`` and the run step needs an
    ``adssim``/``hpeesofsim`` binary. The check mirrors the skip-gate the
    parity integration tests use so the runner skips on exactly the hosts
    those tests skip on.
    """

    return _keysight_importable() and _ads_simulator_available()


def _ads_simulator_available() -> bool:
    if os.environ.get("HPEESOF_DIR"):
        return True
    for name in ("adssim_win.exe", "adssim", "hpeesofsim.exe", "hpeesofsim"):
        if shutil.which(name):
            return True
    return False


def _keysight_importable() -> bool:
    hpeesof = os.environ.get("HPEESOF_DIR")
    if hpeesof:
        packages = Path(hpeesof) / "tools" / "python" / "packages"
        if packages.is_dir() and str(packages) not in sys.path:
            sys.path.insert(0, str(packages))
    try:
        import keysight.ads.de  # noqa: F401
    except Exception:  # noqa: BLE001 -- any failure is "not importable"
        return False
    return True


def _fundamentals_from_spec(spec: DeviceSpec) -> tuple[float, float]:
    """Return the two HB fundamental frequencies (signal, pump) in Hz.

    The Level-10 HB controller references its fundamentals by VAR name;
    resolve those names against the spec's VAR block. A literal numeric
    field passes through unchanged. The order is (fundamental_1,
    fundamental_2) = (f_signal, f_pump) -- the convention the dataset
    reader expects.
    """

    hb = spec.hb_two_tone
    if hb is None:
        raise ValueError(
            "sweep runner expects a two-tone HB spec; this spec carries no "
            "hb_two_tone controller."
        )
    var_values: dict[str, float] = {}
    if spec.var_block is not None:
        var_values = {v.name: float(v.value) for v in spec.var_block.variables}

    def _resolve(field_value: float | str) -> float:
        if isinstance(field_value, str):
            if field_value not in var_values:
                raise ValueError(
                    f"HB fundamental references {field_value!r}, not declared "
                    f"in the VAR block."
                )
            return float(var_values[field_value])
        return float(field_value)

    return _resolve(hb.fundamental_freq_1), _resolve(hb.fundamental_freq_2)


@dataclass(frozen=True)
class SweepPointResult:
    """The outcome of one sweep point.

    ``status`` is ``"completed"`` when ``dataset`` is a readable
    :class:`HBDataset`; otherwise it carries the run/emit failure
    vocabulary (``"error"``, ``"timeout"``, ``"emit_error"``,
    ``"blocked_missing_ads_runtime"``) and ``dataset`` is ``None``.
    ``overrides`` is the per-point VAR override mapping (also stored on
    ``dataset.sweep_point``). ``emit_time_s`` / ``sim_time_s`` split the
    wall time so a dry run can report which step dominates.
    """

    value: float
    overrides: Mapping[str, float]
    status: str
    fundamentals_Hz: tuple[float, float] | None = None
    dataset: HBDataset | None = None
    dataset_path: str | None = None
    emit_time_s: float | None = None
    sim_time_s: float | None = None
    wall_time_s: float | None = None
    error_message: str | None = None

    @property
    def completed(self) -> bool:
        return self.status == _STATUS_COMPLETED and self.dataset is not None


@dataclass(frozen=True)
class SweepResult:
    """The aggregate of a whole sweep -- one :class:`SweepPointResult` per value."""

    sweep_spec: SweepSpec
    points: tuple[SweepPointResult, ...] = field(default_factory=tuple)

    def completed_points(self) -> tuple[SweepPointResult, ...]:
        """Return only the points whose HB run completed and parsed."""

        return tuple(p for p in self.points if p.completed)

    def values(self) -> tuple[float, ...]:
        """Return the sweep-axis values in run order (all points)."""

        return tuple(p.value for p in self.points)

    def datasets(self) -> tuple[HBDataset, ...]:
        """Return the HBDatasets of the completed points, in run order."""

        return tuple(p.dataset for p in self.completed_points())  # type: ignore[misc]

    @property
    def all_blocked(self) -> bool:
        """True iff every point is blocked for missing ADS runtime."""

        return bool(self.points) and all(
            p.status == _STATUS_BLOCKED for p in self.points
        )

    @property
    def n_completed(self) -> int:
        return len(self.completed_points())


class SweepRunner:
    """Run a :class:`SweepSpec` through ADS HB, one point at a time.

    Parameters
    ----------
    out_dir : str or Path
        Root directory for the per-point OA workspaces and ``.ds`` files.
        Created if absent. One subdirectory per sweep point.
    timeout_s : int
        Per-point ``hpeesofsim`` timeout. Defaults to 600 s (the
        Figure-4 value) -- a two-tone 15-cell HB is heavier than the
        passive S_Param leg.
    keep_workspaces : bool
        When ``False`` (default) each point's OA workspace directory is
        removed after its ``.ds`` is read, keeping disk bounded across a
        long sweep; the parsed :class:`HBDataset` is held in memory. When
        ``True`` the workspaces are left on disk for inspection.
    """

    def __init__(
        self,
        out_dir: str | Path,
        *,
        timeout_s: int = 600,
        keep_workspaces: bool = False,
    ) -> None:
        self.out_dir = Path(out_dir)
        self.timeout_s = int(timeout_s)
        self.keep_workspaces = bool(keep_workspaces)

    def run(self, sweep: SweepSpec) -> SweepResult:
        """Execute every point of ``sweep`` and return the aggregate result.

        Never raises on a single bad point: an emit failure, a non-zero
        ``hpeesofsim`` return, a timeout, or a dataset-read failure is
        recorded as that point's ``status`` / ``error_message`` and the
        sweep proceeds. If the ADS runtime is unreachable, every point is
        returned blocked (no emit attempted).
        """

        if not ads_runtime_available():
            blocked = tuple(
                SweepPointResult(
                    value=float(value),
                    overrides=sweep.resolved_overrides(value),
                    status=_STATUS_BLOCKED,
                    error_message=(
                        "ADS runtime unavailable (keysight.ads.de not "
                        "importable or hpeesofsim/adssim not found); set "
                        "HPEESOF_DIR to run the sweep."
                    ),
                )
                for value in sweep.sweep_values
            )
            return SweepResult(sweep_spec=sweep, points=blocked)

        self.out_dir.mkdir(parents=True, exist_ok=True)
        points: list[SweepPointResult] = []
        for index, value in enumerate(sweep.sweep_values):
            points.append(self._run_point(sweep, index, float(value)))
        return SweepResult(sweep_spec=sweep, points=tuple(points))

    # -- per-point ------------------------------------------------------

    def _run_point(
        self, sweep: SweepSpec, index: int, value: float
    ) -> SweepPointResult:
        overrides = sweep.resolved_overrides(value)
        point_spec = sweep.spec_for(value)
        try:
            fundamentals = _fundamentals_from_spec(point_spec)
        except ValueError as exc:
            return SweepPointResult(
                value=value, overrides=overrides, status="error",
                error_message=f"could not resolve HB fundamentals: {exc}",
            )

        ws_dir = self.out_dir / f"point_{index:03d}_{sweep.label}"

        # --- emit ------------------------------------------------------
        from ..oa.oa_emitter import emit_workspace_from_spec

        emit_started = time.time()
        try:
            info = emit_workspace_from_spec(point_spec, ws_dir / "workspace")
        except Exception as exc:  # noqa: BLE001 -- record, do not abort sweep
            return SweepPointResult(
                value=value, overrides=overrides, status="emit_error",
                fundamentals_Hz=fundamentals,
                emit_time_s=time.time() - emit_started,
                error_message=f"OA emit failed: {type(exc).__name__}: {exc}",
            )
        emit_time = time.time() - emit_started

        ads_netlist = Path(info["ads_netlist_path"])

        # --- simulate --------------------------------------------------
        sim_result = run_hpeesofsim(ads_netlist, timeout_s=self.timeout_s)
        sim_time = sim_result.get("wall_time_s")
        status = sim_result.get("status")
        if status != "completed" or not sim_result.get("dataset_path"):
            self._cleanup(ws_dir)
            return SweepPointResult(
                value=value, overrides=overrides,
                status=str(status) if status else "error",
                fundamentals_Hz=fundamentals,
                emit_time_s=emit_time, sim_time_s=sim_time,
                wall_time_s=emit_time + float(sim_time or 0.0),
                error_message=(
                    sim_result.get("error_message")
                    or f"hpeesofsim returned status {status!r}"
                ),
            )

        # --- read ------------------------------------------------------
        # Lazy import: analysis.hb_dataset imports simulate.parser, so a
        # top-level import here would form a cycle at package init.
        from ..analysis.hb_dataset import read_hb_dataset

        dataset_path = str(sim_result["dataset_path"])
        try:
            hb = read_hb_dataset(dataset_path, fundamentals)
        except Exception as exc:  # noqa: BLE001
            self._cleanup(ws_dir)
            return SweepPointResult(
                value=value, overrides=overrides, status="error",
                fundamentals_Hz=fundamentals,
                emit_time_s=emit_time, sim_time_s=sim_time,
                wall_time_s=emit_time + float(sim_time or 0.0),
                error_message=(
                    f"dataset read failed: {type(exc).__name__}: {exc}"
                ),
            )

        # Tag the dataset with the operating point it was run at.
        hb = dataclasses.replace(hb, sweep_point=dict(overrides))
        self._cleanup(ws_dir)
        return SweepPointResult(
            value=value, overrides=overrides, status=_STATUS_COMPLETED,
            fundamentals_Hz=fundamentals, dataset=hb,
            dataset_path=dataset_path,
            emit_time_s=emit_time, sim_time_s=sim_time,
            wall_time_s=emit_time + float(sim_time or 0.0),
        )

    def _cleanup(self, ws_dir: Path) -> None:
        if self.keep_workspaces:
            return
        try:
            if ws_dir.exists():
                shutil.rmtree(ws_dir, ignore_errors=True)
        except OSError:
            pass


