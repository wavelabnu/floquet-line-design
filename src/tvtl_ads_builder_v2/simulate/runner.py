"""Invoke ``hpeesofsim`` on a flat-netlist ``.ckt`` and capture artifacts.

This is the primary ``hpeesofsim`` entry point in the v2 builder (the
``validation`` amplifier legs also invoke the ADS ``CircuitSimulator``
directly). The runner deliberately keeps no retry logic, no fallback
paths, and no notion of "the last successful environment". Detection
either succeeds and the simulator runs, or detection fails and the
caller sees ``status="blocked_missing_ads_runtime"``.

When the simulator returns non-zero, the runner surfaces stdout/stderr
tails so a netlist syntax error is visible in the parity report -- no
silent degenerate "no ADS" status.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import time
from pathlib import Path
from typing import Any, Mapping


SIMULATOR_BINARIES = ("adssim_win.exe", "adssim", "hpeesofsim.exe", "hpeesofsim")
"""Preferred-order list of simulator executables.

``adssim_win.exe`` is the documented entry point on Windows: it wraps
``hpeesofsim.exe`` with the runtime environment the kernel needs and
must be invoked instead of calling ``hpeesofsim.exe`` directly (the
latter exits with ``STATUS_DLL_NOT_FOUND`` if certain ADS DLLs are
unreachable, regardless of how PATH is set). The bare ``hpeesofsim``
spellings are kept as a fallback for non-Windows installs.
"""


def run_hpeesofsim(
    netlist_path: str | Path,
    *,
    ads_environment: Mapping[str, Any] | None = None,
    timeout_s: int = 120,
) -> dict[str, Any]:
    """Run hpeesofsim on ``netlist_path``.

    Parameters
    ----------
    netlist_path : str or Path
        Path to a flat ``.ckt`` deck emitted by
        :func:`tvtl_ads_builder_v2.netlist.emit_hpeesofsim_netlist`. The
        simulator runs with this file's parent directory as its cwd so
        any ``.ds`` it writes lands next to it.
    ads_environment : Mapping or None
        Optional pre-resolved environment with ``hpeesof_dir`` and
        ``simulator_path`` keys. Useful in tests; in production the
        runner auto-detects from ``$HPEESOF_DIR``.
    timeout_s : int
        Hard timeout on the simulator subprocess.

    Returns
    -------
    dict
        Keys ``status`` (one of ``"completed"``, ``"error"``,
        ``"timeout"``, ``"blocked_missing_ads_runtime"``),
        ``dataset_path``, ``log_path``, ``returncode``,
        ``stdout_tail``, ``stderr_tail``, ``wall_time_s``.
    """

    netlist_path = Path(netlist_path).resolve()
    workspace = netlist_path.parent
    blocked = lambda message: {
        "status": "blocked_missing_ads_runtime",
        "dataset_path": None,
        "log_path": None,
        "returncode": None,
        "stdout_tail": None,
        "stderr_tail": None,
        "wall_time_s": None,
        "error_message": message,
    }
    if not netlist_path.exists():
        return {**blocked(f"netlist not found at {netlist_path}"), "status": "error"}

    simulator = _resolve_simulator(ads_environment)
    if simulator is None:
        return blocked(
            "hpeesofsim binary not found; set HPEESOF_DIR or pass ads_environment."
        )

    env = os.environ.copy()
    hpeesof_dir = None
    if ads_environment and "hpeesof_dir" in ads_environment:
        hpeesof_dir = Path(str(ads_environment["hpeesof_dir"]))
    elif env.get("HPEESOF_DIR"):
        hpeesof_dir = Path(env["HPEESOF_DIR"])
    if hpeesof_dir is not None:
        env["HPEESOF_DIR"] = str(hpeesof_dir)
        env.setdefault("ADS_HOME", str(hpeesof_dir))
        # hpeesofsim.exe loads DLLs from the ADS install's bin dirs at
        # process start; if any of these dirs is missing from PATH the
        # binary exits with a Windows ``STATUS_DLL_NOT_FOUND`` (0xC0000135)
        # and writes nothing to stdout/stderr. Prepend them explicitly.
        ads_path_entries = [
            hpeesof_dir / "bin",
            hpeesof_dir / "bin64",
            hpeesof_dir / "tools" / "bin",
            hpeesof_dir / "tools" / "win32_64" / "bin",
        ]
        ads_path = os.pathsep.join(
            str(entry) for entry in ads_path_entries if entry.exists()
        )
        if ads_path:
            env["PATH"] = ads_path + os.pathsep + env.get("PATH", "")

    stdout_path = workspace / (netlist_path.stem + "_stdout.txt")
    stderr_path = workspace / (netlist_path.stem + "_stderr.txt")
    started = time.time()
    try:
        completed = subprocess.run(
            [str(simulator), str(netlist_path.name)],
            cwd=str(workspace),
            env=env,
            text=True,
            capture_output=True,
            check=False,
            timeout=int(timeout_s),
        )
    except subprocess.TimeoutExpired as exc:
        wall_time = time.time() - started
        stdout_path.write_text((exc.stdout or "") if isinstance(exc.stdout, str) else "", encoding="utf-8")
        stderr_path.write_text((exc.stderr or "") if isinstance(exc.stderr, str) else "", encoding="utf-8")
        return {
            "status": "timeout",
            "dataset_path": None,
            "log_path": str(stdout_path),
            "returncode": None,
            "stdout_tail": None,
            "stderr_tail": None,
            "wall_time_s": wall_time,
            "error_message": f"hpeesofsim timed out after {timeout_s} s",
        }
    wall_time = time.time() - started
    stdout_path.write_text(completed.stdout or "", encoding="utf-8")
    stderr_path.write_text(completed.stderr or "", encoding="utf-8")
    if completed.returncode != 0:
        return {
            "status": "error",
            "dataset_path": None,
            "log_path": str(stdout_path),
            "returncode": int(completed.returncode),
            "stdout_tail": _tail(completed.stdout),
            "stderr_tail": _tail(completed.stderr),
            "wall_time_s": wall_time,
            "error_message": "hpeesofsim returned non-zero",
        }
    dataset_path = _own_dataset(workspace, netlist_path, started)
    return {
        "status": "completed" if dataset_path else "error",
        "dataset_path": dataset_path,
        "log_path": str(stdout_path),
        "returncode": 0,
        "stdout_tail": _tail(completed.stdout),
        "stderr_tail": _tail(completed.stderr),
        "wall_time_s": wall_time,
        "error_message": None if dataset_path else "hpeesofsim wrote no .ds dataset",
    }


def _resolve_simulator(ads_environment: Mapping[str, Any] | None) -> Path | None:
    """Locate the ADS simulator executable.

    The preferred binary is ``adssim_win.exe`` on Windows -- it is the
    wrapper that sets up the runtime environment for the underlying
    hpeesofsim kernel. Calling ``hpeesofsim.exe`` directly without that
    wrapper produces ``STATUS_DLL_NOT_FOUND``.
    """

    if ads_environment and ads_environment.get("simulator_path"):
        candidate = Path(str(ads_environment["simulator_path"]))
        if candidate.exists():
            return candidate
    if ads_environment and ads_environment.get("hpeesof_dir"):
        for name in SIMULATOR_BINARIES:
            candidate = Path(str(ads_environment["hpeesof_dir"])) / "bin" / name
            if candidate.exists():
                return candidate
    hpeesof_dir = os.environ.get("HPEESOF_DIR")
    if hpeesof_dir:
        for name in SIMULATOR_BINARIES:
            candidate = Path(hpeesof_dir) / "bin" / name
            if candidate.exists():
                return candidate
    for name in SIMULATOR_BINARIES:
        on_path = shutil.which(name)
        if on_path:
            return Path(on_path)
    return None


def _tail(text: str | None, lines: int = 40) -> str | None:
    if not text:
        return None
    parts = text.splitlines()
    return "\n".join(parts[-lines:])




def _own_dataset(workspace: Path, deck: Path, started: float) -> str | None:
    """Prefer a fresh exact deck-name dataset; reject ambiguous fresh outputs."""
    candidates=[p for p in workspace.glob('*.ds') if p.stat().st_mtime>=started]
    named=[p for p in candidates if p.name in (deck.name+'.ds',deck.stem+'.ds')]
    selected=named or candidates
    if len(selected)>1:raise RuntimeError('Ambiguous datasets produced by '+deck.name+': '+', '.join(p.name for p in selected))
    return str(selected[0]) if selected else None
