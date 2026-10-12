"""Read an hpeesofsim ``.ds`` dataset into a frequency-vs-S-parameter dict.

Two paths are supported:

1. **In-process**: when ``keysight.ads.dataset`` is importable from the
   current interpreter, the dataset is read directly.
2. **Subprocess fallback**: when the dataset module is not importable,
   the parser shells out to a tiny worker script via the ADS-bundled
   Python. The worker prints a JSON payload on stdout that the
   in-process side consumes.

Either path returns the same dict shape.
"""

from __future__ import annotations

import json
import math
import os
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import numpy as np


PARSER_WORKER_FILENAME = "ads_v2_parser_worker.py"

# Worker filename for the analysis-agnostic raw-dataset read.
RAW_PARSER_WORKER_FILENAME = "ads_v2_raw_parser_worker.py"


def parse_s_parameter_results(dataset_path: str | Path) -> dict[str, Any]:
    """Parse an ``.ds`` dataset into a structured dict.

    Parameters
    ----------
    dataset_path : str or Path
        Path to the dataset produced by hpeesofsim.

    Returns
    -------
    dict
        Keys ``frequencies_Hz`` (ndarray of float), ``S11``, ``S21``,
        ``S12``, ``S22`` (ndarrays of complex), and ``analysis_label``
        (``"ads_hpeesofsim"`` on success, ``"blocked_missing_ads_api"``
        or ``"error"`` otherwise).
    """

    path = Path(dataset_path).resolve()
    if not path.exists():
        return _empty(status="error", message=f"dataset not found at {path}")

    payload = _read_in_process(path)
    if payload is not None:
        return payload
    payload = _read_via_subprocess(path)
    if payload is not None:
        return payload
    return _empty(
        status="blocked_missing_ads_api",
        message="keysight.ads.dataset not importable from either Python.",
    )


def _read_in_process(path: Path) -> dict[str, Any] | None:
    try:
        from keysight.ads import dataset as ds_mod  # type: ignore
    except Exception:
        try:
            import keysight.ads.dataset as ds_mod  # type: ignore
        except Exception:
            return None
    try:
        return _extract_from_dataset_module(ds_mod, path)
    except Exception as exc:
        return _empty(status="error", message=f"in-process parse failed: {exc!r}")


def _read_via_subprocess(path: Path) -> dict[str, Any] | None:
    ads_python = _find_ads_python()
    if ads_python is None:
        return None
    worker_path = path.parent / PARSER_WORKER_FILENAME
    worker_path.write_text(_WORKER_SCRIPT, encoding="utf-8")
    env = os.environ.copy()
    hpeesof_dir = env.get("HPEESOF_DIR")
    if not hpeesof_dir:
        guess = ads_python.parents[2]
        env["HPEESOF_DIR"] = str(guess)
    try:
        completed = subprocess.run(
            [str(ads_python), str(worker_path), str(path)],
            text=True,
            capture_output=True,
            check=False,
            timeout=120,
            env=env,
        )
    except subprocess.TimeoutExpired:
        return _empty(status="error", message="parser subprocess timed out.")
    stdout = completed.stdout or ""
    try:
        payload = json.loads(stdout) if stdout.strip() else None
    except json.JSONDecodeError:
        first_brace = stdout.find("{")
        payload = json.loads(stdout[first_brace:]) if first_brace >= 0 else None
    if not payload or not isinstance(payload, dict):
        return _empty(
            status="error",
            message=(
                "parser subprocess produced no JSON; "
                f"stderr={completed.stderr!r}"
            ),
        )
    if payload.get("status") != "ok":
        return _empty(status="error", message=payload.get("error", "parser worker error"))
    return {
        "frequencies_Hz": np.asarray(payload["frequencies_Hz"], dtype=float),
        "S11": _to_complex(payload["S11"]),
        "S21": _to_complex(payload["S21"]),
        "S12": _to_complex(payload["S12"]),
        "S22": _to_complex(payload["S22"]),
        "analysis_label": "ads_hpeesofsim",
    }


def _extract_from_dataset_module(ds_mod: Any, path: Path) -> dict[str, Any]:
    """Common dataset-read body used by both in-process and worker paths."""

    with ds_mod.open(str(path), "r") as ds:
        blocks = list(ds.varblocks.items())
    s11 = s21 = s12 = s22 = None
    frequencies = None
    for _name, block in blocks:
        try:
            frame = block.to_dataframe()
        except Exception:
            continue
        if getattr(frame, "index", None) is not None and frame.index.name == "freq":
            frame = frame.reset_index()
        columns = list(getattr(frame, "columns", []))
        if "freq" not in columns:
            continue
        freq_arr = np.asarray(
            [float(getattr(value, "real", value)) for value in frame["freq"]],
            dtype=float,
        )
        col_pairs = (
            ("S(1,1)", "S11"),
            ("S(2,1)", "S21"),
            ("S(1,2)", "S12"),
            ("S(2,2)", "S22"),
            ("S[1,1]", "S11"),
            ("S[2,1]", "S21"),
            ("S[1,2]", "S12"),
            ("S[2,2]", "S22"),
        )
        extracted: dict[str, np.ndarray] = {}
        for source_col, target in col_pairs:
            if source_col in columns and target not in extracted:
                extracted[target] = np.asarray(
                    [complex(value) for value in frame[source_col]], dtype=np.complex128
                )
        if {"S11", "S21", "S12", "S22"} <= set(extracted):
            frequencies = freq_arr
            s11 = extracted["S11"]
            s21 = extracted["S21"]
            s12 = extracted["S12"]
            s22 = extracted["S22"]
            break
    if (
        frequencies is None
        or s11 is None
        or s21 is None
        or s12 is None
        or s22 is None
    ):
        return _empty(
            status="error",
            message="dataset did not expose all four S-parameters on a freq grid.",
        )
    return {
        "frequencies_Hz": frequencies,
        "S11": s11,
        "S21": s21,
        "S12": s12,
        "S22": s22,
        "analysis_label": "ads_hpeesofsim",
    }


def _find_ads_python() -> Path | None:
    """Locate the Python interpreter bundled with ADS."""

    hpeesof_dir = os.environ.get("HPEESOF_DIR")
    if hpeesof_dir:
        candidate = Path(hpeesof_dir) / "tools" / "python" / "python.exe"
        if candidate.exists():
            return candidate
        candidate = Path(hpeesof_dir) / "tools" / "python" / "python"
        if candidate.exists():
            return candidate
    # Common Windows install locations as a last resort. Each search is
    # cheap; this code path is exercised only when HPEESOF_DIR is unset
    # but an ADS install actually exists.
    for base in (
        Path(r"C:\Program Files\Keysight"),
        Path(r"C:\Program Files (x86)\Keysight"),
    ):
        if not base.exists():
            continue
        for entry in base.iterdir():
            candidate = entry / "tools" / "python" / "python.exe"
            if candidate.exists():
                return candidate
    return None


def _to_complex(values: list[Any]) -> np.ndarray:
    out = np.empty(len(values), dtype=np.complex128)
    for index, value in enumerate(values):
        if isinstance(value, (list, tuple)) and len(value) == 2:
            out[index] = complex(float(value[0]), float(value[1]))
        else:
            out[index] = complex(value)
    return out


def _empty(status: str, message: str | None = None) -> dict[str, Any]:
    return {
        "frequencies_Hz": np.zeros(0, dtype=float),
        "S11": np.zeros(0, dtype=np.complex128),
        "S21": np.zeros(0, dtype=np.complex128),
        "S12": np.zeros(0, dtype=np.complex128),
        "S22": np.zeros(0, dtype=np.complex128),
        "analysis_label": status,
        "error_message": message,
    }


# ===========================================================================
# Analysis-agnostic raw dataset access
#
# ``parse_s_parameter_results`` above interprets the dataset as an S-parameter
# result. ``read_raw_dataset`` below does no interpretation at all: it returns
# every column of every varblock verbatim, so a caller that needs a different
# analysis (Harmonic Balance, in particular) structures the raw columns
# itself. Both share the in-process / ADS-Python-subprocess bridge.
# ===========================================================================


class RawDatasetError(RuntimeError):
    """Raised when a ``.ds`` dataset cannot be read by either bridge."""


@dataclass(frozen=True)
class RawVarBlock:
    """One ADS dataset varblock as plain in-memory columns.

    ``columns`` maps a column name to its values. Every value is a
    Python ``complex`` -- a real column (a frequency axis, a mixing
    index) carries a zero imaginary part -- so a consumer reads any
    column uniformly and takes ``.real`` where it knows the column is
    real.
    """

    name: str
    columns: Mapping[str, tuple[complex, ...]]


@dataclass(frozen=True)
class RawDataset:
    """Every varblock of a ``.ds`` file, read with no analysis applied.

    The low-level, analysis-agnostic dataset structure. The HB reader
    (:mod:`tvtl_ads_builder_v2.analysis.hb_dataset`) builds on it; the
    S-parameter parser predates it and keeps its own path.
    """

    blocks: tuple[RawVarBlock, ...]

    def block_names(self) -> tuple[str, ...]:
        """Return the name of every varblock, in dataset order."""

        return tuple(block.name for block in self.blocks)


def _as_complex(value: Any) -> complex:
    """Coerce one dataset cell to ``complex`` (NaN/inf flattened to 0)."""

    try:
        result = complex(value)
    except (TypeError, ValueError):
        try:
            result = complex(float(getattr(value, "real", value)))
        except (TypeError, ValueError):
            return 0j
    real = 0.0 if (math.isnan(result.real) or math.isinf(result.real)) else result.real
    imag = 0.0 if (math.isnan(result.imag) or math.isinf(result.imag)) else result.imag
    return complex(real, imag)


def read_raw_dataset(dataset_path: str | Path) -> RawDataset:
    """Read every varblock of a ``.ds`` into a :class:`RawDataset`.

    Analysis-agnostic: no column is interpreted; every column of every
    varblock is returned verbatim. Uses the same in-process /
    ADS-Python-subprocess bridge as :func:`parse_s_parameter_results`.

    Raises
    ------
    FileNotFoundError
        If ``dataset_path`` does not exist.
    RawDatasetError
        If neither bridge can read the dataset.
    """

    path = Path(dataset_path).resolve()
    if not path.exists():
        raise FileNotFoundError(f"dataset not found at {path}")
    raw = _read_raw_in_process(path)
    if raw is not None:
        return raw
    raw = _read_raw_via_subprocess(path)
    if raw is not None:
        return raw
    raise RawDatasetError(
        f"keysight.ads.dataset not importable from either Python; "
        f"cannot read {path}."
    )


def _read_raw_in_process(path: Path) -> RawDataset | None:
    """Read the raw dataset directly, when ``keysight.ads.dataset`` imports."""

    try:
        from keysight.ads import dataset as ds_mod  # type: ignore
    except Exception:
        try:
            import keysight.ads.dataset as ds_mod  # type: ignore
        except Exception:
            return None
    return _extract_all_varblocks(ds_mod, path)


def _extract_all_varblocks(ds_mod: Any, path: Path) -> RawDataset:
    """Common raw-read body: every varblock, every column, no filtering."""

    with ds_mod.open(str(path), "r") as ds:
        items = list(ds.varblocks.items())
    blocks: list[RawVarBlock] = []
    for name, block in items:
        try:
            frame = block.to_dataframe()
        except Exception:
            continue
        index = getattr(frame, "index", None)
        if index is not None and getattr(index, "name", None):
            frame = frame.reset_index()
        columns: dict[str, tuple[complex, ...]] = {}
        for col in list(getattr(frame, "columns", [])):
            columns[str(col)] = tuple(_as_complex(v) for v in frame[col])
        blocks.append(RawVarBlock(name=str(name), columns=columns))
    return RawDataset(blocks=tuple(blocks))


def _read_raw_via_subprocess(path: Path) -> RawDataset | None:
    """Read the raw dataset through the ADS-bundled Python interpreter."""

    ads_python = _find_ads_python()
    if ads_python is None:
        return None
    worker_path = path.parent / RAW_PARSER_WORKER_FILENAME
    worker_path.write_text(_RAW_WORKER_SCRIPT, encoding="utf-8")
    env = os.environ.copy()
    if not env.get("HPEESOF_DIR"):
        env["HPEESOF_DIR"] = str(ads_python.parents[2])
    try:
        completed = subprocess.run(
            [str(ads_python), str(worker_path), str(path)],
            text=True,
            capture_output=True,
            check=False,
            timeout=120,
            env=env,
        )
    except subprocess.TimeoutExpired as exc:
        raise RawDatasetError("raw-dataset subprocess timed out.") from exc
    stdout = completed.stdout or ""
    try:
        payload = json.loads(stdout) if stdout.strip() else None
    except json.JSONDecodeError:
        first_brace = stdout.find("{")
        payload = json.loads(stdout[first_brace:]) if first_brace >= 0 else None
    if not isinstance(payload, dict) or payload.get("status") != "ok":
        message = (
            payload.get("error") if isinstance(payload, dict)
            else f"no JSON from worker; stderr={completed.stderr!r}"
        )
        raise RawDatasetError(f"raw-dataset worker failed: {message}")
    return _raw_dataset_from_payload(payload)


def _raw_dataset_from_payload(payload: Mapping[str, Any]) -> RawDataset:
    """Reconstruct a :class:`RawDataset` from the worker's JSON payload."""

    blocks: list[RawVarBlock] = []
    for block in payload.get("blocks", []):
        columns = {
            str(name): tuple(
                complex(float(pair[0]), float(pair[1])) for pair in values
            )
            for name, values in dict(block.get("columns", {})).items()
        }
        blocks.append(RawVarBlock(name=str(block.get("name", "")), columns=columns))
    return RawDataset(blocks=tuple(blocks))


_WORKER_SCRIPT = '''"""ADS-Python parser worker for tvtl_ads_builder_v2.

Reads a hpeesofsim .ds dataset and prints a JSON payload on stdout. The
parent process (running under system Python) consumes that JSON.
"""

import json
import math
import sys


def main() -> int:
    if len(sys.argv) < 2:
        print(json.dumps({"status": "error", "error": "no dataset path given"}))
        return 1
    path = sys.argv[1]
    try:
        from keysight.ads import dataset as ds_mod
    except Exception as exc:
        print(json.dumps({"status": "error", "error": "keysight.ads.dataset import failed: " + repr(exc)}))
        return 2
    payload = {"status": "error", "error": "no usable S-parameter block found"}
    try:
        with ds_mod.open(path, "r") as ds:
            for name, block in ds.varblocks.items():
                try:
                    frame = block.to_dataframe()
                except Exception:
                    continue
                if getattr(frame, "index", None) is not None and frame.index.name == "freq":
                    frame = frame.reset_index()
                columns = list(getattr(frame, "columns", []))
                if "freq" not in columns:
                    continue
                col_pairs = (
                    ("S(1,1)", "S11"),
                    ("S(2,1)", "S21"),
                    ("S(1,2)", "S12"),
                    ("S(2,2)", "S22"),
                    ("S[1,1]", "S11"),
                    ("S[2,1]", "S21"),
                    ("S[1,2]", "S12"),
                    ("S[2,2]", "S22"),
                )
                grabbed = {}
                for src, target in col_pairs:
                    if src in columns and target not in grabbed:
                        grabbed[target] = src
                if len(grabbed) < 4:
                    continue
                freqs = []
                for value in frame["freq"]:
                    try:
                        freqs.append(float(getattr(value, "real", value)))
                    except Exception:
                        freqs.append(float("nan"))
                def _column_to_pairs(col_name):
                    out = []
                    for value in frame[col_name]:
                        try:
                            c = complex(value)
                        except Exception:
                            c = complex(float("nan"), float("nan"))
                        real = c.real
                        imag = c.imag
                        if math.isnan(real) or math.isinf(real):
                            real = 0.0
                        if math.isnan(imag) or math.isinf(imag):
                            imag = 0.0
                        out.append([real, imag])
                    return out
                payload = {
                    "status": "ok",
                    "frequencies_Hz": freqs,
                    "S11": _column_to_pairs(grabbed["S11"]),
                    "S21": _column_to_pairs(grabbed["S21"]),
                    "S12": _column_to_pairs(grabbed["S12"]),
                    "S22": _column_to_pairs(grabbed["S22"]),
                }
                break
    except Exception as exc:
        payload = {"status": "error", "error": "parse failure: " + repr(exc)}
    print(json.dumps(payload))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
'''


_RAW_WORKER_SCRIPT = '''"""ADS-Python raw-dataset worker for tvtl_ads_builder_v2.

Reads every varblock of a .ds dataset and prints a JSON payload on
stdout: {"status": "ok", "blocks": [{"name": .., "columns":
{col: [[re, im], ...]}}]}. No column is interpreted -- the parent
process structures the raw columns for whatever analysis it needs.
"""

import json
import math
import sys


def _pair(value):
    try:
        c = complex(value)
    except Exception:
        try:
            c = complex(float(getattr(value, "real", value)))
        except Exception:
            return [0.0, 0.0]
    re = c.real
    im = c.imag
    if math.isnan(re) or math.isinf(re):
        re = 0.0
    if math.isnan(im) or math.isinf(im):
        im = 0.0
    return [re, im]


def main() -> int:
    if len(sys.argv) < 2:
        print(json.dumps({"status": "error", "error": "no dataset path given"}))
        return 1
    path = sys.argv[1]
    try:
        from keysight.ads import dataset as ds_mod
    except Exception as exc:
        print(json.dumps({"status": "error", "error": "import failed: " + repr(exc)}))
        return 2
    blocks = []
    try:
        with ds_mod.open(path, "r") as ds:
            for name, block in ds.varblocks.items():
                try:
                    frame = block.to_dataframe()
                except Exception:
                    continue
                index = getattr(frame, "index", None)
                if index is not None and getattr(index, "name", None):
                    frame = frame.reset_index()
                columns = {}
                for col in list(getattr(frame, "columns", [])):
                    columns[str(col)] = [_pair(v) for v in frame[col]]
                blocks.append({"name": str(name), "columns": columns})
    except Exception as exc:
        print(json.dumps({"status": "error", "error": "read failure: " + repr(exc)}))
        return 0
    print(json.dumps({"status": "ok", "blocks": blocks}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
'''


