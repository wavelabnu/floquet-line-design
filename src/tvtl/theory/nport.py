"""Generic calibrated frequency-dependent N-port network blocks."""

from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np


@dataclass(frozen=True, slots=True)
class CalibratedNPort:
    """Frequency-dependent N-port network with explicit port ordering."""

    frequencies_hz: np.ndarray
    matrices: np.ndarray
    matrix_kind: str
    port_order: tuple[str, ...]
    reference_impedance_ohm: float = 50.0

    def __post_init__(self) -> None:
        frequencies = np.asarray(self.frequencies_hz, dtype=float)
        matrices = np.asarray(self.matrices, dtype=np.complex128)
        if frequencies.ndim != 1 or len(frequencies) == 0:
            raise ValueError("frequencies_hz must be a non-empty one-dimensional array.")
        if matrices.shape != (len(frequencies), len(self.port_order), len(self.port_order)):
            raise ValueError("matrices must have shape (frequency_count, port_count, port_count).")
        if self.matrix_kind not in {"s", "y"}:
            raise ValueError("matrix_kind must be 's' or 'y'.")
        if len(set(self.port_order)) != len(self.port_order):
            raise ValueError("port_order entries must be unique.")
        if not np.all(np.diff(frequencies) >= 0.0):
            raise ValueError("frequencies_hz must be sorted ascending.")

    @property
    def port_count(self) -> int:
        return len(self.port_order)

    def matrix_at(self, frequency_hz: float, *, kind: str | None = None) -> np.ndarray:
        target_kind = self.matrix_kind if kind is None else kind
        if target_kind not in {"s", "y"}:
            raise ValueError("kind must be 's' or 'y'.")
        matrix = _interpolate_matrix(float(frequency_hz), self.frequencies_hz, self.matrices)
        if target_kind == self.matrix_kind:
            return matrix
        if target_kind == "y":
            return s_to_y(matrix, self.reference_impedance_ohm)
        return y_to_s(matrix, self.reference_impedance_ohm)


def load_nport_csv(
    path: str | Path,
    *,
    matrix_kind: str,
    port_order: Sequence[str],
    reference_impedance_ohm: float = 50.0,
) -> CalibratedNPort:
    """Load an N-port S or Y matrix CSV with real/imag columns."""

    csv_path = Path(path)
    with csv_path.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise ValueError(f"{csv_path} contains no matrix rows.")
    port_count = len(port_order)
    matrices = np.zeros((len(rows), port_count, port_count), dtype=np.complex128)
    frequencies = np.zeros(len(rows), dtype=float)
    kind = matrix_kind.lower()
    for row_index, row in enumerate(rows):
        frequencies[row_index] = float(row["frequency_hz"])
        for i in range(port_count):
            for j in range(port_count):
                if kind == "s":
                    real_key = f"s{i + 1}{j + 1}_real"
                    imag_key = f"s{i + 1}{j + 1}_imag"
                elif kind == "y":
                    real_key = f"y{i + 1}{j + 1}_real_siemens"
                    imag_key = f"y{i + 1}{j + 1}_imag_siemens"
                else:
                    raise ValueError("matrix_kind must be 's' or 'y'.")
                matrices[row_index, i, j] = complex(float(row[real_key]), float(row[imag_key]))
    return CalibratedNPort(
        frequencies_hz=frequencies,
        matrices=matrices,
        matrix_kind=kind,
        port_order=tuple(str(item) for item in port_order),
        reference_impedance_ohm=reference_impedance_ohm,
    )


def load_nport_calibration(path: str | Path) -> CalibratedNPort:
    """Load a JSON calibration artifact that points at S or Y matrix data."""

    calibration_path = Path(path)
    payload = json.loads(calibration_path.read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping):
        raise ValueError("Calibration artifact must contain a JSON object.")
    port_order = tuple(str(item) for item in payload["port_order"])
    reference_impedance = float(payload.get("reference_impedance_ohm", 50.0))
    if payload.get("sparameter_csv"):
        matrix_path = _resolve_calibration_path(calibration_path, str(payload["sparameter_csv"]))
        return load_nport_csv(matrix_path, matrix_kind="s", port_order=port_order, reference_impedance_ohm=reference_impedance)
    if payload.get("y_matrix_csv"):
        matrix_path = _resolve_calibration_path(calibration_path, str(payload["y_matrix_csv"]))
        return load_nport_csv(matrix_path, matrix_kind="y", port_order=port_order, reference_impedance_ohm=reference_impedance)
    raise ValueError("Calibration artifact must reference sparameter_csv or y_matrix_csv.")


def s_to_y(s_matrix: np.ndarray, reference_impedance_ohm: float = 50.0) -> np.ndarray:
    s = np.asarray(s_matrix, dtype=np.complex128)
    identity = np.eye(s.shape[0], dtype=np.complex128)
    y0 = 1.0 / float(reference_impedance_ohm)
    return y0 * (identity - s) @ np.linalg.inv(identity + s)


def y_to_s(y_matrix: np.ndarray, reference_impedance_ohm: float = 50.0) -> np.ndarray:
    y = np.asarray(y_matrix, dtype=np.complex128)
    identity = np.eye(y.shape[0], dtype=np.complex128)
    z0 = float(reference_impedance_ohm)
    return (identity - z0 * y) @ np.linalg.inv(identity + z0 * y)


def s_to_abcd(s_matrix: np.ndarray, reference_impedance_ohm: float = 50.0) -> np.ndarray:
    s = np.asarray(s_matrix, dtype=np.complex128)
    if s.shape != (2, 2):
        raise ValueError("s_to_abcd requires a two-port S matrix.")
    s11, s12 = s[0, 0], s[0, 1]
    s21, s22 = s[1, 0], s[1, 1]
    if abs(s21) <= np.finfo(float).tiny:
        raise FloatingPointError("Cannot convert S to ABCD when S21 is zero.")
    z0 = float(reference_impedance_ohm)
    return np.asarray(
        [
            [((1 + s11) * (1 - s22) + s12 * s21) / (2 * s21), z0 * ((1 + s11) * (1 + s22) - s12 * s21) / (2 * s21)],
            [((1 - s11) * (1 - s22) - s12 * s21) / (2 * z0 * s21), ((1 - s11) * (1 + s22) + s12 * s21) / (2 * s21)],
        ],
        dtype=np.complex128,
    )


def _interpolate_matrix(frequency_hz: float, frequencies_hz: np.ndarray, matrices: np.ndarray) -> np.ndarray:
    if frequency_hz < float(frequencies_hz[0]) or frequency_hz > float(frequencies_hz[-1]):
        raise ValueError("Requested frequency is outside the calibrated range.")
    result = np.empty(matrices.shape[1:], dtype=np.complex128)
    for i in range(matrices.shape[1]):
        for j in range(matrices.shape[2]):
            values = matrices[:, i, j]
            real = np.interp(frequency_hz, frequencies_hz, values.real)
            imag = np.interp(frequency_hz, frequencies_hz, values.imag)
            result[i, j] = complex(real, imag)
    return result


def _resolve_calibration_path(calibration_path: Path, value: str) -> Path:
    path = Path(value)
    if path.is_absolute():
        return path
    return calibration_path.parent / path


