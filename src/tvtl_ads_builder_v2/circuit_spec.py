"""Shared source-of-truth specs for small validation circuits."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any


@dataclass(frozen=True)
class SelfPumpedCircuitSpec:
    """Minimal self-pumped circuit used by analytical, netlist, and OA paths."""

    fs_hz: float = 2.0e9
    psig_dbm: float = -20.0
    ppump_dbm: float = 0.0
    source_z_ohm: float = 100.0
    load_z_ohm: float = 50.0
    line_z_ohm: float = 50.0
    line_e_deg: float = 90.0
    line_freq_hz: float | None = None
    hb_order: int = 5
    hb_max_order: int = 4

    def __post_init__(self) -> None:
        values = {
            "fs_hz": self.fs_hz,
            "psig_dbm": self.psig_dbm,
            "ppump_dbm": self.ppump_dbm,
            "source_z_ohm": self.source_z_ohm,
            "load_z_ohm": self.load_z_ohm,
            "line_z_ohm": self.line_z_ohm,
            "line_e_deg": self.line_e_deg,
            "hb_order": self.hb_order,
            "hb_max_order": self.hb_max_order,
        }
        for name, value in values.items():
            if name in {"hb_order", "hb_max_order"}:
                if int(value) <= 0:
                    raise ValueError(f"{name} must be positive")
                continue
            if float(value) <= 0.0 and name not in {"psig_dbm", "ppump_dbm"}:
                raise ValueError(f"{name} must be positive")
        if self.line_freq_hz is not None and float(self.line_freq_hz) <= 0.0:
            raise ValueError("line_freq_hz must be positive when supplied")

    @property
    def effective_line_freq_hz(self) -> float:
        """Frequency used by the TLIN electrical-length parameter."""

        return self.fs_hz if self.line_freq_hz is None else self.line_freq_hz

    @property
    def source_z_parallel_ohm(self) -> float:
        """Equivalent source impedance of the two finite same-line sources."""

        z = float(self.source_z_ohm)
        return 1.0 / (1.0 / z + 1.0 / z)

    @property
    def expected_rf_main_net(self) -> str:
        return "RF_MAIN"

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["line_freq_hz"] = self.effective_line_freq_hz
        data["source_z_parallel_ohm"] = self.source_z_parallel_ohm
        data["expected_rf_main_net"] = self.expected_rf_main_net
        return data


def default_self_pumped_spec(*, freq_ghz: float = 2.0) -> SelfPumpedCircuitSpec:
    """Return the default minimal self-pumped validation circuit."""

    return SelfPumpedCircuitSpec(fs_hz=float(freq_ghz) * 1.0e9)


def ads_number(value: float) -> str:
    """Format a numeric value for compact ADS netlist/parameter text."""

    return f"{float(value):.12g}"


def ads_ghz(value_hz: float) -> str:
    """Format a frequency in Hz as an ADS GHz value string."""

    return f"{float(value_hz) * 1.0e-9:.12g} GHz"
