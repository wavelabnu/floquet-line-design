"""ADS-free gain definitions for self-pumped validation reports."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
import math
from typing import Literal, Mapping, Sequence


Number = int | float


def _finite(value: Number, *, name: str) -> float:
    out = float(value)
    if not math.isfinite(out):
        raise ValueError(f"{name} must be finite.")
    return out


def _non_negative_power(value: Number, *, name: str) -> float:
    out = _finite(value, name=name)
    if out < 0.0:
        raise ValueError(f"{name} must be non-negative.")
    return out


def db20(x: float) -> float:
    """Return ``20*log10(x)`` for a positive finite amplitude ratio."""

    value = _finite(x, name="x")
    if value <= 0.0:
        raise ValueError("db20 requires a positive value.")
    return 20.0 * math.log10(value)


def db10(x: float) -> float:
    """Return ``10*log10(x)`` for a positive finite power ratio."""

    value = _finite(x, name="x")
    if value <= 0.0:
        raise ValueError("db10 requires a positive value.")
    return 10.0 * math.log10(value)


def db20_or_none(x: float | None) -> float | None:
    if x is None or x <= 0.0:
        return None
    return db20(x)


def db10_or_none(x: float | None) -> float | None:
    if x is None or x <= 0.0:
        return None
    return db10(x)


def voltage_gain(v_out: complex, v_in: complex) -> float | None:
    """Return ``abs(v_out) / abs(v_in)``, or ``None`` for zero input."""

    denom = abs(complex(v_in))
    if denom == 0.0:
        return None
    return abs(complex(v_out)) / denom


def voltage_gain_db(v_out: complex, v_in: complex) -> float | None:
    return db20_or_none(voltage_gain(v_out, v_in))


def power_gain(p_out: float, p_in: float) -> float | None:
    """Return ``p_out / p_in`` with explicit negative-power validation."""

    numerator = _non_negative_power(p_out, name="p_out")
    denominator = _non_negative_power(p_in, name="p_in")
    if denominator == 0.0:
        return None
    return numerator / denominator


def power_gain_db(p_out: float, p_in: float) -> float | None:
    return db10_or_none(power_gain(p_out, p_in))


def load_power_from_voltage(
    v_load: complex,
    r_load: float,
    convention: Literal["rms", "peak"],
) -> float:
    """Return load power from a complex load-voltage phasor.

    ``rms`` uses ``|V|^2/R``. ``peak`` uses ``|V|^2/(2R)``.
    """

    resistance = _finite(r_load, name="r_load")
    if resistance <= 0.0:
        raise ValueError("r_load must be positive.")
    magnitude_sq = abs(complex(v_load)) ** 2
    if convention == "rms":
        return magnitude_sq / resistance
    if convention == "peak":
        return magnitude_sq / (2.0 * resistance)
    raise ValueError("convention must be 'rms' or 'peak'.")


def transducer_gain(p_load: float, p_avs: float) -> float | None:
    """Return ``p_load / p_avs`` relative to available signal-source power."""

    return power_gain(p_load, p_avs)


def transducer_gain_db(p_load: float, p_avs: float) -> float | None:
    return db10_or_none(transducer_gain(p_load, p_avs))


def full_sum_gain(
    powers_out_by_harmonic: Mapping[int, float],
    p_in: float,
    harmonics: Sequence[int],
) -> float | None:
    """Return summed output power over ``harmonics`` divided by ``p_in``.

    Missing harmonics return ``None``. Callers that want notes should use
    :func:`missing_harmonics` and record the returned indices explicitly.
    """

    denominator = _non_negative_power(p_in, name="p_in")
    if denominator == 0.0:
        return None
    missing = missing_harmonics(powers_out_by_harmonic, harmonics)
    if missing:
        return None
    total = 0.0
    for harmonic in harmonics:
        total += _non_negative_power(
            powers_out_by_harmonic[int(harmonic)],
            name=f"power harmonic {int(harmonic)}",
        )
    return total / denominator


def full_sum_gain_db(
    powers_out_by_harmonic: Mapping[int, float],
    p_in: float,
    harmonics: Sequence[int],
) -> float | None:
    return db10_or_none(full_sum_gain(powers_out_by_harmonic, p_in, harmonics))


def missing_harmonics(
    powers_out_by_harmonic: Mapping[int, float],
    harmonics: Sequence[int],
) -> list[int]:
    present = {int(key) for key in powers_out_by_harmonic.keys()}
    return [int(harmonic) for harmonic in harmonics if int(harmonic) not in present]


def dbm_to_watts(dbm: float) -> float:
    """Convert dBm to watts."""

    return 1.0e-3 * (10.0 ** (_finite(dbm, name="dbm") / 10.0))


@dataclass(frozen=True)
class GainMetricReport:
    """JSON-safe container for gain metrics that may be unavailable."""

    voltage_gain_db: float | None = None
    load_power_gain_db: float | None = None
    transducer_gain_db: float | None = None
    coherent_gain_db: float | None = None
    full_sum_gain_db: float | None = None
    pump_normalized_signal: float | None = None
    pump_normalized_sum: float | None = None
    harmonics_in_sum: list[int] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return asdict(self)


def gain_definitions_payload() -> dict[str, dict[str, str]]:
    """Return the formula set embedded in JSON reports."""

    return {
        "voltage_gain_signal_frequency": {
            "linear": "G_V(fs) = |V_out(fs)| / |V_in(fs)|",
            "db": "G_V,dB(fs) = 20 log10(G_V(fs))",
        },
        "load_power_gain_signal_frequency": {
            "linear": "G_P(fs) = P_L(fs) / P_in(fs)",
            "db": "G_P,dB(fs) = 10 log10(G_P(fs))",
            "p_in": "power entering the DUT at the RF input plane",
        },
        "transducer_gain_signal_frequency": {
            "linear": "G_T(fs) = P_L(fs) / P_avs(fs)",
            "db": "G_T,dB(fs) = 10 log10(G_T(fs))",
        },
        "s_like_insertion_gain": {
            "linear": "G_S21(fs) = |S21(fs)|^2",
            "db": "G_S21,dB(fs) = 20 log10(|S21(fs)|)",
        },
        "coherent_signal_gain": {
            "voltage": "G_coh,V = |V_out(fs)| / |V_ref(fs)|",
            "power": "G_coh,P = P_out(fs) / P_in(fs)",
        },
        "full_sum_gain": {
            "linear": "G_sum = sum(P_out(k fs) for k in K_out) / P_in(fs)",
            "db": "G_sum,dB = 10 log10(G_sum)",
        },
        "pump_normalized_conversion": {
            "signal": "eta_sig = P_out(fs) / P_pump,av",
            "sum": "eta_sum = sum(P_out(k fs) for k in K_out) / P_pump,av",
        },
    }


def gain_definitions_markdown_section() -> str:
    """Return the user-facing equations embedded in generated reports."""

    return "\n".join(
        [
            "## Gain definitions used in this report",
            "",
            "### Voltage gain at signal frequency",
            "",
            "```text",
            "G_V(fs) = |V_out(fs)| / |V_in(fs)|",
            "G_V,dB(fs) = 20 log10(G_V(fs))",
            "```",
            "",
            (
                "This is a phasor-amplitude ratio at the signal-frequency HB "
                "component. RMS/peak convention cancels if both voltages use "
                "the same convention."
            ),
            "",
            "### Load power gain at signal frequency",
            "",
            "```text",
            "G_P(fs) = P_L(fs) / P_in(fs)",
            "G_P,dB(fs) = 10 log10(G_P(fs))",
            "```",
            "",
            (
                "`P_in(fs)` means power entering the DUT at the RF input "
                "plane. If that input-plane power is unavailable, reports "
                "write `null`/blank values with notes."
            ),
            "",
            "### Transducer gain",
            "",
            "```text",
            "G_T(fs) = P_L(fs) / P_avs(fs)",
            "G_T,dB(fs) = 10 log10(G_T(fs))",
            "```",
            "",
            (
                "ADS `P_1Tone.P` is treated as available source power only "
                "where that convention has been verified for the comparison "
                "mode."
            ),
            "",
            "### S-like / insertion gain",
            "",
            "```text",
            "G_S21(fs) = |S21(fs)|^2",
            "G_S21,dB(fs) = 20 log10(|S21(fs)|)",
            "```",
            "",
            (
                "Large-signal HB power gain is not equated with small-signal "
                "`S21` unless the comparison mode explicitly supports it."
            ),
            "",
            "### Coherent signal gain",
            "",
            "```text",
            "G_coh,V = |V_out(fs)| / |V_ref(fs)|",
            "G_coh,V,dB = 20 log10(G_coh,V)",
            "```",
            "",
            "or, if normalized to input power:",
            "",
            "```text",
            "G_coh,P = P_out(fs) / P_in(fs)",
            "G_coh,P,dB = 10 log10(G_coh,P)",
            "```",
            "",
            (
                "This report states which normalization is used for each "
                "available coherent metric."
            ),
            "",
            "### Full-sum harmonic gain",
            "",
            "```text",
            "G_sum = (sum_{k in K_out} P_out(k fs)) / P_in(fs)",
            "G_sum,dB = 10 log10(G_sum)",
            "```",
            "",
            (
                "This is not the same as coherent signal gain because it "
                "includes selected output harmonics/idlers. The selected "
                "`K_out` is written into JSON and Markdown reports."
            ),
            "",
            "### Pump-normalized conversion metrics",
            "",
            "```text",
            "eta_sig = P_out(fs) / P_pump,av",
            "eta_sum = (sum_{k in K_out} P_out(k fs)) / P_pump,av",
            "```",
            "",
            (
                "These are pump-normalized conversion/efficiency metrics, "
                "not ordinary gain."
            ),
        ]
    )


def gain_equations_markdown_section() -> str:
    """Return the compact equation block expected near the top of reports."""

    return "\n".join(
        [
            "## Gain equations used",
            "",
            "Voltage gain:",
            "",
            "```text",
            "G_V(fs) = |V_out(fs)| / |V_in(fs)|",
            "G_V,dB(fs) = 20 log10(G_V(fs))",
            "```",
            "",
            "Load power gain:",
            "",
            "```text",
            "G_P(fs) = P_L(fs) / P_in(fs)",
            "G_P,dB(fs) = 10 log10(G_P(fs))",
            "```",
            "",
            "Transducer gain:",
            "",
            "```text",
            "G_T(fs) = P_L(fs) / P_avs(fs)",
            "G_T,dB(fs) = 10 log10(G_T(fs))",
            "```",
            "",
            "Coherent signal gain:",
            "",
            "```text",
            "G_coh,V = |V_out(fs)| / |V_ref(fs)|",
            "G_coh,V,dB = 20 log10(G_coh,V)",
            "```",
            "",
            "Full-sum harmonic gain:",
            "",
            "```text",
            "G_sum = (sum_{k in K_out} P_out(k fs)) / P_in(fs)",
            "G_sum,dB = 10 log10(G_sum)",
            "```",
        ]
    )
