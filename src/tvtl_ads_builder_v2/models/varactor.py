"""Varactor (device-C) models for the chain model registry.

The varactor is the **single source** of the device
capacitance consumed by all three legs (emitter SDD charge body, analytical
small-signal C / modulation, optimizer pump calibration). Three registered variants:

* ``power_law``  -> a :class:`tvtl.varactors.RealisticVaractor` (CJ0/VJ/M Skyworks-SPICE
  C-V + package L/R). **This is the reduced default** (Varactor A is
  ``power_law(CJ0=0.3pF, VJ=3.5, M=1.7, CP=0, L_pkg=0.3nH, R=0.5)``). Builds the exact
  same object the current workflow uses -> byte-identical recovery path.
* ``effective_c`` -> a single constant capacitance (``RealisticVaractor`` in
  ``ideal_linear`` mode, optionally with package L/R). A constant C has no pump
  response, so the self-pumped path rejects it (passive / static use only) -- stated,
  not fudged.
* ``polynomial`` -> a :class:`PolynomialVaractor` built from polynomial coefficients
  plus a C(V)/Q(V) flag. The emitter and analytical leg derive their charge body /
  small-signal C from the **same** coefficients (one definition, never two).

A non-``RealisticVaractor`` device only needs the duck-typed surface the three legs
read (see :class:`PolynomialVaractor`); enriched PT-paper devices register the same way.
"""

from __future__ import annotations

import cmath
import math
from dataclasses import dataclass, field

from tvtl.design.nonlinear_capacitor import ChargePolynomialModel
from tvtl.varactors import RealisticVaractor

__all__ = [
    "PolynomialVaractor",
    "build_power_law_varactor",
    "build_effective_c_varactor",
    "build_polynomial_varactor",
    "smv1405",
]


# --------------------------------------------------------------------------- #
# Shared harmonic band sum                                                     #
# --------------------------------------------------------------------------- #
def _band_sum(taylor: list[float], swing: float, d: int, max_k: int) -> float:
    """One band ``d`` of the cosine-driven capacitance expansion.

    ``C(V_bias + swing*cos(theta)) = sum_d C_d e^{i d theta}`` with

        C_d = sum_{k >= d, (k-d) even} taylor[k] * swing^k * binom(k,(k-d)//2) / 2^k

    -- the identical formula :class:`RealisticVaractor` uses, factored out for reuse
    by :class:`PolynomialVaractor` (``RealisticVaractor`` itself is left untouched).
    """

    acc = 0.0
    for k in range(max(d, 0), max_k + 1):
        if (k - d) % 2 != 0:
            continue
        acc += taylor[k] * (swing ** k) * math.comb(k, (k - d) // 2) / (2.0 ** k)
    return acc


# --------------------------------------------------------------------------- #
# Polynomial varactor                                                          #
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class PolynomialVaractor:
    """A varactor whose junction C-V is an explicit polynomial (C(V) or Q(V)).

    Duck-types the surface the three legs read off ``RealisticVaractor``:
    ``ideal_linear``, ``C0_f``, ``dC_dV_f_per_v``, ``R_series_ohm``, ``L_pkg_h``,
    ``charge_polynomial()``, ``effective_capacitance_coeffs(...)``,
    ``capacitance_coeffs_from_junction_vpump(...)``, ``m_effective(...)``,
    ``pump_divider(...)``, ``control_port_capacitance_coeffs(...)``.

    Parameters
    ----------
    coeffs : tuple[float, ...]
        Polynomial coefficients ``[c0, c1, ...]`` (power-indexed). Interpreted per
        ``form``.
    form : str
        ``"CV"`` -- ``coeffs`` are the capacitance Taylor coefficients about the bias
        point, ``C(V_bias + u) = c0 + c1 u + c2 u^2 + ...`` (so ``a_k = coeffs[k]``).
        ``"QV"`` -- ``coeffs`` are the charge Taylor coefficients ``Q(u) = q0 + q1 u +
        q2 u^2 + ...`` (``q0`` should be 0); the capacitance is the derivative
        ``a_k = (k+1) * coeffs[k+1]``.
    L_pkg_h, R_series_ohm : float
        Package series inductance / resistance (signal-path branch wrap), as for
        ``RealisticVaractor``.
    CP_f : float
        Parallel package capacitance, folded into ``a0`` (the static C).
    V_bias_v : float
        Bias point (carried for provenance; ``coeffs`` are assumed already expanded
        about it).
    """

    coeffs: tuple[float, ...]
    form: str = "CV"
    L_pkg_h: float = 0.0
    R_series_ohm: float = 0.0
    CP_f: float = 0.0
    V_bias_v: float = 0.0
    ideal_linear: bool = field(default=False, init=False)

    def __post_init__(self) -> None:
        if self.form not in ("CV", "QV"):
            raise ValueError(f"form must be 'CV' or 'QV' (got {self.form!r}).")
        if not self.coeffs:
            raise ValueError("coeffs must be non-empty.")
        if float(self.L_pkg_h) < 0.0 or float(self.R_series_ohm) < 0.0 or float(self.CP_f) < 0.0:
            raise ValueError("L_pkg_h / R_series_ohm / CP_f must be non-negative.")
        if self._a[0] <= 0.0:
            raise ValueError(
                f"static capacitance a0 must be positive (got {self._a[0]}); for "
                f"form='QV' that is coeffs[1] (the linear charge term)."
            )

    # -- capacitance Taylor coeffs about the bias point (length 6, padded) -- #
    @property
    def _a(self) -> list[float]:
        c = [float(x) for x in self.coeffs]
        if self.form == "CV":
            a = list(c)
        else:  # QV: differentiate -> a_k = (k+1) * q_{k+1}
            a = [float(k + 1) * c[k + 1] for k in range(len(c) - 1)]
        a = (a + [0.0] * 6)[:6]
        a[0] = a[0] + float(self.CP_f)  # CP fold (parallel package cap)
        return a

    @property
    def _max_k(self) -> int:
        a = self._a
        last = 0
        for k in range(1, 6):
            if a[k] != 0.0:
                last = k
        return last

    # -- duck-typed attributes -------------------------------------------- #
    @property
    def C0_f(self) -> float:
        return self._a[0]

    @property
    def dC_dV_f_per_v(self) -> float:
        return self._a[1]

    @property
    def reference_tag(self) -> str:
        return (
            f"polynomial[{self.form}](C0={self._a[0] * 1e12:.3g}pF,"
            f"order={self._max_k},L_pkg={float(self.L_pkg_h) * 1e9:.3g}nH,"
            f"R={float(self.R_series_ohm):.3g}ohm)"
        )

    # -- charge body (SDD2P) ---------------------------------------------- #
    def charge_polynomial(self) -> ChargePolynomialModel:
        a = self._a
        return ChargePolynomialModel(
            c0_f=a[0], c1_f_per_v=a[1], c2_f_per_v2=a[2],
            c3_f_per_v3=a[3], c4_f_per_v4=a[4], c5_f_per_v5=a[5],
        )

    def C(self, v: float) -> float:
        a = self._a
        u = float(v) - float(self.V_bias_v)
        return sum(a[k] * u ** k for k in range(6))

    # -- pump-delivery helpers (identical to RealisticVaractor) ----------- #
    def pump_divider(self, f_pump_hz: float) -> complex:
        omega = 2.0 * math.pi * float(f_pump_hz)
        if omega == 0.0:
            raise ValueError("pump_divider is singular at DC (f = 0).")
        if float(self.L_pkg_h) == 0.0 and float(self.R_series_ohm) == 0.0:
            return complex(1.0, 0.0)
        z_c = 1.0 / (1j * omega * self.C0_f)
        z_pkg = float(self.R_series_ohm) + 1j * omega * float(self.L_pkg_h)
        return z_c / (z_c + z_pkg)

    def m_effective(self, f_pump_hz: float, m_branch: float) -> float:
        return float(m_branch) * abs(self.pump_divider(f_pump_hz))

    # -- harmonic capacitance spectra ------------------------------------- #
    def control_port_capacitance_coeffs(self, *, n_terms: int | None = None) -> list[float]:
        a = self._a
        max_k = self._max_k
        if n_terms is None:
            n_terms = max_k
        out = [self.C0_f]
        for k in range(1, n_terms + 1):
            out.append(a[k] if k <= 5 else 0.0)
        return out

    def effective_capacitance_coeffs(
        self, f_pump_hz: float, m_branch: float, *, phi: float = 0.0,
        n_harmonics: int | None = None,
    ) -> list[complex]:
        c0 = self.C0_f
        m_eff = self.m_effective(f_pump_hz, m_branch)
        dcdv = self.dC_dV_f_per_v
        linear = dcdv == 0.0
        if n_harmonics is None:
            n_harmonics = 1 if linear else max(self._max_k, 1)
        if n_harmonics < 1:
            raise ValueError("n_harmonics must be >= 1 (fundamental needed).")
        coeffs: list[complex] = [complex(c0)] + [0j] * n_harmonics
        coeffs[1] = 0.5 * m_eff * c0 * cmath.exp(1j * float(phi))
        if linear:
            return coeffs
        v_c = m_eff * c0 / dcdv
        a = self._a
        for d in range(1, n_harmonics + 1):
            coeffs[d] = _band_sum(a, v_c, d, self._max_k) * cmath.exp(1j * float(d) * float(phi))
        return coeffs

    def capacitance_coeffs_from_junction_vpump(
        self, v_pump: float, *, phi: float = 0.0, n_harmonics: int | None = None,
    ) -> list[complex]:
        max_k = self._max_k
        if n_harmonics is None:
            n_harmonics = max(max_k, 1)
        if n_harmonics < 1:
            raise ValueError("n_harmonics must be >= 1 (fundamental needed).")
        a = self.control_port_capacitance_coeffs(n_terms=5)
        coeffs: list[complex] = []
        for d in range(0, n_harmonics + 1):
            coeffs.append(_band_sum(a, float(v_pump), d, max_k) * cmath.exp(1j * float(d) * float(phi)))
        coeffs[0] = complex(coeffs[0].real, 0.0)
        return coeffs


# --------------------------------------------------------------------------- #
# Builders (registered in registry.py)                                         #
# --------------------------------------------------------------------------- #
def build_power_law_varactor(params: dict) -> RealisticVaractor:
    """``power_law`` -> ``RealisticVaractor`` (CJ0/VJ/M + package L/R). Reduced default."""

    return RealisticVaractor(
        CJ0_f=float(params["CJ0_f"]),
        VJ_v=float(params["VJ_v"]),
        M=float(params["M"]),
        CP_f=float(params.get("CP_f", 0.0)),
        L_pkg_h=float(params.get("L_pkg_h", 0.0)),
        R_series_ohm=float(params.get("R_series_ohm", 0.0)),
        V_bias_v=float(params.get("V_bias_v", 0.0)),
        taylor_order=int(params.get("taylor_order", 5)),
        c0_static_override_f=(
            None if params.get("c0_static_override_f") is None
            else float(params["c0_static_override_f"])
        ),
    )


def build_effective_c_varactor(params: dict) -> RealisticVaractor:
    """``effective_c`` -> a single constant C (``ideal_linear``), optional package L/R.

    A constant C has no pump response, so the self-pumped path rejects it; this variant
    is for passive / static use (or an externally-pumped path), not self-pumped gain.
    """

    return RealisticVaractor(
        CJ0_f=float(params["C0_f"]),
        VJ_v=1.0, M=1.0, CP_f=0.0,
        L_pkg_h=float(params.get("L_pkg_h", 0.0)),
        R_series_ohm=float(params.get("R_series_ohm", 0.0)),
        V_bias_v=0.0, c0_static_override_f=None, ideal_linear=True,
    )


def build_polynomial_varactor(params: dict) -> PolynomialVaractor:
    """``polynomial`` -> :class:`PolynomialVaractor` from coeffs + a C(V)/Q(V) flag."""

    return PolynomialVaractor(
        coeffs=tuple(float(x) for x in params["coeffs"]),
        form=str(params.get("form", "CV")),
        L_pkg_h=float(params.get("L_pkg_h", 0.0)),
        R_series_ohm=float(params.get("R_series_ohm", 0.0)),
        CP_f=float(params.get("CP_f", 0.0)),
        V_bias_v=float(params.get("V_bias_v", 0.0)),
    )


# --------------------------------------------------------------------------- #
# Named device: Skyworks SMV1405 (reference behavioural SDD C(v) fit)            #
# --------------------------------------------------------------------------- #
# The Skyworks SMV1405 abrupt-junction varactor as the REFERENCE workflow realises
# it in the active "different topology" design (cell ``circuit2_Symbol_SDD_mlin``):
# a behavioural SDD with a 5th-order *differential-capacitance* polynomial
#
#     C(v) = C0 + C1 v + C2 v^2 + C3 v^3 + C4 v^4 + C5 v^5
#
# and explicit series package parasitics ``R = 1 ohm``, ``L = 1.7 nH``. Coefficients are
# VERBATIM from the flattened netlist of the REFERENCE top design (the
# reconciliation study; netlist line 167):
#
#     C0=2.66e-12 C1=-1.36e-12 C2=7.76e-13 C3=-2.7e-13 C4=4.85e-14 C5=-3.43e-15
#
# The REFERENCE SDD uses ``I[1,0]=C(v)*f_nom*dv_dt`` (the C(v)-via-charge form). Our
# framework's SDD2P body is the *charge* polynomial Q(v); we INTEGRATE that C(v) to Q(v) and reuse the
# existing Q(v) machinery -- :class:`PolynomialVaractor` ``form="CV"`` does exactly this
# (``charge_polynomial()`` -> ``Q(v) = C0 v + C1 v^2/2 + ... + C5 v^6/6``, the SDD2P body).
# The consistency-pass identity ``C = dQ/dv`` recovers C0..C5 EXACTLY (regression-gated),
# so the two representations are the same physics. ``CP_f = 0`` (the branch has no parallel
# package cap; R and L are series), so the integration is fold-free and the round-trip is
# bit-exact term by term.
_REFERENCE_SMV_CV_COEFFS_F: tuple[float, ...] = (
    2.66e-12,   # C0
    -1.36e-12,  # C1
    7.76e-13,   # C2
    -2.7e-13,   # C3
    4.85e-14,   # C4
    -3.43e-15,  # C5
)
_REFERENCE_SMV_R_SERIES_OHM: float = 1.0    # the reference-device (SMV1405) Rvar
_REFERENCE_SMV_L_PKG_H: float = 1.7e-9      # the reference-device (SMV1405) Lvar


def smv1405(
    *,
    L_pkg_h: float = _REFERENCE_SMV_L_PKG_H,
    R_series_ohm: float = _REFERENCE_SMV_R_SERIES_OHM,
    CP_f: float = 0.0,
) -> PolynomialVaractor:
    """Skyworks SMV1405 (3 pF) as a 5th-order SDD ``C(v)`` polynomial.

    Returns a :class:`PolynomialVaractor` (``form="CV"``) carrying the device's
    verbatim differential-capacitance coefficients (``C0..C5`` above) and series
    package ``R = 1 ohm`` / ``L = 1.7 nH``. The device reuses the Q(v) machinery the
    rest of the framework consumes (the integrated charge body drives the SDD2P;
    ``C = dQ/dv`` round-trips to these coefficients exactly). Parasitics are
    overridable for sensitivity probes; the defaults are the datasheet/netlist values.

    Mirrors the :meth:`tvtl.varactors.RealisticVaractor.smv1245` named-device idiom for
    the polynomial-C(v) device class.
    """

    return PolynomialVaractor(
        coeffs=_REFERENCE_SMV_CV_COEFFS_F,
        form="CV",
        L_pkg_h=float(L_pkg_h),
        R_series_ohm=float(R_series_ohm),
        CP_f=float(CP_f),
        V_bias_v=0.0,
    )


