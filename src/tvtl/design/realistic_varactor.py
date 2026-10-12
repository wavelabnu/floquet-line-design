"""Realistic varactor device model (SMV1245-class) with package parasitics.

:class:`RealisticVaractor` parameterises a diode
varactor by its SPICE junction C-V law plus the package parasitics that
were identified as dispositive for pump delivery -- a series package
inductance ``L_pkg`` and resistance ``R_series`` between the branch tap and
the junction, and a parallel package capacitance ``CP``. It is a *strict
generalisation* of the ideal externally-pumped capacitor used by the
Level10IdealC cross-validation: :meth:`RealisticVaractor.ideal` zeroes every
parasitic and the C-V curvature so the device reduces to a constant ``C0``.

The class is the single source of truth that both framework legs consume:

* the **analytical Floquet leg**
  (:func:`tvtl_ads_builder_v2.analytical.level10_realistic_param_gain`) reads
  the effective capacitance modulation (via :meth:`m_effective`) and the
  branch package impedance to build the harmonic-ABCD branch;
* the **ADS device-spec emitter**
  (:func:`tvtl_ads_builder_v2.parity.level10_realistic_c_canonical`) reads
  :meth:`charge_polynomial` for the SDD2P ``Q(v)`` body and the package
  ``L`` / ``R`` for the explicit branch elements.

C-V convention
--------------
The junction law is the Skyworks SPICE form ``C(v) = CJ0 / (1 + v/VJ)^M``
(``v`` the instantaneous voltage across the junction; reverse bias is
``v > 0``, which *lowers* C -- the convention the DC-bias diagnostic
used). The charge polynomial is built by re-expanding this law around the
operating point ``V_bias`` via
:func:`tvtl.design.nonlinear_capacitor.smv1245_skyworks_spice_polynomial`
(the validated, mechanical Taylor expansion), so the higher-order C-V content
tracks the bias point.

CP folding (preserved exactly from the paper-fidelity canonical)
----------------------------------------------------------------
The parallel package capacitance ``CP`` is folded into the charge
polynomial's ``q[1]`` coefficient -- ``q[1] -> q[1] + CP`` -- rather than
emitted as a literal parallel ``C``. This is mathematically exact (the total
displacement current into the junction-||-CP pair is
``d(Q_junction + CP*v)/dt``) and is the identical convention
:func:`tvtl_ads_builder_v2.parity.canonical_circuits._q_poly_from_varactor`
applies; folding it *inside* :meth:`charge_polynomial` means callers never
have to know about the convention. See that function for the equivalence
proof and the OA-layout-engine context.
"""

from __future__ import annotations

import cmath
import math
from dataclasses import dataclass

from .nonlinear_capacitor import (
    ChargePolynomialModel,
    smv1245_skyworks_spice_polynomial,
)

__all__ = ["RealisticVaractor"]


@dataclass(frozen=True)
class RealisticVaractor:
    """A diode varactor with SPICE C-V law and package parasitics (SI units).

    Parameters
    ----------
    CJ0_f : float
        Zero-bias junction capacitance [F] (Skyworks ``CJ0``).
    VJ_v : float
        Junction barrier potential [V] (Skyworks ``VJ``).
    M : float
        Grading coefficient (Skyworks ``M``).
    CP_f : float, optional
        Parallel package capacitance [F]. Folded into the charge
        polynomial's ``q[1]`` (see module docstring). Default ``0.0``.
    L_pkg_h : float, optional
        Series package inductance [H] between the branch tap and the
        junction. Default ``0.0`` (ideal limit). SMV1245 datasheet value
        is 1.7 nH.
    R_series_ohm : float, optional
        Series package resistance [Ω]. Default ``0.0``. SMV1245 datasheet
        value is 1.5 Ω (the paper-fidelity canonical historically uses
        2.0 Ω; both are supported via this parameter).
    V_bias_v : float, optional
        DC bias point [V]; reverse bias is ``v > 0`` (lowers C). The charge
        polynomial is re-expanded around this point. Default ``0.0``.
    taylor_order : int, optional
        Q(v) Taylor order for the SDD2P body; capped at 5 by the
        :class:`ChargePolynomialModel` schema. Default 5.
    c0_static_override_f : float | None, optional
        Optional override of the *static* capacitance term ``q[1]`` (before
        the CP fold), mirroring the paper canonical's ``analytical_c0_f``
        (the paper-stated ``C0 = 7.0 pF`` that slightly differs from
        ``CJ0 = 6.9 pF``). ``None`` (default) uses the junction value
        ``C(V_bias)``. This is the knob that lets the realistic path
        reproduce the paper-fidelity canonical bit-for-bit when desired;
        :meth:`smv1245` leaves it ``None`` (true datasheet device).
    ideal_linear : bool, optional
        When ``True`` the device is a constant linear capacitor
        ``C(v) = CJ0`` (no curvature, the ``M``/``VJ`` fields are inert)
        -- the reduction :meth:`ideal` builds. Default ``False``.
    """

    CJ0_f: float
    VJ_v: float
    M: float
    CP_f: float = 0.0
    L_pkg_h: float = 0.0
    R_series_ohm: float = 0.0
    V_bias_v: float = 0.0
    taylor_order: int = 5
    c0_static_override_f: float | None = None
    ideal_linear: bool = False

    def __post_init__(self) -> None:
        if float(self.CJ0_f) <= 0.0:
            raise ValueError(f"CJ0_f must be positive (got {self.CJ0_f}).")
        if not self.ideal_linear:
            if float(self.VJ_v) <= 0.0:
                raise ValueError(f"VJ_v must be positive (got {self.VJ_v}).")
            if float(self.M) <= 0.0:
                raise ValueError(f"M must be positive (got {self.M}).")
            if 1.0 + float(self.V_bias_v) / float(self.VJ_v) <= 0.0:
                raise ValueError(
                    f"V_bias_v must satisfy V_bias > -VJ (got "
                    f"V_bias={self.V_bias_v}, VJ={self.VJ_v})."
                )
        if float(self.CP_f) < 0.0:
            raise ValueError(f"CP_f must be non-negative (got {self.CP_f}).")
        if float(self.L_pkg_h) < 0.0:
            raise ValueError(f"L_pkg_h must be non-negative (got {self.L_pkg_h}).")
        if float(self.R_series_ohm) < 0.0:
            raise ValueError(
                f"R_series_ohm must be non-negative (got {self.R_series_ohm})."
            )

    # ------------------------------------------------------------------ #
    # Constructors                                                       #
    # ------------------------------------------------------------------ #
    @classmethod
    def ideal(cls, C0_f: float) -> "RealisticVaractor":
        """Constructor for ideal constant-C mode (the Level10IdealC reduction).

        No parasitics, no CP, no C-V curvature, ``V_bias = 0``: the device is
        a constant linear capacitor ``C(v) = C0``. This is the reduction the
        analytical leg must reproduce bit-for-bit against
        :func:`...level10_ideal_c_param_gain` (the ideal-C reduction contract).
        """

        return cls(
            CJ0_f=float(C0_f), VJ_v=1.0, M=1.0, CP_f=0.0,
            L_pkg_h=0.0, R_series_ohm=0.0, V_bias_v=0.0,
            c0_static_override_f=None, ideal_linear=True,
        )

    @classmethod
    def smv1245(
        cls,
        *,
        V_bias_v: float = 0.0,
        L_pkg_h: float = 1.7e-9,
        R_series_ohm: float = 1.5,
        CP_f: float = 0.47e-12,
        taylor_order: int = 5,
    ) -> "RealisticVaractor":
        """Skyworks SMV1245-011LF defaults: CJ0=6.9 pF, VJ=3.5 V, M=1.7.

        Parasitics default to the datasheet values (L_pkg=1.7 nH,
        R_series=1.5 Ω, CP=0.47 pF) and are overridable so low-L_pkg
        variants are a single call away. ``c0_static_override_f`` is left
        ``None`` -- this is the *true datasheet* device (junction CJ0=6.9 pF),
        as distinct from the paper-fidelity canonical's 7.0 pF override.
        """

        return cls(
            CJ0_f=6.9e-12, VJ_v=3.5, M=1.7, CP_f=float(CP_f),
            L_pkg_h=float(L_pkg_h), R_series_ohm=float(R_series_ohm),
            V_bias_v=float(V_bias_v), taylor_order=int(taylor_order),
            c0_static_override_f=None, ideal_linear=False,
        )

    # ------------------------------------------------------------------ #
    # C-V curve                                                          #
    # ------------------------------------------------------------------ #
    def C(self, v: float) -> float:
        """Junction capacitance [F] at instantaneous voltage ``v``.

        ``CJ0 / (1 + v/VJ)^M`` (the package CP is *not* included -- this is
        the bare junction). Ideal mode returns ``CJ0`` for all ``v``.
        """

        if self.ideal_linear:
            return float(self.CJ0_f)
        bias_factor = 1.0 + float(v) / float(self.VJ_v)
        if bias_factor <= 0.0:
            raise ValueError(
                f"C(v) undefined for v <= -VJ (got v={v}, VJ={self.VJ_v})."
            )
        return float(self.CJ0_f) / bias_factor ** float(self.M)

    @property
    def C0_f(self) -> float:
        """Static shunt capacitance [F] at the bias point, *including* CP.

        ``C0 = C_static(V_bias) + CP`` where ``C_static`` is the
        :attr:`c0_static_override_f` if set, else the junction ``C(V_bias)``.
        This is the total capacitance the SDD2P presents (CP folded) and the
        capacitance the package divider / ``Z_branch`` use. Ideal mode
        returns ``CJ0`` (CP is zero).
        """

        if self.ideal_linear:
            return float(self.CJ0_f)
        if self.c0_static_override_f is not None:
            static = float(self.c0_static_override_f)
        else:
            static = self.C(float(self.V_bias_v))
        return static + float(self.CP_f)

    @property
    def dC_dV_f_per_v(self) -> float:
        """``dC/dv`` of the junction at the bias point [F/V].

        Analytic derivative of the SPICE law:
        ``dC/dv = -(M/VJ) * CJ0 * (1 + V_bias/VJ)^(-M-1)``. CP is
        voltage-independent so it does not contribute. Ideal mode returns
        ``0.0``.
        """

        if self.ideal_linear:
            return 0.0
        vj = float(self.VJ_v)
        bias_factor = 1.0 + float(self.V_bias_v) / vj
        return -(float(self.M) / vj) * float(self.CJ0_f) * bias_factor ** (
            -float(self.M) - 1.0
        )

    # ------------------------------------------------------------------ #
    # Charge polynomial (SDD2P body) -- CP folded internally             #
    # ------------------------------------------------------------------ #
    def charge_polynomial(self) -> ChargePolynomialModel:
        """Return the SDD2P charge model ``Q(v)`` for this device.

        Builds the Skyworks Taylor expansion of ``C(v)`` around ``V_bias``
        (reusing the validated
        :func:`...nonlinear_capacitor.smv1245_skyworks_spice_polynomial`),
        applies the optional static-C override on ``C0`` (mirroring the paper
        canonical's ``analytical_c0_f``), and folds ``CP`` into ``C0``
        (``C0 -> C0 + CP``). The returned model's
        ``charge_polynomial_coefficients`` is the ``q_poly_coefficients``
        tuple the SDD2P element consumes. Ideal mode returns a pure linear
        capacitor ``Q(v) = CJ0*v``.
        """

        if self.ideal_linear:
            return ChargePolynomialModel(c0_f=float(self.CJ0_f))
        model = smv1245_skyworks_spice_polynomial(
            cj0_f=float(self.CJ0_f), vj_v=float(self.VJ_v),
            m_grade=float(self.M), order=int(self.taylor_order),
            v_bias_v=float(self.V_bias_v),
        )
        coeffs = dict(model.coefficients)  # C0..C5 (capacitance convention)
        if self.c0_static_override_f is not None:
            coeffs["C0"] = float(self.c0_static_override_f)
        # CP fold: q[1] (== capacitance C0) carries junction + parallel CP.
        coeffs["C0"] = float(coeffs["C0"]) + float(self.CP_f)
        return ChargePolynomialModel(
            c0_f=coeffs["C0"], c1_f_per_v=coeffs["C1"], c2_f_per_v2=coeffs["C2"],
            c3_f_per_v3=coeffs["C3"], c4_f_per_v4=coeffs["C4"],
            c5_f_per_v5=coeffs["C5"],
        )

    # ------------------------------------------------------------------ #
    # Impedance / pump-delivery helpers                                  #
    # ------------------------------------------------------------------ #
    def Z_branch(self, f_hz: float, *, load_ohm: complex | None = None) -> complex:
        """Series branch impedance [Ω] at frequency ``f``.

        ``R_series + jωL_pkg + 1/(jωC0)`` -- the small-signal impedance the
        realistic varactor branch presents looking in from the tap (the
        quantity the L_pkg probe tabulated). ``load_ohm``, if given,
        adds a series stub/termination impedance below the junction.
        """

        omega = 2.0 * math.pi * float(f_hz)
        if omega == 0.0:
            raise ValueError("Z_branch is singular at DC (f = 0).")
        z_c = 1.0 / (1j * omega * self.C0_f)
        z = float(self.R_series_ohm) + 1j * omega * float(self.L_pkg_h) + z_c
        if load_ohm is not None:
            z = z + complex(load_ohm)
        return z

    def pump_divider(self, f_pump_hz: float) -> complex:
        """Fraction of a branch-top pump that appears *across* the junction C.

        ``Z_C / (Z_C + jωL_pkg + R_series)`` at the pump frequency, with
        ``Z_C = 1/(jωC0)``. This is the L_pkg-shielding transfer function the
        diagnostic chain identified: at ``f_pump = 6.2 GHz`` with
        ``L_pkg = 1.7 nH`` the SMV1245 junction sees only ~6-8 % of the
        branch-top pump (the probe measured ~60 mV across the junction for a
        ~800 mV branch-top pump). Ideal mode (no parasitics) returns ``1``.
        """

        omega = 2.0 * math.pi * float(f_pump_hz)
        if omega == 0.0:
            raise ValueError("pump_divider is singular at DC (f = 0).")
        # No package -> the full branch-top pump reaches the junction. Return
        # exactly 1 (not z_c/z_c, which need not be bit-exact) so the ideal
        # reduction is exact (the Section-1f bit-for-bit contract).
        if float(self.L_pkg_h) == 0.0 and float(self.R_series_ohm) == 0.0:
            return complex(1.0, 0.0)
        z_c = 1.0 / (1j * omega * self.C0_f)
        z_pkg = float(self.R_series_ohm) + 1j * omega * float(self.L_pkg_h)
        return z_c / (z_c + z_pkg)

    def m_effective(self, f_pump_hz: float, m_branch: float) -> float:
        """Effective junction modulation depth from a branch-top depth.

        ``m_eff = m_branch * |pump_divider(f_pump)|``. A branch-top
        modulation depth ``m_branch`` (in the framework's
        ``C(t) = C0*(1 + m*cos)`` convention) is reduced to ``m_eff`` at the
        junction by the package divider -- depths transform by the same
        divider as voltages because ``depth = (dC/dV)*V_swing/C0`` and the
        ``dC/dV``, ``C0`` factors cancel. Ideal mode returns ``m_branch``
        (divider magnitude 1).
        """

        return float(m_branch) * abs(self.pump_divider(f_pump_hz))

    def effective_capacitance_coeffs(
        self,
        f_pump_hz: float,
        m_branch: float,
        *,
        phi: float = 0.0,
        n_harmonics: int | None = None,
    ) -> list[complex]:
        """One-sided capacitance Fourier spectrum ``[C_0, C_1, ...]`` [F].

        The capacitance waveform the pumped junction presents,
        ``C(t) = C(V_bias + V_C*cos(wm*t + phi)) + CP``, expanded as
        ``C(t) = sum_d C_d e^{i*d*(wm*t)}`` (``C_{-d} = conj(C_d)``). This is
        the input to
        :func:`tvtl.floquet.harmonic_abcd.tv_capacitor_harmonic_admittance_general`.

        Construction (small-signal, design-proxy):

        * the **fundamental** is pinned to the effective depth:
          ``C_1 = 0.5 * m_eff * C0 * e^{i*phi}`` with
          ``m_eff = m_effective(f_pump, m_branch)`` -- so the package
          divider's pump shielding sets the fundamental modulation;
        * **higher bands** come from the bias-expanded C-V Taylor curvature:
          the junction AC swing is ``V_C = m_eff * C0 / (dC/dV)`` and band
          ``d`` collects ``sum_k C_taylor[k] * V_C^k * binom(k, (k-d)//2) /
          2^k`` over the curvature terms ``k``, then carries the pump phase
          ``e^{i*d*phi}``;
        * the **DC band** ``C_0`` is the static ``C0`` (junction(V_bias) +
          CP, with any static-C override) -- the curvature's second-order DC
          shift is absorbed into the operating-point ``C0``.

        For a linear C-V (``dC/dV = 0`` / ideal mode) only ``C_0`` and ``C_1``
        are non-zero and the result is exactly the single-tone cosine
        ``[C0, 0.5*m_eff*C0*e^{i*phi}]`` -- the form the ideal-C leg uses, so
        the realistic leg reduces to it bit-for-bit.

        Parameters
        ----------
        f_pump_hz : float
            Pump frequency [Hz] (sets the package divider).
        m_branch : float
            Branch-top modulation depth (framework ``C(t)=C0(1+m cos)``
            convention).
        phi : float, optional
            Pump phase [rad]. Default 0.
        n_harmonics : int | None, optional
            Number of AC bands to return (``len`` is ``n_harmonics + 1``).
            ``None`` -> 1 in ideal/linear mode, else ``min(taylor_order, 5)``.
        """

        c0 = self.C0_f
        m_eff = self.m_effective(f_pump_hz, m_branch)
        dcdv = self.dC_dV_f_per_v
        linear = self.ideal_linear or dcdv == 0.0
        if n_harmonics is None:
            n_harmonics = 1 if linear else min(int(self.taylor_order), 5)
        if n_harmonics < 1:
            raise ValueError("n_harmonics must be >= 1 (fundamental needed).")

        coeffs: list[complex] = [complex(c0)] + [0j] * n_harmonics
        # Fundamental: pinned to the effective depth (exact in linear mode).
        coeffs[1] = 0.5 * m_eff * c0 * cmath.exp(1j * float(phi))
        if linear:
            return coeffs

        # Higher bands from C-V curvature. Junction AC swing V_C such that the
        # linear-term fundamental equals 0.5*m_eff*c0 (so band 1 already
        # carries the k=1 contribution; we ADD the k>=3 corrections to it).
        v_c = m_eff * c0 / dcdv
        # Bias-expanded junction capacitance Taylor coeffs C_taylor[k]
        # (k = 1..taylor_order); the static term is handled by C_0 above.
        cap = smv1245_skyworks_spice_polynomial(
            cj0_f=float(self.CJ0_f), vj_v=float(self.VJ_v),
            m_grade=float(self.M), order=int(self.taylor_order),
            v_bias_v=float(self.V_bias_v),
        ).coefficients
        c_taylor = [
            cap["C0"], cap["C1"], cap["C2"], cap["C3"], cap["C4"], cap["C5"]
        ]
        max_k = min(int(self.taylor_order), 5)
        for d in range(1, n_harmonics + 1):
            acc = 0.0
            for k in range(max(d, 1), max_k + 1):
                if (k - d) % 2 != 0:
                    continue
                acc += c_taylor[k] * (v_c ** k) * math.comb(k, (k - d) // 2) / (2.0 ** k)
            coeffs[d] = acc * cmath.exp(1j * float(d) * float(phi))
        return coeffs

    def control_port_capacitance_coeffs(
        self, *, n_terms: int | None = None
    ) -> list[float]:
        """Capacitance Taylor coefficients ``[a0, a1, ..., aK]`` (F / V^k).

        ``a0 = C0`` (the static capacitance at the bias point, **incl. CP**;
        the CP fold) and ``a_k = (1/k!) d^k C_junction/dv^k`` at ``V_bias`` for
        ``k >= 1`` (CP is voltage-independent, so it touches only ``a0``).
        These are the coefficients of the **control-port realistic SDD** body
        ``Q1 = (sum_k a_k * _v2^k) * _v1`` -- i.e. the time-varying capacitance
        ``C(t) = C(V_bias + v2(t))`` driven by the control-port pump ``v2``.
        Distinct from :meth:`charge_polynomial` (the *charge* ``Q(v)`` for the
        self-pumped charge SDD); this is the *capacitance* expansion the
        control-port SDD multiplies by the signal voltage.
        """

        if self.ideal_linear:
            raise ValueError(
                "control_port_capacitance_coeffs is for realistic (curved) "
                "varactors; an ideal constant-C device has no pump response "
                "(use the Level10IdealC control-port path instead)."
            )
        max_k = min(int(self.taylor_order), 5)
        if n_terms is None:
            n_terms = max_k
        cap = smv1245_skyworks_spice_polynomial(
            cj0_f=float(self.CJ0_f), vj_v=float(self.VJ_v),
            m_grade=float(self.M), order=int(self.taylor_order),
            v_bias_v=float(self.V_bias_v),
        ).coefficients
        c_taylor = [cap[f"C{k}"] for k in range(6)]
        a = [self.C0_f]  # a0 = static C incl CP
        for k in range(1, n_terms + 1):
            a.append(float(c_taylor[k]) if k <= max_k else 0.0)
        return a

    def capacitance_coeffs_from_junction_vpump(
        self,
        v_pump: float,
        *,
        phi: float = 0.0,
        n_harmonics: int | None = None,
    ) -> list[complex]:
        """One-sided spectrum ``[C_0, C_1, ...]`` of ``C(V_bias + v_pump*cos)``.

        The *control-port* counterpart of :meth:`effective_capacitance_coeffs`:
        the junction sees the pump voltage ``v_pump`` **directly** (no package
        divider -- the control port imposes the pump, bypassing ``L_pkg`` for
        pump delivery while ``L_pkg``/``R_series`` remain in the signal path).
        Every band is the **natural** Taylor content (no depth-pinning), so it
        matches the control-port SDD body
        ``(sum_k a_k v2^k) v1`` with ``v2 = v_pump*cos`` term-for-term:

            C_d = e^{i d phi} * sum_{k >= d, (k-d) even} a_k * v_pump^k *
                  binom(k, (k-d)//2) / 2^k

        with ``a_k`` from :meth:`control_port_capacitance_coeffs` (so the DC
        band ``C_0`` carries the curvature DC shift exactly as the SDD does).

        Parameters
        ----------
        v_pump : float
            Junction pump voltage amplitude [V] (the control-port ``Voc``).
        phi : float, optional
            Pump phase [rad]. Default 0.
        n_harmonics : int | None, optional
            Number of AC bands (``len`` is ``n_harmonics + 1``); ``None`` ->
            ``min(taylor_order, 5)``.
        """

        if self.ideal_linear:
            raise ValueError(
                "capacitance_coeffs_from_junction_vpump is for realistic "
                "(curved) varactors; an ideal constant-C device does not "
                "respond to a junction pump voltage."
            )
        max_k = min(int(self.taylor_order), 5)
        if n_harmonics is None:
            n_harmonics = max_k
        if n_harmonics < 1:
            raise ValueError("n_harmonics must be >= 1 (fundamental needed).")
        a = self.control_port_capacitance_coeffs(n_terms=max_k)
        vp = float(v_pump)
        coeffs: list[complex] = []
        for d in range(0, n_harmonics + 1):
            acc = 0.0
            for k in range(max(d, 0), max_k + 1):
                if (k - d) % 2 != 0:
                    continue
                acc += a[k] * (vp ** k) * math.comb(k, (k - d) // 2) / (2.0 ** k)
            coeffs.append(acc * cmath.exp(1j * float(d) * float(phi)))
        # C_0 must be real-positive for the admittance helper.
        coeffs[0] = complex(coeffs[0].real, 0.0)
        return coeffs


