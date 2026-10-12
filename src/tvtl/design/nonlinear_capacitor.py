"""Generic nonlinear-capacitor models and ADS realization policy."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from typing import Mapping


class AdsAnalysisMode(StrEnum):
    """Analysis intent used to select an ADS realization."""

    SPARAM_STATIC = "sparam_static"
    HB_NONLINEAR = "hb_nonlinear"
    SDD_STATIC_SCAFFOLD = "sdd_static_scaffold"
    PRESENTATION_MANUAL = "presentation_manual"


class AdsRealizationMode(StrEnum):
    """ADS implementation strategy for a mathematical device model."""

    NATIVE_C0_RSER_STATIC = "native_c0_rser_static"
    NATIVE_SDD2P_QV = "native_sdd2p_qv"
    MANUAL_GOLDEN_CELL = "manual_golden_cell"


@dataclass(frozen=True)
class ChargePolynomialModel:
    """Charge polynomial using the paper supplement convention."""

    c0_f: float
    c1_f_per_v: float = 0.0
    c2_f_per_v2: float = 0.0
    c3_f_per_v3: float = 0.0
    c4_f_per_v4: float = 0.0
    c5_f_per_v5: float = 0.0

    @property
    def coefficients(self) -> dict[str, float]:
        return {
            "C0": float(self.c0_f),
            "C1": float(self.c1_f_per_v),
            "C2": float(self.c2_f_per_v2),
            "C3": float(self.c3_f_per_v3),
            "C4": float(self.c4_f_per_v4),
            "C5": float(self.c5_f_per_v5),
        }

    @property
    def expression(self) -> str:
        return "Q(v)=C0*v+C1*v**2/2+C2*v**3/3+C3*v**4/4+C4*v**5/5+C5*v**6/6"

    def charge_c(self, voltage_v: float) -> float:
        v = float(voltage_v)
        c = self.coefficients
        return (
            c["C0"] * v
            + c["C1"] * v**2 / 2.0
            + c["C2"] * v**3 / 3.0
            + c["C3"] * v**4 / 4.0
            + c["C4"] * v**5 / 5.0
            + c["C5"] * v**6 / 6.0
        )

    def capacitance_f(self, voltage_v: float) -> float:
        v = float(voltage_v)
        c = self.coefficients
        return c["C0"] + c["C1"] * v + c["C2"] * v**2 + c["C3"] * v**3 + c["C4"] * v**4 + c["C5"] * v**5

    @property
    def charge_polynomial_coefficients(self) -> tuple[float, ...]:
        """Charge polynomial ``Q(v)`` as a power-indexed coefficient array.

        ``coefficients`` exposes the *capacitance* coefficients
        ``C0..C5``; this exposes the *charge* polynomial, which is the
        integral of ``C(v)``::

            Q(v) = q[0] + q[1]*v + q[2]*v**2 + ... + q[6]*v**6

        with ``q[k]`` in coulombs / V**k. The element at index ``k`` is
        the coefficient of ``v**k``; ``q[0]`` is ``0.0`` because
        ``Q(0) = 0`` by construction, and ``q[k] = C{k-1} / k`` for
        ``k >= 1`` (the ``1/(n+1)`` divisor of the charge convention
        ``Q(v)=C0*v+C1*v**2/2+...``).

        This is the single source of truth for the charge polynomial in
        coefficient-array form -- the SDD2P device-spec parameter and
        both SDD2P emitters draw the array from here rather than
        re-deriving the ``1/k`` divisors.
        """

        c = self.coefficients
        return (
            0.0,
            c["C0"],
            c["C1"] / 2.0,
            c["C2"] / 3.0,
            c["C3"] / 4.0,
            c["C4"] / 5.0,
            c["C5"] / 6.0,
        )


@dataclass(frozen=True)
class StaticEquivalentModel:
    """Static small-signal equivalent used for passive S-parameter checks."""

    capacitance_f: float
    series_resistance_ohm: float | None = None


@dataclass(frozen=True)
class NonlinearCapacitorModel:
    """Mathematical nonlinear capacitance model independent of ADS."""

    name: str
    charge_model: ChargePolynomialModel
    series_resistance_ohm: float | None = None

    def static_equivalent(self) -> StaticEquivalentModel:
        return StaticEquivalentModel(
            capacitance_f=self.charge_model.c0_f,
            series_resistance_ohm=self.series_resistance_ohm,
        )

    def to_manifest(self) -> dict[str, object]:
        return {
            "name": self.name,
            "model_type": "charge_polynomial",
            "charge_expression": self.charge_model.expression,
            "coefficients": self.charge_model.coefficients,
            "static_equivalent": {
                "C": self.static_equivalent().capacitance_f,
                "Rser": self.static_equivalent().series_resistance_ohm,
            },
        }


@dataclass(frozen=True)
class AdsRealizationPolicy:
    """Selected ADS realization for a nonlinear-capacitor model."""

    analysis_mode: AdsAnalysisMode
    ads_realization_mode: AdsRealizationMode
    available: bool
    reason: str


def smv1245_default_polynomial() -> ChargePolynomialModel:
    """SMV1245-representative Q(v) polynomial used by HB validation.

    The coefficients are the linearised C-V fit around the -2 V default bias
    in the paper SI's Figure 2 envelope: ``C0 = 7 pF`` and
    ``dC/dV ~= -2 pF/V``. Higher-order coefficients are zero, so this is a
    documented small-signal linearisation rather than a hand-tuned fit.
    Centralising the polynomial here keeps a single source of truth for both
    the analytical Floquet path and the HB-validation workspaces.
    """

    return ChargePolynomialModel(
        c0_f=7.0e-12,
        c1_f_per_v=-2.0e-12,
        c2_f_per_v2=0.0,
        c3_f_per_v3=0.0,
        c4_f_per_v4=0.0,
        c5_f_per_v5=0.0,
    )


def smv1245_skyworks_spice_polynomial(
    *,
    cj0_f: float = 6.9e-12,
    vj_v: float = 3.5,
    m_grade: float = 1.7,
    order: int = 5,
    v_bias_v: float = 0.0,
) -> ChargePolynomialModel:
    """Taylor polynomial of the Skyworks SMV1245 SPICE C(v) around v = V_bias.

    Note on package parasitics
    --------------------------
    The Skyworks SMV1245-011LF SPICE card carries a parallel parasitic
    package capacitance ``CP = 0.47 pF`` (and a series ``RS = 2 Ω`` plus
    external ``L_pkg = 1.7 nH``). The Level-10 paper-geometry canonical
    represents ``RS`` and ``L_pkg`` as separate spec ``R`` and ``L``
    elements in series with the SDD2P, but **CP is folded into the
    SDD2P's q[1] coefficient** at spec-construction time rather than
    emitted as a literal parallel C: ``q[1] -> q[1] + CP``.

    The fold is mathematically exact, not approximate. The total
    displacement current into the (junction || CP) parallel pair is

        i = dQ_junction/dt + CP·dv/dt = d(Q_junction(v) + CP·v)/dt

    so augmenting the polynomial ``Q(v) -> Q(v) + CP·v`` produces the
    identical terminal current at every voltage. Higher-order ``Q(v)``
    coefficients (``q[2..]``) are unchanged because ``CP`` is
    voltage-independent. The fold is a deliberate workaround for the
    v2 OA layout engine's single-chain shunt-branch constraint: a
    literal parallel ``C`` across the SDD2P creates a fork (two
    elements at the same chain net) that triggers a grid-layout
    fallback and mis-routes nets. The fold sidesteps this without
    losing information.

    Future authors: do not reverse the fold thinking it is an
    approximation; it is exact. See
    ``tvtl_ads_builder_v2.parity.canonical_circuits._q_poly_from_varactor``
    for the call site and the equivalence proof.

    The Skyworks SMV1245-011LF datasheet supplies a junction-capacitance
    SPICE model

    .. math::
        C(v) = \\frac{CJ0}{(1 + v / VJ)^{M}}

    with the published parameters ``CJ0 = 6.9 pF``, ``VJ = 3.5 V``,
    ``M = 1.7`` for the bias polarity used by the paper (``v`` is the
    instantaneous voltage across the junction; ``v = 0`` corresponds to
    the locked Level-10 zero-reverse-bias operating point). The charge
    polynomial used by the v2 ``SDD2P`` element is the antiderivative of
    ``C(v)`` with the integration constant chosen so ``Q(0) = 0``. A
    Taylor expansion of ``C(v)`` around ``v = 0`` is

    .. math::
        C(v) = CJ0 \\sum_{k=0}^{\\infty} \\binom{-M}{k}
                \\left(\\frac{v}{VJ}\\right)^{k}
             = CJ0 + C_1 v + C_2 v^2 + \\dots

    with ``C_k = CJ0 / VJ^k * \\binom{-M}{k}``. Integrating term-by-term
    yields the charge polynomial ``Q(v) = \\sum_{k=0}^{\\infty} \\frac{C_k}{k+1} v^{k+1}``,
    which maps onto :class:`ChargePolynomialModel` via the convention
    ``C{k} = (k+1)*q[k+1]``.

    The default ``order`` of 5 covers the dominant intermod up through
    the 5th-order ``C(v)`` term — enough for the paper's locked
    +16 dBm pump amplitude (peak junction voltage on the order of 1-2 V,
    well below the ``VJ = 3.5 V`` knee where the expansion would lose
    fidelity). ``order = 6`` may be requested when a higher-fidelity fit
    is needed; the :class:`ChargePolynomialModel` schema caps the
    underlying capacitance coefficient list at ``C5``, so ``order > 5``
    is silently truncated to 5. The expansion is published-source-of-
    truth code: it derives every coefficient mechanically from the
    Skyworks SPICE parameters, so the polynomial cannot drift
    relative to the SPICE-card values that ADS would otherwise consume.

    Parameters
    ----------
    cj0_f : float
        Skyworks ``CJ0`` (zero-bias junction capacitance) in farads.
        Default 6.9 pF (the locked Level-10 paper value).
    vj_v : float
        Skyworks ``VJ`` (junction potential) in volts. Default 3.5 V.
    m_grade : float
        Skyworks ``M`` (grading coefficient). Default 1.7.
    order : int
        Taylor-expansion order of the capacitance series ``C(v)``;
        capped at 5 to fit :class:`ChargePolynomialModel`. Default 5.
    v_bias_v : float
        DC bias point around which the capacitance series is expanded.
        Default ``0.0`` -- the locked Level-10 zero-reverse-bias operating
        point, for which the result is **bit-for-bit identical** to the
        historical v=0 expansion (existing callers pass nothing and are
        unaffected). For a non-zero bias the expansion variable is the AC
        voltage ``u = v - V_bias`` and the series is built around the
        operating point: writing ``C(V_bias + u) = CJ0/((1 + V_bias/VJ) +
        u/VJ)^M = CJ0_eff/(1 + u/VJ_eff)^M`` with the *effective* zero-bias
        capacitance ``CJ0_eff = C(V_bias) = CJ0/(1 + V_bias/VJ)^M`` and the
        *effective* junction potential ``VJ_eff = VJ + V_bias``. In this
        convention reverse bias (``v > 0``) lowers the capacitance -- the
        same convention the DC-bias diagnostic used (re-expanding
        ``Q`` around the operating point and feeding it to the SDD2P is
        physically equivalent to biasing the device for the AC/HB analysis).
        Requires ``1 + V_bias/VJ > 0`` (``V_bias > -VJ``); a forward bias at
        or past the barrier has no real expansion.

    Returns
    -------
    ChargePolynomialModel
        A :class:`ChargePolynomialModel` carrying the Taylor coefficients
        of ``Q(v)`` mapped onto the
        :attr:`ChargePolynomialModel.coefficients` convention.

    Notes
    -----
    The capacitance-coefficient mapping is ``C{k} = (k+1)*q[k+1]`` so
    the model's :attr:`coefficients` dict carries the Taylor capacitance
    expansion ``C(v) = C0 + C1*v + C2*v^2 + ...`` directly. Caller code
    reading the polynomial via
    :attr:`ChargePolynomialModel.charge_polynomial_coefficients` will see
    the antiderivative form ``Q(v) = q1*v + q2*v^2/2 + q3*v^3/3 + ...``,
    matching the v2 ``SDD2P`` ``q_poly_coefficients`` parameter.
    """

    cj0 = float(cj0_f)
    vj = float(vj_v)
    m = float(m_grade)
    if cj0 <= 0.0:
        raise ValueError(f"cj0_f must be positive (got {cj0}).")
    if vj <= 0.0:
        raise ValueError(f"vj_v must be positive (got {vj}).")
    if not (m > 0.0):
        raise ValueError(f"m_grade must be positive (got {m}).")
    requested_order = int(order)
    if requested_order < 1:
        raise ValueError(f"order must be >= 1 (got {requested_order}).")
    # ChargePolynomialModel exposes C0..C5; orders above 5 are truncated
    # at the schema's published ceiling rather than rejected, so callers
    # can request a "natural" 6 without learning the model's cap.
    effective_order = min(requested_order, 5)

    # Re-centre the SPICE law on the bias point. C(V_bias + u) =
    # CJ0_eff / (1 + u/VJ_eff)^M with CJ0_eff = C(V_bias) and
    # VJ_eff = VJ + V_bias. At V_bias = 0 this is bit-for-bit the v=0
    # expansion (bias_factor = 1.0, cj0_eff = cj0, vj_eff = vj).
    v_bias = float(v_bias_v)
    bias_factor = 1.0 + v_bias / vj
    if bias_factor <= 0.0:
        raise ValueError(
            f"v_bias_v must satisfy 1 + v_bias/VJ > 0 (v_bias > -VJ); got "
            f"v_bias={v_bias}, VJ={vj} (bias_factor={bias_factor})."
        )
    cj0_eff = cj0 / bias_factor**m
    vj_eff = vj + v_bias

    # Taylor coefficients of C(V_bias + u) = CJ0_eff * (1 + u/VJ_eff)^(-M):
    #   C_k = CJ0_eff / VJ_eff^k * binom(-M, k)
    # where binom(-M, k) = product_{j=0..k-1} ((-M - j) / (j+1))
    # =  (-1)^k * M*(M+1)*(M+2)*...*(M+k-1) / k!  ; built iteratively
    # for numerical robustness across modest k.
    capacitance_coeffs: list[float] = [cj0_eff]  # C_0 = CJ0_eff = C(V_bias)
    coeff = cj0_eff
    for k in range(1, effective_order + 1):
        # binom(-M, k) = binom(-M, k-1) * (-M - (k-1)) / k
        coeff = coeff * (-(m + (k - 1))) / (float(k) * vj_eff)
        capacitance_coeffs.append(float(coeff))

    while len(capacitance_coeffs) < 6:
        capacitance_coeffs.append(0.0)

    return ChargePolynomialModel(
        c0_f=capacitance_coeffs[0],
        c1_f_per_v=capacitance_coeffs[1],
        c2_f_per_v2=capacitance_coeffs[2],
        c3_f_per_v3=capacitance_coeffs[3],
        c4_f_per_v4=capacitance_coeffs[4],
        c5_f_per_v5=capacitance_coeffs[5],
    )


def charge_polynomial_from_coefficients(
    coefficients: Mapping[str, float],
) -> ChargePolynomialModel:
    return ChargePolynomialModel(
        c0_f=float(coefficients.get("C0", 0.0)),
        c1_f_per_v=float(coefficients.get("C1", 0.0)),
        c2_f_per_v2=float(coefficients.get("C2", 0.0)),
        c3_f_per_v3=float(coefficients.get("C3", 0.0)),
        c4_f_per_v4=float(coefficients.get("C4", 0.0)),
        c5_f_per_v5=float(coefficients.get("C5", 0.0)),
    )


def select_ads_realization_policy(
    analysis_mode: AdsAnalysisMode | str,
    *,
    manual_golden_cell_validated: bool = False,
) -> AdsRealizationPolicy:
    mode = AdsAnalysisMode(str(analysis_mode))
    if mode == AdsAnalysisMode.SPARAM_STATIC:
        return AdsRealizationPolicy(
            analysis_mode=mode,
            ads_realization_mode=AdsRealizationMode.NATIVE_C0_RSER_STATIC,
            available=True,
            reason="Static S-parameter analysis uses the native ADS C=C0 plus optional Rser equivalent.",
        )
    if mode == AdsAnalysisMode.HB_NONLINEAR:
        return AdsRealizationPolicy(
            analysis_mode=mode,
            ads_realization_mode=AdsRealizationMode.NATIVE_SDD2P_QV,
            available=True,
            reason="Future nonlinear/HB validation uses native ADS SDD2P with the Q(_v1) convention.",
        )
    if mode == AdsAnalysisMode.SDD_STATIC_SCAFFOLD:
        return AdsRealizationPolicy(
            analysis_mode=mode,
            ads_realization_mode=AdsRealizationMode.NATIVE_SDD2P_QV,
            available=True,
            reason="Explicit SDD scaffold analysis requested native ADS SDD2P representation.",
        )
    if manual_golden_cell_validated:
        return AdsRealizationPolicy(
            analysis_mode=mode,
            ads_realization_mode=AdsRealizationMode.MANUAL_GOLDEN_CELL,
            available=True,
            reason="A manually ingested golden ADS component has been marked validated.",
        )
    return AdsRealizationPolicy(
        analysis_mode=mode,
        ads_realization_mode=AdsRealizationMode.MANUAL_GOLDEN_CELL,
        available=False,
        reason="Manual golden-cell realization is unavailable until a vetted ADS component is validated.",
    )


def static_ads_realization_report(
    model: NonlinearCapacitorModel,
    policy: AdsRealizationPolicy,
) -> dict[str, object]:
    """Summarize how a nonlinear capacitor is emitted for ADS review."""

    static = model.static_equivalent()
    if policy.ads_realization_mode == AdsRealizationMode.NATIVE_C0_RSER_STATIC:
        emitted_components = ["C"]
        if static.series_resistance_ohm is not None:
            emitted_components.append("R")
        sdd2p_emitted = False
    elif policy.ads_realization_mode == AdsRealizationMode.NATIVE_SDD2P_QV:
        emitted_components = ["SDD2P"]
        sdd2p_emitted = True
    else:
        emitted_components = []
        sdd2p_emitted = False

    return {
        "nonlinear_cap_model_kind": "charge_polynomial",
        "ads_realization_mode": policy.ads_realization_mode.value,
        "static_C_F": static.capacitance_f,
        "static_Rser_ohm": static.series_resistance_ohm,
        "emitted_ads_components": emitted_components,
        "sdd2p_emitted": sdd2p_emitted,
    }


