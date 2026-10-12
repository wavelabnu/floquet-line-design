from __future__ import annotations

from tvtl.design.nonlinear_capacitor import (
    AdsAnalysisMode,
    AdsRealizationMode,
    ChargePolynomialModel,
    NonlinearCapacitorModel,
    select_ads_realization_policy,
    static_ads_realization_report,
)


def test_nonlinear_capacitor_model_can_be_defined_generically() -> None:
    charge = ChargePolynomialModel(c0_f=0.35e-12, c1_f_per_v=1.0e-15)
    model = NonlinearCapacitorModel("candidate_cap", charge, series_resistance_ohm=1.0)
    assert model.static_equivalent().capacitance_f == 0.35e-12
    assert model.static_equivalent().series_resistance_ohm == 1.0
    assert charge.charge_c(0.5) > 0.0


def test_charge_polynomial_coefficients_are_recorded() -> None:
    charge = ChargePolynomialModel(
        c0_f=1.0,
        c1_f_per_v=2.0,
        c2_f_per_v2=3.0,
        c3_f_per_v3=4.0,
        c4_f_per_v4=5.0,
        c5_f_per_v5=6.0,
    )
    assert charge.coefficients == {"C0": 1.0, "C1": 2.0, "C2": 3.0, "C3": 4.0, "C4": 5.0, "C5": 6.0}
    assert "Q(v)=C0*v" in charge.expression


def test_static_analysis_maps_to_native_c0_rser() -> None:
    policy = select_ads_realization_policy(AdsAnalysisMode.SPARAM_STATIC)
    assert policy.ads_realization_mode == AdsRealizationMode.NATIVE_C0_RSER_STATIC
    assert policy.available is True


def test_nonlinear_capacitor_static_report_emits_native_c_and_r_not_sdd2p() -> None:
    model = NonlinearCapacitorModel(
        "static_cap",
        ChargePolynomialModel(c0_f=0.35e-12, c1_f_per_v=2.0e-15),
        series_resistance_ohm=1.2,
    )
    policy = select_ads_realization_policy(AdsAnalysisMode.SPARAM_STATIC)
    report = static_ads_realization_report(model, policy)
    assert report["nonlinear_cap_model_kind"] == "charge_polynomial"
    assert report["ads_realization_mode"] == "native_c0_rser_static"
    assert report["static_C_F"] == 0.35e-12
    assert report["static_Rser_ohm"] == 1.2
    assert report["emitted_ads_components"] == ["C", "R"]
    assert report["sdd2p_emitted"] is False


def test_explicit_sdd_scaffold_maps_to_native_sdd2p_qv() -> None:
    policy = select_ads_realization_policy(AdsAnalysisMode.SDD_STATIC_SCAFFOLD)
    assert policy.ads_realization_mode == AdsRealizationMode.NATIVE_SDD2P_QV
    assert policy.available is True


def test_sdd2p_static_report_is_explicit_opt_in_only() -> None:
    model = NonlinearCapacitorModel("sdd_cap", ChargePolynomialModel(c0_f=0.35e-12))
    static_policy = select_ads_realization_policy(AdsAnalysisMode.SPARAM_STATIC)
    sdd_policy = select_ads_realization_policy(AdsAnalysisMode.SDD_STATIC_SCAFFOLD)
    assert static_ads_realization_report(model, static_policy)["sdd2p_emitted"] is False
    assert static_ads_realization_report(model, sdd_policy)["sdd2p_emitted"] is True


def test_manual_golden_cell_is_unavailable_unless_validated() -> None:
    unavailable = select_ads_realization_policy(AdsAnalysisMode.PRESENTATION_MANUAL)
    assert unavailable.ads_realization_mode == AdsRealizationMode.MANUAL_GOLDEN_CELL
    assert unavailable.available is False

    available = select_ads_realization_policy(
        AdsAnalysisMode.PRESENTATION_MANUAL,
        manual_golden_cell_validated=True,
    )
    assert available.ads_realization_mode == AdsRealizationMode.MANUAL_GOLDEN_CELL
    assert available.available is True


