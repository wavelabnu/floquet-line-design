"""SMV2020-079LF carries no polynomial C(V) (v1.0.1): the transcribed one was the SMV1405 polynomial."""
import pytest

from tvtl import varactor_palette as VP


def test_smv2020_no_longer_returns_the_smv1405_polynomial():
    e = VP.get_entry('SMV2020-079LF')
    assert e.cv_poly_coeffs is None and e.cv_poly_valid_max_v is None
    assert VP._POLY_XLSX_SMV2020_REJECTED == pytest.approx(VP._POLY_REFERENCE_SMV1405, rel=0, abs=0)
    assert all(x.cv_poly_coeffs != VP._POLY_REFERENCE_SMV1405 for x in VP.PALETTE)
    assert e.c_poly_pf(1.0) is None and e.c0_poly_pf() is None and e.cv_poly_window_v() is None
    c, model = e.cv_pf_guarded(1.0)
    assert model == 'junction' and c == pytest.approx(e.c_junction_pf(1.0))
    assert 'cv_poly_rejected_copy_of_SMV1405_reference' in e.unknown_flags


def test_smv2020_spice_junction_parameters_unchanged():
    e = VP.get_entry('SMV2020-079LF')
    assert (e.cj0_pf, e.vj_v, e.m, e.cp_pf, e.r_series_ohm, e.l_package_nh) == (3.20, 4.4, 1.45, 0.0, 2.5, 0.7)
    v = e.to_realistic_varactor(v_bias_v=14.2)
    assert v.CJ0_f == pytest.approx(3.2e-12) and v.VJ_v == 4.4 and v.M == 1.45
