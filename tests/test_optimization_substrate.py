"""Unit tests for the optimization substrate config (base Python)."""

from __future__ import annotations

import pytest

from tvtl.optimization.substrate import ROGERS_4003C, SubstrateConfig


def test_default_is_rogers_4003c():
    s = ROGERS_4003C
    assert s.eps_r == pytest.approx(3.38)
    assert s.h_m == pytest.approx(1.54e-3)
    assert s.tan_delta == pytest.approx(0.0027)
    assert s.sigma_S_per_m == pytest.approx(5.8e7)
    assert s.t_m == pytest.approx(17.5e-6)
    assert s.name == "msub1"


def test_stub_mapping_has_required_keys():
    m = ROGERS_4003C.stub_substrate_mapping()
    # Y_stub_at_freq reads eps_r, h_m, t_m; the rest mirror the v2 schema.
    for key in ("eps_r", "h_m", "t_m", "tan_delta", "sigma_S_per_m"):
        assert key in m
    assert m["eps_r"] == pytest.approx(3.38)


def test_frozen_and_validated():
    with pytest.raises(ValueError):
        SubstrateConfig(eps_r=0.0)
    with pytest.raises(ValueError):
        SubstrateConfig(h_m=-1.0)
    with pytest.raises(ValueError):
        SubstrateConfig(sigma_S_per_m=0.0)
    # immutable
    with pytest.raises(Exception):
        ROGERS_4003C.eps_r = 4.0  # type: ignore[misc]


def test_substrate_is_configurable():
    """Q3 hook: a different substrate is a single object swap."""
    alt = SubstrateConfig(eps_r=10.2, h_m=0.635e-3, name="alumina")
    assert alt.eps_r == pytest.approx(10.2)
    assert alt.stub_substrate_mapping()["h_m"] == pytest.approx(0.635e-3)


