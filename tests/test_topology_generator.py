"""Unit tests for the generative topology generator (base Python).

Covers the enumeration arithmetic (20 = 12 linear + 8 COMBINED), the
physical-validity pruning of the via/termination grid, the parameter box, the
Sobol sampler, and -- the load-bearing checks -- that every generated linear
spec is validator-clean and that the topology stub admittance matches the
spec-extracted stub admittance (so A1 and the cross-validation cascade see the
same stub).
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from tvtl.optimization.substrate import ROGERS_4003C
from tvtl.optimization.topology_generator import (
    LENGTH_MAX_M,
    LENGTH_MIN_M,
    WIDTH_MAX_M,
    WIDTH_MIN_M,
    Termination,
    TopologyFamily,
    ViaConfig,
    combined_families,
    enumerate_families,
    is_valid,
    linear_families,
    sample_family,
    stub_segments,
    topology_y_stub_fn,
)
from tvtl.varactors import RealisticVaractor
from tvtl_ads_builder_v2.analytical.floquet_stub import (
    build_stub_y_fn_from_spec,
    extract_stub_segments,
)
from tvtl_ads_builder_v2.device_spec.validator import validate_device_spec
from tvtl_ads_builder_v2.parity.optimized_design import optimized_control_port_spec

_VAR = RealisticVaractor(
    CJ0_f=0.3e-12, VJ_v=3.5, M=1.7, CP_f=0.0, L_pkg_h=0.3e-9, R_series_ohm=0.5
)


def test_enumeration_counts():
    fams = enumerate_families()
    assert len(fams) == 20
    assert len(linear_families()) == 12
    assert len(combined_families()) == 8
    # 12 linear + 8 combined == 20 (not the loose "15/5" sketch)
    assert len(linear_families()) + len(combined_families()) == len(fams)


def test_all_families_unique_and_valid():
    fams = enumerate_families()
    keys = {f.key for f in fams}
    assert len(keys) == 20
    for f in fams:
        # every enumerated family is physically valid by construction
        assert f.n_segments in (1, 2, 3, 4)


@pytest.mark.parametrize(
    "via,term",
    [
        (ViaConfig.NONE, Termination.SHORT),
        (ViaConfig.NONE, Termination.COMBINED),
        (ViaConfig.SINGLE, Termination.OPEN),
        (ViaConfig.DOUBLE, Termination.OPEN),
    ],
)
def test_nonphysical_via_term_rejected(via, term):
    with pytest.raises(ValueError):
        TopologyFamily(n_segments=2, via=via, termination=term)


def test_n_params():
    assert TopologyFamily(3, ViaConfig.DOUBLE, Termination.SHORT).n_params == 6
    assert TopologyFamily(2, ViaConfig.NONE, Termination.OPEN).n_params == 4
    # COMBINED carries the extra open stublet (w, l)
    assert TopologyFamily(2, ViaConfig.SINGLE, Termination.COMBINED).n_params == 6


def test_sampler_in_box_and_budget():
    fam = TopologyFamily(3, ViaConfig.DOUBLE, Termination.SHORT)
    samples = list(sample_family(fam, 128, substrate=ROGERS_4003C, seed=0))
    assert 0 < len(samples) <= 128
    for p in samples:
        assert p.shape == (fam.n_params,)
        assert np.all(p[0::2] >= WIDTH_MIN_M) and np.all(p[0::2] <= WIDTH_MAX_M)
        assert np.all(p[1::2] >= LENGTH_MIN_M) and np.all(p[1::2] <= LENGTH_MAX_M)


def test_is_valid_rejects_out_of_box():
    fam = TopologyFamily(1, ViaConfig.SINGLE, Termination.SHORT)
    bad = np.array([WIDTH_MAX_M * 2, 5e-3])  # width out of box
    assert not is_valid(fam, bad, substrate=ROGERS_4003C)
    short = np.array([1e-3])  # wrong length
    assert not is_valid(fam, short, substrate=ROGERS_4003C)


def test_sampler_reproducible():
    fam = TopologyFamily(2, ViaConfig.SINGLE, Termination.SHORT)
    a = list(sample_family(fam, 32, substrate=ROGERS_4003C, seed=7))
    b = list(sample_family(fam, 32, substrate=ROGERS_4003C, seed=7))
    assert len(a) == len(b)
    for pa, pb in zip(a, b):
        assert np.allclose(pa, pb)


@pytest.mark.parametrize(
    "fam",
    [
        TopologyFamily(2, ViaConfig.NONE, Termination.OPEN),
        TopologyFamily(1, ViaConfig.SINGLE, Termination.SHORT),
        TopologyFamily(3, ViaConfig.DOUBLE, Termination.SHORT),
        TopologyFamily(2, ViaConfig.SINGLE, Termination.COMBINED),
    ],
)
def test_y_stub_finite_all_terminations(fam):
    p = next(sample_family(fam, 64, substrate=ROGERS_4003C, seed=1))
    yfn = topology_y_stub_fn(fam, p, ROGERS_4003C)
    for f_hz in (2.0e9, 3.1e9, 6.2e9):
        y = yfn(2.0 * math.pi * f_hz)
        assert math.isfinite(abs(y))


@pytest.mark.parametrize(
    "fam",
    [
        TopologyFamily(2, ViaConfig.NONE, Termination.OPEN),
        TopologyFamily(3, ViaConfig.DOUBLE, Termination.SHORT),
        TopologyFamily(1, ViaConfig.SINGLE, Termination.SHORT),
    ],
)
def test_generated_linear_spec_validator_clean(fam):
    p = next(sample_family(fam, 64, substrate=ROGERS_4003C, seed=2))
    spec = optimized_control_port_spec(
        topology=fam, params=p, varactor=_VAR, substrate=ROGERS_4003C, n_cells=4,
    )
    validate_device_spec(spec)  # raises DeviceSpecError if not clean


@pytest.mark.parametrize(
    "fam",
    [
        TopologyFamily(2, ViaConfig.NONE, Termination.OPEN),
        TopologyFamily(3, ViaConfig.DOUBLE, Termination.SHORT),
    ],
)
def test_topology_stub_matches_spec_extracted(fam):
    """A1's stub admittance == the cascade's spec-extracted stub admittance."""
    p = next(sample_family(fam, 64, substrate=ROGERS_4003C, seed=2))
    spec = optimized_control_port_spec(
        topology=fam, params=p, varactor=_VAR, substrate=ROGERS_4003C, n_cells=4,
    )
    y_topo = topology_y_stub_fn(fam, p, ROGERS_4003C)
    y_spec = build_stub_y_fn_from_spec(spec, 0)
    for f_hz in (2.5e9, 3.1e9, 5.0e9):
        w = 2.0 * math.pi * f_hz
        assert y_topo(w) == pytest.approx(y_spec(w), rel=1e-9, abs=1e-12)


def test_open_termination_extracted_as_open():
    """A generated OPEN spec's stub chain extracts termination='open'."""
    fam = TopologyFamily(3, ViaConfig.NONE, Termination.OPEN)
    p = next(sample_family(fam, 64, substrate=ROGERS_4003C, seed=2))
    spec = optimized_control_port_spec(
        topology=fam, params=p, varactor=_VAR, substrate=ROGERS_4003C, n_cells=2,
    )
    segments, _substrate, termination = extract_stub_segments(spec, 0)
    assert termination == "open"
    assert len(segments) == 3  # all three stub segments recovered


def test_short_termination_still_extracted_as_short():
    """The Decision-B edit leaves the short/inductive path unchanged."""
    fam = TopologyFamily(3, ViaConfig.DOUBLE, Termination.SHORT)
    p = next(sample_family(fam, 64, substrate=ROGERS_4003C, seed=2))
    spec = optimized_control_port_spec(
        topology=fam, params=p, varactor=_VAR, substrate=ROGERS_4003C, n_cells=2,
    )
    _segments, _substrate, termination = extract_stub_segments(spec, 0)
    assert termination == "short"


def test_combined_spec_generation_not_implemented():
    fam = TopologyFamily(2, ViaConfig.SINGLE, Termination.COMBINED)
    p = next(sample_family(fam, 64, substrate=ROGERS_4003C, seed=2))
    with pytest.raises(NotImplementedError):
        optimized_control_port_spec(
            topology=fam, params=p, varactor=_VAR, substrate=ROGERS_4003C, n_cells=2,
        )


def test_stub_segments_layout():
    fam = TopologyFamily(2, ViaConfig.SINGLE, Termination.COMBINED)
    p = np.array([1e-3, 2e-3, 1.5e-3, 3e-3, 0.9e-3, 4e-3])
    segs = stub_segments(fam, p)
    assert len(segs) == 2
    assert segs[0].width_m == pytest.approx(1e-3)
    assert segs[1].length_m == pytest.approx(3e-3)


