"""Tests for the design-layer schema and validator.

ADS-free: the design schema dataclasses and :func:`validate_design`
are pure Python. These tests exercise field construction, equality
(the schema "round-trip" -- two identical designs compare equal), and
every accept/reject branch of the validator, including named-reference
resolution.
"""

from __future__ import annotations

import dataclasses

import pytest

from tvtl_ads_builder_v2.design import (
    AnalysisSpec,
    BranchSpec,
    CellSpec,
    DesignError,
    ElementSpec,
    HBTwoToneSpec,
    MSUBSpec,
    MicrostripSpec,
    SParamSpec,
    SourceSpec,
    TVTLDesign,
    validate_design,
)


# --------------------------------------------------------------------------
# Builders
# --------------------------------------------------------------------------


def _branch() -> BranchSpec:
    """The Level-7-style R - L - C - SDD2P shunt branch to ground."""

    return BranchSpec(
        elements=(
            ElementSpec(type="R", params={"r_ohm": "R"}),
            ElementSpec(type="L", params={"l_h": "L"}),
            ElementSpec(type="C", params={"c_f": "C"}),
            ElementSpec(
                type="SDD2P_varactor",
                params={"q_poly_reference": "smv1245_default_polynomial"},
            ),
        ),
        termination="ground",
    )


def _cell() -> CellSpec:
    return CellSpec(
        through_line=MicrostripSpec(width=4.0e-3, length=3.0e-3),
        branches=(_branch(),),
    )


def _good_design(**overrides) -> TVTLDesign:
    """A valid two-source, two-tone-HB three-cell design."""

    base = dict(
        n_cells=3,
        cell=_cell(),
        parameters={
            "f_signal": 3.0e9,
            "f_pump": 6.0e9,
            "signal_power_dBm": -30.0,
            "pump_power_dBm": 5.0,
            "R": 2.0,
            "L": 2.0e-9,
            "C": 4.0e-12,
        },
        sources=(
            SourceSpec(
                name="SIGNAL_SRC", freq="f_signal", power_dBm="signal_power_dBm"
            ),
            SourceSpec(
                name="PUMP_SRC", freq="f_pump", power_dBm="pump_power_dBm"
            ),
        ),
        analysis=AnalysisSpec(
            s_param=SParamSpec(
                f_start_Hz=1.0e9, f_stop_Hz=10.0e9, n_points=91,
                port_impedance_ohm=50.0,
            ),
            hb_two_tone=HBTwoToneSpec(
                fundamental_freq_1="f_signal", fundamental_order_1=5,
                fundamental_freq_2="f_pump", fundamental_order_2=5,
                max_order=6,
            ),
        ),
        substrate=MSUBSpec(
            eps_r=3.4, h_m=1.5e-3, tan_delta=0.003, sigma_S_per_m=58.0e6
        ),
    )
    base.update(overrides)
    return TVTLDesign(**base)


# --------------------------------------------------------------------------
# Schema construction / round-trip
# --------------------------------------------------------------------------


def test_schema_fields_are_accessible() -> None:
    """Every design dataclass exposes the declared fields."""

    design = _good_design()
    assert design.n_cells == 3
    assert design.cell.through_line.width == 4.0e-3
    assert design.cell.branches[0].termination == "ground"
    assert design.cell.branches[0].elements[0].type == "R"
    assert design.sources[0].name == "SIGNAL_SRC"
    assert design.sources[0].num == 1  # default
    assert design.sources[0].z_ohm == 50.0  # default
    assert design.analysis.s_param.n_points == 91
    assert design.substrate.model == "hammerstad_kirschning"  # default


def test_schema_round_trip_equality() -> None:
    """Two designs built from identical data compare equal (frozen dataclasses)."""

    assert _good_design() == _good_design()
    assert _cell() == _cell()
    assert _branch() == _branch()


def test_schema_distinct_designs_compare_unequal() -> None:
    """Changing one field makes the design unequal -- equality is structural."""

    assert _good_design(n_cells=3) != _good_design(n_cells=5)


def test_branchspec_defaults_to_ground_termination() -> None:
    """BranchSpec.termination defaults to 'ground'."""

    assert BranchSpec(elements=(ElementSpec("R", {"r_ohm": 2.0}),)).termination == (
        "ground"
    )


def test_cellspec_branches_default_empty() -> None:
    """A CellSpec with no branches is a pure transmission line."""

    cell = CellSpec(through_line=MicrostripSpec(width=4.0e-3, length=3.0e-3))
    assert cell.branches == ()


# --------------------------------------------------------------------------
# Validator -- accept
# --------------------------------------------------------------------------


def test_validator_accepts_a_good_design() -> None:
    """A well-formed design passes validation without raising."""

    validate_design(_good_design())


def test_validator_accepts_literal_parameters() -> None:
    """A design whose cell/source values are literals (no refs) validates."""

    cell = CellSpec(
        through_line=MicrostripSpec(width=4.0e-3, length=3.0e-3),
        branches=(
            BranchSpec(
                elements=(
                    ElementSpec(type="R", params={"r_ohm": 2.0}),
                    ElementSpec(type="C", params={"c_f": 4.0e-12}),
                ),
            ),
        ),
    )
    design = _good_design(
        cell=cell,
        sources=(SourceSpec(name="SRC", freq=3.0e9, power_dBm=-30.0),),
        analysis=AnalysisSpec(
            s_param=SParamSpec(f_start_Hz=1.0e9, f_stop_Hz=10.0e9, n_points=91)
        ),
    )
    validate_design(design)


def test_validator_accepts_one_cell() -> None:
    """n_cells == 1 is the minimum valid count."""

    validate_design(_good_design(n_cells=1))


def test_validator_accepts_floating_termination() -> None:
    """A branch with a 'floating' termination validates."""

    cell = CellSpec(
        through_line=MicrostripSpec(width=4.0e-3, length=3.0e-3),
        branches=(
            BranchSpec(
                elements=(ElementSpec(type="C", params={"c_f": "C"}),),
                termination="floating",
            ),
        ),
    )
    validate_design(_good_design(cell=cell))


# --------------------------------------------------------------------------
# Validator -- reject
# --------------------------------------------------------------------------


def test_validator_rejects_zero_cells() -> None:
    with pytest.raises(DesignError, match="n_cells"):
        validate_design(_good_design(n_cells=0))


def test_validator_rejects_negative_cells() -> None:
    with pytest.raises(DesignError, match="n_cells"):
        validate_design(_good_design(n_cells=-2))


def test_validator_rejects_unknown_parameter_reference_in_branch() -> None:
    """A branch element referencing an undeclared parameter is rejected."""

    cell = CellSpec(
        through_line=MicrostripSpec(width=4.0e-3, length=3.0e-3),
        branches=(
            BranchSpec(
                elements=(ElementSpec(type="R", params={"r_ohm": "R_missing"}),),
            ),
        ),
    )
    with pytest.raises(DesignError, match="R_missing"):
        validate_design(_good_design(cell=cell))


def test_validator_rejects_unknown_reference_in_through_line() -> None:
    """A through-line dimension referencing an undeclared parameter is rejected."""

    cell = CellSpec(
        through_line=MicrostripSpec(width="undeclared_w", length=3.0e-3),
        branches=(_branch(),),
    )
    with pytest.raises(DesignError, match="undeclared_w"):
        validate_design(_good_design(cell=cell))


def test_validator_rejects_unknown_reference_in_source() -> None:
    """A source frequency referencing an undeclared parameter is rejected."""

    bad_sources = (
        SourceSpec(name="SRC", freq="f_nonexistent", power_dBm=-30.0),
    )
    with pytest.raises(DesignError, match="f_nonexistent"):
        validate_design(_good_design(sources=bad_sources))


def test_validator_rejects_unknown_reference_in_hb_controller() -> None:
    """An HB controller frequency referencing an undeclared parameter fails."""

    analysis = AnalysisSpec(
        s_param=SParamSpec(f_start_Hz=1.0e9, f_stop_Hz=10.0e9, n_points=91),
        hb_two_tone=HBTwoToneSpec(
            fundamental_freq_1="f_signal", fundamental_order_1=5,
            fundamental_freq_2="f_absent", fundamental_order_2=5, max_order=6,
        ),
    )
    with pytest.raises(DesignError, match="f_absent"):
        validate_design(_good_design(analysis=analysis))


def test_validator_rejects_unknown_branch_element_type() -> None:
    """An element type outside the design vocabulary is rejected."""

    cell = CellSpec(
        through_line=MicrostripSpec(width=4.0e-3, length=3.0e-3),
        branches=(
            BranchSpec(elements=(ElementSpec(type="DIODE", params={}),)),
        ),
    )
    with pytest.raises(DesignError, match="DIODE"):
        validate_design(_good_design(cell=cell))


def test_validator_rejects_missing_s_param() -> None:
    """analysis.s_param is required -- the device spec needs an analysis block."""

    analysis = AnalysisSpec(
        s_param=None,
        hb_two_tone=HBTwoToneSpec(
            fundamental_freq_1="f_signal", fundamental_order_1=5,
            fundamental_freq_2="f_pump", fundamental_order_2=5, max_order=6,
        ),
    )
    with pytest.raises(DesignError, match="s_param"):
        validate_design(_good_design(analysis=analysis))


def test_validator_rejects_bad_termination() -> None:
    """A branch termination outside {ground, floating} is rejected."""

    cell = CellSpec(
        through_line=MicrostripSpec(width=4.0e-3, length=3.0e-3),
        branches=(
            BranchSpec(
                elements=(ElementSpec(type="R", params={"r_ohm": "R"}),),
                termination="antenna",
            ),
        ),
    )
    with pytest.raises(DesignError, match="termination"):
        validate_design(_good_design(cell=cell))


def test_validator_rejects_varactor_not_last_in_branch() -> None:
    """An SDD2P_varactor must be the branch's last (terminating) element."""

    cell = CellSpec(
        through_line=MicrostripSpec(width=4.0e-3, length=3.0e-3),
        branches=(
            BranchSpec(
                elements=(
                    ElementSpec(
                        type="SDD2P_varactor",
                        params={"q_poly_reference": "smv1245_default_polynomial"},
                    ),
                    ElementSpec(type="R", params={"r_ohm": "R"}),
                ),
            ),
        ),
    )
    with pytest.raises(DesignError, match="last element"):
        validate_design(_good_design(cell=cell))


def test_validator_rejects_varactor_without_polynomial_reference() -> None:
    """An SDD2P_varactor missing q_poly_reference is rejected."""

    cell = CellSpec(
        through_line=MicrostripSpec(width=4.0e-3, length=3.0e-3),
        branches=(
            BranchSpec(
                elements=(ElementSpec(type="SDD2P_varactor", params={}),),
            ),
        ),
    )
    with pytest.raises(DesignError, match="q_poly_reference"):
        validate_design(_good_design(cell=cell))


def test_validator_rejects_empty_branch() -> None:
    """A branch with no elements is rejected."""

    cell = CellSpec(
        through_line=MicrostripSpec(width=4.0e-3, length=3.0e-3),
        branches=(BranchSpec(elements=()),),
    )
    with pytest.raises(DesignError, match="at least one"):
        validate_design(_good_design(cell=cell))


def test_validator_rejects_non_numeric_parameter_value() -> None:
    """A VAR-block parameter whose value is not a finite number is rejected."""

    with pytest.raises(DesignError, match="parameters"):
        validate_design(
            _good_design(
                parameters={
                    "f_signal": 3.0e9, "f_pump": 6.0e9,
                    "signal_power_dBm": -30.0, "pump_power_dBm": 5.0,
                    "R": "not_a_number", "L": 2.0e-9, "C": 4.0e-12,
                }
            )
        )


def test_validator_rejects_descending_frequency_grid() -> None:
    """An S-parameter grid whose stop is below its start is rejected."""

    analysis = dataclasses.replace(
        _good_design().analysis,
        s_param=SParamSpec(f_start_Hz=10.0e9, f_stop_Hz=1.0e9, n_points=91),
    )
    with pytest.raises(DesignError, match="f_stop_Hz"):
        validate_design(_good_design(analysis=analysis))


