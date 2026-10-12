"""Contract proof for the SDD2P polynomial content cross-check.

This is the load-bearing test file of the SDD2P work. The
content cross-check (:mod:`tvtl_ads_builder_v2.oa.sdd2p_content_check`)
exists to catch the exact bug that survived in v1: an SDD2P whose body
equation is not the charge polynomial the device spec asked for -- v1's
was the ADS default ``(_v1)/50.0``, a 50-ohm resistor.

The KCL-closure fix added the port-2 mirror: a correct
SDD2P body carries ``I[1,1]=+Q(_v1)`` and ``I[2,1]=-Q(_v1)``. The check
now verifies BOTH ports -- port 1 against the spec, port 2 as the exact
negation of port 1 -- and flags any non-whitelisted ``I[k,m]`` entry.

These tests prove the check fails loudly on every v1-style port-1 bug,
every port-2 bug shape, and cross-backend disagreement, and that it
honours its floating-point tolerance boundary. A negative test that
*passes* the check (when it should fail) is itself a failure --
``assert not result.match`` flips a broken check straight to a red test.
"""

from __future__ import annotations

import pytest

from tvtl.design.nonlinear_capacitor import smv1245_default_polynomial
from tvtl_ads_builder_v2.netlist.emit import emit_hpeesofsim_netlist
from tvtl_ads_builder_v2.oa.sdd2p_content_check import (
    check_sdd2p_content,
    format_content_report,
    parse_charge_polynomial_expression,
    parse_sdd_body_from_netlist,
)
from tvtl_ads_builder_v2.parity.canonical_circuits import (
    make_single_sdd2p_varactor,
)

# The Level-5 circuit and its correct hpeesofsim netlist.
SPEC = make_single_sdd2p_varactor(label="level5_single_sdd2p_varactor")
CORRECT_DECK = emit_hpeesofsim_netlist(SPEC)

# The correct SMV1245 port-1 charge polynomial and its exact negation
# (the port-2 KCL-closure mirror). q1 = C0 = 7e-12, q2 = C1/2 = -1e-12.
_P1 = "(7e-12)*_v1+(-1e-12)*_v1**2"
_P2 = "(-7e-12)*_v1+(1e-12)*_v1**2"


def _deck_with_sdd_body(body_equations: str) -> str:
    """Return the Level-5 deck with the SDD line's equations replaced.

    ``body_equations`` is the text after the four net tokens, e.g.
    ``"I[1,1]=0.02 I[2,1]=..."``. The four nets are preserved so only
    the body equations change -- this is exactly how a builder bug
    would manifest: correct topology, wrong physics.
    """

    out: list[str] = []
    for line in CORRECT_DECK.splitlines():
        if line.startswith("SDD:VARACTOR"):
            nets = " ".join(line.split()[1:5])
            out.append(f"SDD:VARACTOR {nets} {body_equations}")
        else:
            out.append(line)
    return "\n".join(out) + "\n"


# ---------------------------------------------------------------------------
# Positive tests -- a correct two-port body passes
# ---------------------------------------------------------------------------


def test_positive_correct_polynomial_passes() -> None:
    """A valid SDD2P spec produces matching netlists -> the check passes."""

    result = check_sdd2p_content(
        SPEC,
        hpeesofsim_netlist_text=CORRECT_DECK,
        ads_netlist_text=CORRECT_DECK,
    )
    assert result.match, result.failures
    assert result.checked == ["VARACTOR"]
    assert not result.failures


def test_positive_emits_port2_kcl_closure_mirror() -> None:
    """The correct deck carries I[1,1]=+Q and I[2,1]=-Q (exact mirror)."""

    assert f"I[1,1]={_P1}" in CORRECT_DECK
    assert f"I[2,1]={_P2}" in CORRECT_DECK
    assert "I[2,0]" not in CORRECT_DECK


def test_positive_passes_with_ads_style_auto_named_nets() -> None:
    """The check reads I[k,1], not nets -- ADS auto-net names still pass."""

    ads = CORRECT_DECK.replace(
        "n_in 0 n_return n_return", "N__0 0 N__2 N__2"
    )
    result = check_sdd2p_content(
        SPEC, hpeesofsim_netlist_text=CORRECT_DECK, ads_netlist_text=ads
    )
    assert result.match, result.failures


def test_positive_emitted_polynomial_is_the_smv1245_charge_poly() -> None:
    """The emitted port-1 body carries the actual SMV1245 charge poly."""

    expected = smv1245_default_polynomial().charge_polynomial_coefficients
    body = parse_sdd_body_from_netlist(CORRECT_DECK, "VARACTOR")
    assert body is not None and body.ok, body
    # q1 = C0 = 7 pF, q2 = C1/2 = -1 pF/V (the SMV1245 default).
    assert body.coefficients[1] == pytest.approx(expected[1])
    assert body.coefficients[2] == pytest.approx(expected[2])


# ---------------------------------------------------------------------------
# Negative tests -- v1-style port-1 bugs MUST fail the check
# ---------------------------------------------------------------------------


def test_negative_v1_bug_constant_placeholder_fails() -> None:
    """A bare-constant port-1 body (a fixed-value placeholder) fails."""

    deck = _deck_with_sdd_body(f"I[1,1]=0.02 I[2,1]={_P2}")
    result = check_sdd2p_content(
        SPEC, hpeesofsim_netlist_text=deck, ads_netlist_text=deck
    )
    assert not result.match
    assert any(
        "constant body" in f.lower() and "placeholder" in f.lower()
        for f in result.failures
    ), result.failures


def test_negative_v1_bug_literal_50ohm_default_fails() -> None:
    """v1's literal bug -- the ADS default (_v1)/50.0, a 50-ohm resistor."""

    deck = _deck_with_sdd_body(f"I[1,1]=(_v1)/50.0 I[2,1]={_P2}")
    result = check_sdd2p_content(
        SPEC, hpeesofsim_netlist_text=deck, ads_netlist_text=deck
    )
    assert not result.match
    assert any(
        "passive-element" in f.lower() for f in result.failures
    ), result.failures


def test_negative_v1_bug_weight0_current_source_no_charge_eq_fails() -> None:
    """v1's other shape: only weight-0 I[1,0]/I[2,0], no charge equations."""

    deck = _deck_with_sdd_body("I[1,0]=(_v1)/50.0 I[2,0]=(_v2)/50.0")
    result = check_sdd2p_content(
        SPEC, hpeesofsim_netlist_text=deck, ads_netlist_text=deck
    )
    assert not result.match
    assert any(
        "empty/default f-equation" in f.lower() for f in result.failures
    ), result.failures


def test_negative_missing_sdd_line_fails() -> None:
    """A netlist with the SDD2P element dropped entirely fails."""

    deck = (
        "\n".join(
            line
            for line in CORRECT_DECK.splitlines()
            if not line.startswith("SDD:VARACTOR")
        )
        + "\n"
    )
    result = check_sdd2p_content(
        SPEC, hpeesofsim_netlist_text=deck, ads_netlist_text=deck
    )
    assert not result.match
    assert any(
        "no sdd line" in f.lower() for f in result.failures
    ), result.failures


def test_negative_template_placeholder_variable_fails() -> None:
    """An unsubstituted template variable in the port-1 body fails."""

    deck = _deck_with_sdd_body(
        f"I[1,1]=(C0)*_v1+(C1)*_v1**2 I[2,1]={_P2}"
    )
    result = check_sdd2p_content(
        SPEC, hpeesofsim_netlist_text=deck, ads_netlist_text=deck
    )
    assert not result.match
    assert any(
        "template placeholder" in f.lower() for f in result.failures
    ), result.failures


def test_negative_coefficients_off_by_10x_fails() -> None:
    """A port-1 polynomial whose coefficients are all 10x too large fails.

    Port 2 here is the exact negation of the (wrong) port 1, so the
    only failure is the port-1-vs-spec disagreement.
    """

    deck = _deck_with_sdd_body(
        "I[1,1]=(7e-11)*_v1+(-1e-11)*_v1**2 "
        "I[2,1]=(-7e-11)*_v1+(1e-11)*_v1**2"
    )
    result = check_sdd2p_content(
        SPEC, hpeesofsim_netlist_text=deck, ads_netlist_text=deck
    )
    assert not result.match
    assert any(
        "port 1" in f and "disagrees with the device spec" in f
        for f in result.failures
    ), result.failures


def test_negative_single_coefficient_off_by_2x_fails() -> None:
    """Even a 2x error on one O(1e-12) coefficient is caught.

    This guards the tolerance choice: a 1e-12-scale absolute tolerance
    would have let a 2x error on the picocoulomb-scale q2 read as
    "close". The relative gate (1e-9) must catch it.
    """

    deck = _deck_with_sdd_body(
        "I[1,1]=(7e-12)*_v1+(-2e-12)*_v1**2 "
        "I[2,1]=(-7e-12)*_v1+(2e-12)*_v1**2"
    )
    result = check_sdd2p_content(
        SPEC, hpeesofsim_netlist_text=deck, ads_netlist_text=deck
    )
    assert not result.match
    assert any(
        "port 1" in f and "disagrees with the device spec" in f
        and "v**2" in f
        for f in result.failures
    ), result.failures


# ---------------------------------------------------------------------------
# Negative tests -- port-2 bugs MUST fail the check
# ---------------------------------------------------------------------------


def test_negative_port2_missing_fails() -> None:
    """Port 2 has no I[2,1] entry -- the KCL closure is missing."""

    deck = _deck_with_sdd_body(f"I[1,1]={_P1}")
    result = check_sdd2p_content(
        SPEC, hpeesofsim_netlist_text=deck, ads_netlist_text=deck
    )
    assert not result.match
    assert any(
        "port-2 charge equation i[2,1] is missing" in f.lower()
        for f in result.failures
    ), result.failures


def test_negative_port2_algebraic_zero_fails() -> None:
    """Port 2 is the known bug pattern I[2,0]=0 (weight-0 open)."""

    deck = _deck_with_sdd_body(f"I[1,1]={_P1} I[2,0]=0")
    result = check_sdd2p_content(
        SPEC, hpeesofsim_netlist_text=deck, ads_netlist_text=deck
    )
    assert not result.match
    # the weight-0 I[2,0] is flagged as a non-whitelisted equation ...
    assert any(
        "i[2,0]" in f.lower() and "non-whitelisted" in f.lower()
        for f in result.failures
    ), result.failures
    # ... and the required I[2,1] mirror is reported missing.
    assert any(
        "port-2 charge equation i[2,1] is missing" in f.lower()
        for f in result.failures
    ), result.failures


def test_negative_port2_wrong_sign_fails() -> None:
    """Port 2 carries +Q(_v1) instead of -Q(_v1) -- the wrong sign."""

    deck = _deck_with_sdd_body(f"I[1,1]={_P1} I[2,1]={_P1}")
    result = check_sdd2p_content(
        SPEC, hpeesofsim_netlist_text=deck, ads_netlist_text=deck
    )
    assert not result.match
    assert any(
        "port-2 polynomial i[2,1] is not the negation of port-1" in f.lower()
        for f in result.failures
    ), result.failures


def test_negative_port2_wrong_magnitude_fails() -> None:
    """Port 2 has the right sign but coefficients off by 2x."""

    deck = _deck_with_sdd_body(
        f"I[1,1]={_P1} I[2,1]=(-14e-12)*_v1+(2e-12)*_v1**2"
    )
    result = check_sdd2p_content(
        SPEC, hpeesofsim_netlist_text=deck, ads_netlist_text=deck
    )
    assert not result.match
    assert any(
        "port-2 polynomial i[2,1] is not the negation of port-1" in f.lower()
        for f in result.failures
    ), result.failures


def test_negative_cross_backend_port2_disagrees_fails() -> None:
    """One backend has the correct port-2, the other a wrong port-2."""

    wrong = _deck_with_sdd_body(f"I[1,1]={_P1} I[2,1]={_P1}")  # +Q, wrong
    result = check_sdd2p_content(
        SPEC,
        hpeesofsim_netlist_text=CORRECT_DECK,  # correct port 2 (-Q)
        ads_netlist_text=wrong,                # wrong port 2 (+Q)
    )
    assert not result.match
    assert any(
        "backends emit different port-2 polynomials" in f
        for f in result.failures
    ), result.failures


# ---------------------------------------------------------------------------
# Negative test -- the two backends disagree on port 1
# ---------------------------------------------------------------------------


def test_negative_cross_backend_disagreement_fails() -> None:
    """One backend emits the correct port-1 polynomial, the other wrong."""

    wrong = _deck_with_sdd_body(
        "I[1,1]=(7e-11)*_v1+(-1e-11)*_v1**2 "
        "I[2,1]=(-7e-11)*_v1+(1e-11)*_v1**2"
    )
    result = check_sdd2p_content(
        SPEC,
        hpeesofsim_netlist_text=CORRECT_DECK,  # correct
        ads_netlist_text=wrong,                # wrong
    )
    assert not result.match
    # the wrong backend disagrees with the spec ...
    assert any(
        "OA/ADS" in f and "disagrees with the device spec" in f
        for f in result.failures
    ), result.failures
    # ... and the two backends disagree with each other.
    assert any(
        "backends emit different port-1 polynomials" in f
        for f in result.failures
    ), result.failures


# ---------------------------------------------------------------------------
# Negative test -- the floating-point tolerance boundary is honoured
# ---------------------------------------------------------------------------


def test_precision_tolerance_boundary_is_honoured() -> None:
    """A sub-tolerance perturbation passes; a super-tolerance one fails.

    The relative tolerance is 1e-9. A coefficient perturbed by 5e-10
    relative is within it (representation noise); one perturbed by
    5e-9 relative is outside it (a real error). Port 2 stays the exact
    negation of the spec so only the port-1 perturbation is exercised.
    """

    q1 = 7.0e-12

    within = q1 * (1.0 + 5.0e-10)
    deck_within = _deck_with_sdd_body(
        f"I[1,1]=({within:.16g})*_v1+(-1e-12)*_v1**2 I[2,1]={_P2}"
    )
    result_within = check_sdd2p_content(
        SPEC,
        hpeesofsim_netlist_text=deck_within,
        ads_netlist_text=deck_within,
    )
    assert result_within.match, result_within.failures

    outside = q1 * (1.0 + 5.0e-9)
    deck_outside = _deck_with_sdd_body(
        f"I[1,1]=({outside:.16g})*_v1+(-1e-12)*_v1**2 I[2,1]={_P2}"
    )
    result_outside = check_sdd2p_content(
        SPEC,
        hpeesofsim_netlist_text=deck_outside,
        ads_netlist_text=deck_outside,
    )
    assert not result_outside.match
    assert any(
        "disagrees with the device spec" in f
        for f in result_outside.failures
    ), result_outside.failures


# ---------------------------------------------------------------------------
# Expression-parser unit coverage
# ---------------------------------------------------------------------------


def test_parser_classifies_clean_polynomial() -> None:
    parsed = parse_charge_polynomial_expression(
        "(7e-12)*_v1+(-1e-12)*_v1**2"
    )
    assert parsed.ok
    assert parsed.coefficients == {1: 7e-12, 2: -1e-12}


def test_parser_classifies_negated_mirror_polynomial() -> None:
    """The port-2 mirror polynomial parses to the negated coefficients."""

    parsed = parse_charge_polynomial_expression(_P2)
    assert parsed.ok
    assert parsed.coefficients == {1: -7e-12, 2: 1e-12}


def test_parser_classifies_empty_body() -> None:
    parsed = parse_charge_polynomial_expression("")
    assert not parsed.ok
    assert "empty/default" in parsed.diagnostic.lower()


def test_parser_classifies_zero_polynomial_as_clean() -> None:
    """An all-zero charge polynomial renders as '0' and is clean."""

    parsed = parse_charge_polynomial_expression("0")
    assert parsed.ok
    assert parsed.coefficients == {}


def test_format_content_report_renders_failures() -> None:
    deck = _deck_with_sdd_body(f"I[1,1]=0.02 I[2,1]={_P2}")
    result = check_sdd2p_content(
        SPEC, hpeesofsim_netlist_text=deck, ads_netlist_text=deck
    )
    report = format_content_report(result)
    assert "MISMATCH" in report
    assert "VARACTOR" in report


