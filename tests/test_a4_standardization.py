"""Termination-model standardization tests.

Additive coverage for the termination-model standardization:

* the standard palette is VIAGND (via) + MLEF (open-end) -- ``standard.yaml`` and
  :func:`standard_chain_models` build exactly those, and they emit the distributed
  ADS elements end-to-end;
* the simpler, non-distributed termination models (``lumped_L`` / ``rlc`` vias,
  ``ideal`` / ``mloc`` opens) are DEPRECATED -- they warn on instantiation but stay
  functional; the standard models do NOT warn;
* the cell's inline lumped-L / ideal-open DEFAULT path (no model passed) emits the
  byte-for-byte legacy ``LVIA`` and raises NO termination-deprecation warning, so the
  anchor stays byte-identical and the flat path is untouched.

These tests add coverage only; they assert no change to the existing flat-path
behaviour (the byte-identical anchor is the recovery gate).
"""

from __future__ import annotations

import warnings
from pathlib import Path

import numpy as np
import pytest

from tvtl.optimization.topology_generator import Termination, linear_families
from tvtl.varactors import RealisticVaractor
from tvtl_ads_builder_v2.models import (
    DEPRECATED_OPEN_END_KINDS,
    DEPRECATED_VIA_KINDS,
    IdealOpen,
    LumpedLVia,
    MlefOpen,
    MlocOpen,
    OPEN_END_REGISTRY,
    RlcVia,
    STANDARD_OPEN_END_KIND,
    STANDARD_VIA_KIND,
    VIA_REGISTRY,
    ViagndVia,
    build_chain_models,
    standard_chain_models,
)
from tvtl_ads_builder_v2.models.io import load_chain_config
from tvtl_ads_builder_v2.netlist import emit_flat_hb_netlist
from tvtl_ads_builder_v2.parity.self_pumped_design import optimized_self_pumped_spec

REPO_CONFIGS = str(Path(__file__).resolve().parents[1] / "configs" / "model_registry")
VAR_A = RealisticVaractor(
    CJ0_f=0.3e-12, VJ_v=3.5, M=1.7, CP_f=0.0, L_pkg_h=0.3e-9, R_series_ohm=0.5,
)
_SHORT = next(f for f in linear_families() if f.termination == Termination.SHORT and f.n_segments == 1)
_OPEN = next(f for f in linear_families() if f.termination == Termination.OPEN and f.n_segments == 1)
_PARAMS = np.full(_SHORT.n_params, 2.0e-3)
_PARAMS_OPEN = np.full(_OPEN.n_params, 2.0e-3)

_TERMINATION_DEPR = "non-distributed termination"


def _termination_deprecations(caught) -> list:
    return [
        w for w in caught
        if issubclass(w.category, DeprecationWarning)
        and _TERMINATION_DEPR in str(w.message)
    ]


# --------------------------------------------------------------------------- #
# Standard palette (VIAGND via + MLEF open-end)                                #
# --------------------------------------------------------------------------- #
def test_standard_yaml_builds_viagnd_mlef():
    cm = build_chain_models(load_chain_config(f"{REPO_CONFIGS}/standard.yaml"))
    assert isinstance(cm.via, ViagndVia) and cm.via.kind == "viagnd"
    assert isinstance(cm.open_end, MlefOpen) and cm.open_end.kind == "mlef"
    assert cm.name == "standard"


def test_standard_chain_models_helper_builds_viagnd_mlef():
    cm = standard_chain_models(VAR_A)
    assert isinstance(cm.via, ViagndVia) and cm.via.kind == STANDARD_VIA_KIND
    assert isinstance(cm.open_end, MlefOpen) and cm.open_end.kind == STANDARD_OPEN_END_KIND
    assert cm.varactor is VAR_A
    assert cm.name == "standard"


def test_standard_palette_emits_distributed_terminations():
    """End-to-end: the standard models emit VIAGND (SHORT) / MLEF (OPEN), no lumped via."""
    cm = standard_chain_models(VAR_A)
    s = optimized_self_pumped_spec(
        topology=_SHORT, params=_PARAMS, varactor=VAR_A, n_cells=2,
        via_model=cm.via, open_end_model=cm.open_end)
    # v1.0.1: an OPEN-stub HB deck needs a bias-return element (no_dc_return_for_varactor otherwise)
    o = optimized_self_pumped_spec(
        topology=_OPEN, params=_PARAMS_OPEN, varactor=VAR_A, n_cells=2,
        via_model=cm.via, open_end_model=cm.open_end, bias_return_ohm=1e5)
    s_deck, o_deck = emit_flat_hb_netlist(s), emit_flat_hb_netlist(o)
    assert "VIAGND:VIAGND_C1" in s_deck and "L:LVIA_C1" not in s_deck
    assert "MLEF:MLEF_C1" in o_deck


def test_standard_and_deprecated_kind_constants():
    assert STANDARD_VIA_KIND == "viagnd" and STANDARD_VIA_KIND in VIA_REGISTRY
    assert STANDARD_OPEN_END_KIND == "mlef" and STANDARD_OPEN_END_KIND in OPEN_END_REGISTRY
    assert set(DEPRECATED_VIA_KINDS) == {"lumped_L", "rlc"}
    assert set(DEPRECATED_OPEN_END_KINDS) == {"ideal", "mloc"}
    # the deprecated kinds stay REGISTERED (functional), just deprecated
    assert set(DEPRECATED_VIA_KINDS) <= set(VIA_REGISTRY)
    assert set(DEPRECATED_OPEN_END_KINDS) <= set(OPEN_END_REGISTRY)
    assert STANDARD_VIA_KIND not in DEPRECATED_VIA_KINDS
    assert STANDARD_OPEN_END_KIND not in DEPRECATED_OPEN_END_KINDS


# --------------------------------------------------------------------------- #
# Deprecation warnings (simpler models warn; standard models do not)          #
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("ctor", [
    pytest.param(lambda: LumpedLVia(), id="lumped_L"),
    pytest.param(lambda: RlcVia(Lvia_h=0.19e-9), id="rlc"),
    pytest.param(lambda: IdealOpen(), id="ideal"),
    pytest.param(lambda: MlocOpen(), id="mloc"),
])
def test_deprecated_termination_models_warn(ctor):
    with pytest.warns(DeprecationWarning, match=_TERMINATION_DEPR):
        ctor()


def test_standard_termination_models_do_not_warn():
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        ViagndVia()
        MlefOpen()
    assert _termination_deprecations(caught) == []


# --------------------------------------------------------------------------- #
# The default (inline) path stays byte-identical AND warning-free             #
# --------------------------------------------------------------------------- #
def test_default_inline_via_path_emits_lvia_without_deprecation():
    """No model passed -> the cell emits the inline lumped-L LVIA byte-for-byte and
    raises NO termination-deprecation warning (the deprecated class is never built),
    so the anchor / flat path is untouched."""
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        spec = optimized_self_pumped_spec(
            topology=_SHORT, params=_PARAMS, varactor=VAR_A, n_cells=2)
        deck = emit_flat_hb_netlist(spec)
    assert "L:LVIA_C1" in deck       # historical inline lumped-L default
    assert "VIAGND:" not in deck     # default is NOT switched to the standard via
    assert _termination_deprecations(caught) == []


