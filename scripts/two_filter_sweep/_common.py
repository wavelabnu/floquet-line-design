"""Shared plumbing for the two-filter feasibility sweep (Fig. 2c of the paper; ``task2.py``).

Everything here is read-only against the repo: it imports the public framework,
loads the committed element-accurate stub Touchstone tables, and rebuilds the
fabricated SMV1245 cell at a scaled stub geometry. ``T_REQ`` below is the 2026-09
screening threshold (``tvtl.design.feasibility.T_REQ_SCREEN_2026_09``) the
published sweep used; the adopted gate default is 0.17378.

Conventions (cross-checked against the repository code):

  T(f)      = |Z_C| / |Z_branch|,
              Z_branch = R_series + j w L_pkg + 1/(j w C0) + Z_stub(f)
              (identical to examples/real_parts/boundary/branch_delivery_runner.py)
  (A+D)/2   = central (n=0) 2x2 half-trace of the pump-off unit cell
              (identical to examples/figures/fig1d_dispersion.py)
"""
from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
for _p in (REPO / "src",
           REPO / "examples" / "real_parts",
           REPO / "examples" / "real_parts" / "boundary"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from tvtl.optimization.substrate import ROGERS_4003C            # noqa: E402
from tvtl.theory.dispersion import bloch_dispersion             # noqa: E402
from tvtl.varactor_palette import get_entry                     # noqa: E402
from tvtl_ads_builder_v2.analytical.realistic_c_gain import (    # noqa: E402
    realistic_pumped_unit_cell_abcd,
)
from tvtl_ads_builder_v2.device_spec import with_var_overrides   # noqa: E402
from tvtl_ads_builder_v2.models import MlefOpen, ViagndVia       # noqa: E402

from _stub_table import STUB_55GHz, STUB_80GHz, load_s1p, make_y_override   # noqa: E402
import real_part_designs as RPD                                  # noqa: E402
from real_part_designs import spec_smv1245, _SMV1245_MAIN_PERIOD_M  # noqa: E402

# --------------------------------------------------------------------------- #
# Frozen board constants (mirrored from examples/real_parts/real_part_designs.py)
# --------------------------------------------------------------------------- #
PERIOD_M = float(_SMV1245_MAIN_PERIOD_M)          # 6.11 mm
STUB_PARAMS_M = tuple(RPD._SMV1245_STUB_PARAMS_M)  # (w0,l0, w1,l1, w2,l2)
SEC_WIDTHS_M = tuple(RPD._SMV1245_SEC_WIDTHS_M)    # (1.93, 1.41, 1.98) mm
VIA_BOARD: ViagndVia = RPD._VIA_BOARD              # D = 1.6 mm board via
OPEN_END: MlefOpen = RPD._OPEN
SUBSTRATE_CFG = ROGERS_4003C

# Package / junction (branch_delivery_runner.py conventions)
R_SERIES_OHM = 2.0
L_PKG_H = 0.7e-9
C0_F_NOMINAL = 7.37e-12          # CJ0 6.9 pF + CP 0.47 pF

from tvtl.design.feasibility import T_REQ_SCREEN_2026_09 as T_REQ   # noqa: E402  (0.1438)


def _tvtl_substrate():
    """The tvtl.materials.Substrate the analytical leg uses for this board."""
    from tvtl_ads_builder_v2.analytical.realistic_c_gain import (
        _tvtl_substrate_from_spec,
    )
    return _tvtl_substrate_from_spec(spec_smv1245(n_cells=1))


# --------------------------------------------------------------------------- #
# Tabulated (ADS element-accurate) stub
# --------------------------------------------------------------------------- #
def tabulated_stub(which: str = "80"):
    """Return ``(f_hz, Y_stub, y_override_callable)`` for a committed .s1p."""
    path = STUB_80GHz if str(which) == "80" else STUB_55GHz
    f_hz, y = load_s1p(path)
    return f_hz, y, make_y_override(f_hz, y)


# --------------------------------------------------------------------------- #
# Cell half-trace
# --------------------------------------------------------------------------- #
class CellEvaluator:
    """Pump-off central-block half-trace ``(A+D)/2`` of the fabricated cell.

    ``stub_y_fn`` is injected verbatim as ``stub_y_fn_override``; everything else
    (through-line, MTEE, package, static C0) is the framework analytical model at
    exactly the fig1d settings.
    """

    def __init__(self, stub_y_fn, *, v_bias_v: float = 0.0,
                 c0_override_f: float | None = None):
        self._spec0 = spec_smv1245(n_cells=1, v_bias_v=v_bias_v)
        dev = get_entry("SMV1245-079LF").to_realistic_varactor(v_bias_v=v_bias_v)
        if c0_override_f is not None:
            # TASK-2 model probe: set the TOTAL static shunt capacitance the
            # branch presents, C0_f = c0_static_override_f + CP_f. Targets below
            # the package CP (0.47 pF) are unreachable while CP is held, so the
            # probe zeroes CP and puts the whole target on the static override.
            # At m_branch_pump = 0 the shunt element is exactly j*w*C0_f, so this
            # is a clean C0 probe and nothing else moves.
            import dataclasses as _dc
            dev = _dc.replace(dev, CP_f=0.0,
                              c0_static_override_f=float(c0_override_f))
            assert abs(dev.C0_f - float(c0_override_f)) < 1e-24
        self.dev = dev
        self.stub_y_fn = stub_y_fn
        self.period_m = PERIOD_M

    def half_trace(self, f_hz: float) -> complex:
        spec = with_var_overrides(self._spec0,
                                  {"f_signal": float(f_hz),
                                   "f_pump": 2.0 * float(f_hz)})
        a, b, c, d = realistic_pumped_unit_cell_abcd(
            spec, self.dev, m_branch_pump=0.0, truncation_N=1,
            use_mtee_parasitic=True, tee_model_variant=None,
            stub_y_fn_override=self.stub_y_fn,
        )
        return complex((a[1, 1] + d[1, 1]) / 2.0)

    def dispersion(self, f_hz: float):
        """``(betap_over_pi, alphap_Np_per_cell, half_trace)`` at ``f_hz``."""
        spec = with_var_overrides(self._spec0,
                                  {"f_signal": float(f_hz),
                                   "f_pump": 2.0 * float(f_hz)})
        a, b, c, d = realistic_pumped_unit_cell_abcd(
            spec, self.dev, m_branch_pump=0.0, truncation_N=1,
            use_mtee_parasitic=True, tee_model_variant=None,
            stub_y_fn_override=self.stub_y_fn,
        )
        blk = (a[1, 1], b[1, 1], c[1, 1], d[1, 1])
        gamma = complex(np.asarray(bloch_dispersion(blk, self.period_m)).ravel()[0])
        half = (blk[0] + blk[3]) / 2.0
        return (gamma.imag * self.period_m / math.pi,
                gamma.real * self.period_m,
                half)


# --------------------------------------------------------------------------- #
# Branch delivery
# --------------------------------------------------------------------------- #
def branch_T(f_hz: float, y_stub_fn, *, c0_f: float = C0_F_NOMINAL) -> float:
    """``T = |Z_C / (Z_pkg + Z_C + Z_stub)|`` at ``f_hz``."""
    w = 2.0 * math.pi * float(f_hz)
    z_c = 1.0 / (1j * w * float(c0_f))
    z_pkg = R_SERIES_OHM + 1j * w * L_PKG_H
    z_stub = 1.0 / complex(y_stub_fn(w))
    return abs(z_c / (z_pkg + z_c + z_stub))
