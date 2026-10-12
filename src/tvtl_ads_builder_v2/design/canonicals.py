"""Level-8 canonical circuits -- the design API exercised end to end.

These two canonicals are the first circuits built from a declarative
:class:`TVTLDesign` rather than hand-assembled element by element. Each
builds a design, calls :func:`to_device_spec`, and the resulting
:class:`DeviceSpec` flows through the existing v2 pipeline (emitters,
cross-check, content checks) unchanged.

They live HERE, inside ``design/``, rather than in
``parity/canonical_circuits.py``: a circuit built from a ``TVTLDesign``
is design-layer functionality, and the load-bearing dependency rule is
that nothing outside ``design/`` imports from ``design/``. Keeping the
Level-8 canonicals in ``design/`` honours that rule -- ``design/`` may
import the rest of the platform, never the reverse -- while still
exercising the whole pipeline. ``parity/canonical_circuits.py`` is
deliberately left untouched.

Both canonicals share one cell topology, differing only in ``n_cells``;
this is what verifies the design API scales without a hardcoded cell
count. The cell mirrors the Level-7 unit cell: a through-line MLIN, an
MTEE tapping one shunt branch, and the branch a series R - L - C chain
terminating in an SDD2P varactor to ground. Every value is a generic
round-number test value -- nothing paper-specific.
"""

from __future__ import annotations

import dataclasses

from ..device_spec import DeviceSpec, HBTwoToneSpec
from .elements import (
    AnalysisSpec,
    BranchSpec,
    CellSpec,
    ElementSpec,
    MSUBSpec,
    MicrostripSpec,
    SParamSpec,
    SourceSpec,
    TVTLDesign,
)
from .expand import to_device_spec

# The Level-8 parity-gate label, parallel to LEVEL_1_LABEL .. LEVEL_7_LABEL.
LEVEL_8_LABEL = "parity_level_8"


def _cascade_design(n_cells: int) -> TVTLDesign:
    """Build the Level-8 TVTL-cascade design for ``n_cells`` unit cells.

    One unit-cell template, replicated ``n_cells`` times by the
    expander. The cell is a through-line MLIN, an MTEE, and a shunt
    branch R - L - C - SDD2P-varactor to ground. The branch R / L / C
    are named references into the VAR block; the source and HB
    frequencies and powers likewise -- so the named-reference machinery
    is exercised across every cell. Generic round-number test values.
    """

    cell = CellSpec(
        through_line=MicrostripSpec(width=4.0e-3, length=3.0e-3),
        branches=(
            BranchSpec(
                elements=(
                    ElementSpec(type="R", params={"r_ohm": "R"}),
                    ElementSpec(type="L", params={"l_h": "L"}),
                    ElementSpec(type="C", params={"c_f": "C"}),
                    ElementSpec(
                        type="SDD2P_varactor",
                        params={
                            "q_poly_reference": "smv1245_default_polynomial"
                        },
                    ),
                ),
                termination="ground",
            ),
        ),
    )
    return TVTLDesign(
        n_cells=n_cells,
        cell=cell,
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
            # Signal source = Num 1, pump source = Num 2, output TermG
            # = Num 3 (set in the expander). Each P_1Tone source carries
            # its own ADS port number so hpeesofsim does not abort with
            # "Duplicate Port or Term number" when the deck runs
            # end-to-end.
            SourceSpec(
                name="SIGNAL_SRC",
                freq="f_signal",
                power_dBm="signal_power_dBm",
                num=1,
            ),
            SourceSpec(
                name="PUMP_SRC",
                freq="f_pump",
                power_dBm="pump_power_dBm",
                num=2,
            ),
        ),
        analysis=AnalysisSpec(
            s_param=SParamSpec(
                f_start_Hz=1.0e9,
                f_stop_Hz=10.0e9,
                n_points=91,
                port_impedance_ohm=50.0,
            ),
            hb_two_tone=HBTwoToneSpec(
                fundamental_freq_1="f_signal",
                fundamental_order_1=5,
                fundamental_freq_2="f_pump",
                fundamental_order_2=5,
                max_order=6,
            ),
        ),
        substrate=MSUBSpec(
            eps_r=3.4,
            h_m=1.5e-3,
            tan_delta=0.003,
            sigma_S_per_m=58.0e6,
        ),
    )


def make_tvtl_cascade(*, n_cells: int, label: str) -> DeviceSpec:
    """Expand an ``n_cells`` TVTL cascade into a labelled :class:`DeviceSpec`.

    The expander returns an unlabelled spec (``to_device_spec`` does not
    take a label); the canonical's parity label is set here.
    """

    return dataclasses.replace(
        to_device_spec(_cascade_design(n_cells)), label=label
    )


def level8_three_cell_tvtl_cascade() -> DeviceSpec:
    """The three-cell Level-8 canonical."""

    return make_tvtl_cascade(
        n_cells=3, label="level8_three_cell_tvtl_cascade"
    )


def level8_five_cell_tvtl_cascade() -> DeviceSpec:
    """The five-cell Level-8 canonical -- same builder, ``n_cells=5``.

    This is the scaling check: it shares the :func:`_cascade_design`
    template with the three-cell canonical and differs only in the cell
    count, so it confirms the design API expands an arbitrary N without
    a hardcoded cell count anywhere.
    """

    return make_tvtl_cascade(
        n_cells=5, label="level8_five_cell_tvtl_cascade"
    )


LEVEL_8_CANONICAL_CIRCUITS: tuple[DeviceSpec, ...] = (
    level8_three_cell_tvtl_cascade(),
    level8_five_cell_tvtl_cascade(),
)


