"""Emit the single-tone Harmonic Balance ``HB:`` analysis directive.

The HB controller is ADS's harmonic-balance analysis controller. For a
single-tone analysis it carries one fundamental frequency, one harmonic
order, and a maximum combined order.

The hpeesofsim netlist syntax is a single ``HB:`` line, e.g.::

    HB:HB1 Freq[1]=3.1 GHz Order[1]=5 MaxOrder=5 StatusLevel=2

``Freq`` and ``Order`` are 1-indexed array parameters; a single-tone
analysis populates index 1 only. This syntax matches ADS-generated
harmonic-balance decks and the OA ``ads_simulation/HarmonicBalance``
cell parameters ``Freq`` / ``Order`` / ``MaxOrder`` / ``StatusLevel``.

``StatusLevel`` is the solver's diagnostic-verbosity knob. This emitter
uses level 2. It is not a circuit-design
parameter, so it is not carried in :class:`HBSingleToneSpec`.
"""

from __future__ import annotations

from ...device_spec import HBSingleToneSpec, HBTwoToneSpec

# The HB controller instance name. Both backends use the same name so the netlist comparator and
# the content check pair the controller across the two decks.
HB_CONTROLLER_NAME = "HB1"

# Solver diagnostic-verbosity level. Fixed at 2; it
# is not a circuit parameter, so it is not part of HBSingleToneSpec.
HB_STATUS_LEVEL = 2


def emit_hb_single_tone(hb: HBSingleToneSpec) -> str:
    """Return the hpeesofsim ``HB:`` directive for a single-tone analysis.

    Frequency is emitted in GHz with ``{:.12g}`` formatting, matching
    the unit and precision ADS itself writes into a generated deck.
    ``Freq[1]`` and ``Order[1]`` carry the single tone; ``MaxOrder`` is
    the maximum combined order.
    """

    freq_GHz = float(hb.fundamental_freq) * 1.0e-9
    return (
        f"HB:{HB_CONTROLLER_NAME} "
        f"Freq[1]={freq_GHz:.12g} GHz "
        f"Order[1]={int(hb.fundamental_order)} "
        f"MaxOrder={int(hb.max_order)} "
        f"StatusLevel={HB_STATUS_LEVEL}"
    )


def emit_hb_two_tone(hb: HBTwoToneSpec) -> str:
    """Return the hpeesofsim ``HB:`` directive for a two-tone analysis.

    Both fundamentals are emitted in GHz (``{:.12g}``). The parameter
    order -- ``Freq[1] Freq[2] Order[1] Order[2] MaxOrder StatusLevel``
    -- matches ADS-generated two-tone decks, e.g.
    ``HB:HB1 Freq[1]=3.1 GHz Freq[2]=6.2 GHz Order[1]=5 Order[2]=5
    MaxOrder=6 StatusLevel=2``.
    """

    freq1_GHz = float(hb.fundamental_freq_1) * 1.0e-9
    freq2_GHz = float(hb.fundamental_freq_2) * 1.0e-9
    return (
        f"HB:{HB_CONTROLLER_NAME} "
        f"Freq[1]={freq1_GHz:.12g} GHz "
        f"Freq[2]={freq2_GHz:.12g} GHz "
        f"Order[1]={int(hb.fundamental_order_1)} "
        f"Order[2]={int(hb.fundamental_order_2)} "
        f"MaxOrder={int(hb.max_order)} "
        f"StatusLevel={HB_STATUS_LEVEL}"
    )


