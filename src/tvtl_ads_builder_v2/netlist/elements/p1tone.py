"""Emit an ADS ``P_1Tone`` single-tone power-source line for hpeesofsim.

``P_1Tone`` is ADS's one-tone power source (library ``ads_sources``).
A two-tone (signal + pump) stimulus is two ``P_1Tone`` sources on one
input net, both ``Num=1``. The public builder emits that explicit
two-source arrangement because ADS harmonic balance sums independent
sources within one input port.

The hpeesofsim netlist line is, e.g.::

    P_1Tone:SIGNAL_SRC n_in 0 Num=1 Z=50 Ohm Freq=3.1 GHz P=polar(dbmtow(-30),0) Noise=no

The two nodes are the positive terminal (the driven ``port_node``) and the negative
terminal, which is the global ground ``0`` -- the same literal-``0``
treatment the ``TermG`` / ``Port:`` emitter uses for its implicit
ground terminal.

``P`` carries the source power inside ADS's ``polar(dbmtow(<dBm>),0)``
wrapper: ``dbmtow`` converts dBm to watts, ``polar(...,0)`` gives it
zero phase. This wrapper is reproduced verbatim from v1. ``Noise=no``
is emitted unconditionally -- it is a fixed solver knob, not a circuit
parameter, so it is not a :class:`P1ToneSource` field.

The format helpers are shared with the OA emitter so the hpeesofsim
flat deck and the OA-generated netlist carry a byte-identical
``P_1Tone`` parameter set -- the invariant the ``p1tone_content_check``
relies on.
"""

from __future__ import annotations

import math

from ...device_spec import P1ToneSource

# ADS ``P_1Tone`` ``Noise`` value. v1 fixes it at "no"; it is a solver
# knob, not a circuit parameter, so it is not part of P1ToneSource.
P1TONE_NOISE = "no"

# The literal hpeesofsim ground-reference token for the source's
# negative terminal (v1 always wires it to global ground).
_GROUND_TOKEN = "0"


def format_p1tone_z(z_ohm: float) -> str:
    """Return the ADS ``Z`` value string, e.g. ``'50 Ohm'``."""

    return f"{float(z_ohm):.12g} Ohm"


def format_p1tone_freq(freq_Hz: float) -> str:
    """Return the ADS ``Freq`` value string in GHz, e.g. ``'3.1 GHz'``.

    The device spec stores frequency in Hz; ADS and the hpeesofsim deck
    use GHz, so the stored value is scaled by 1e-9 -- the same unit and
    ``{:.12g}`` precision the HB-controller emitter uses.
    """

    return f"{float(freq_Hz) * 1.0e-9:.12g} GHz"


def format_p1tone_power(power_dBm: float, phase_rad: float = 0.0) -> str:
    """Return the ADS ``P`` value string ``polar(dbmtow(<dBm>),<phase_deg>)``.

    The power is given in dBm; ADS's ``dbmtow`` converts it to watts and
    ``polar(mag, angle)`` gives the complex amplitude, ``angle`` in **degrees**
    (the same convention the flat-deck ``Port:`` emitter uses, and what
    :class:`P1ToneSource`'s ``phase_rad`` field documents). ``phase_rad``
    defaults to ``0.0`` -> ``polar(dbmtow(<dBm>),0)`` for phase-0 sources.
    A non-zero ``phase_rad`` -- e.g. the
    degenerate-PSA optimal pump phase baked into an OA workspace -- emits the
    converted degrees, so the OA ``P_1Tone`` honours ``source.phase_rad`` instead
    of silently dropping it.
    """

    phase_deg = math.degrees(float(phase_rad))
    return f"polar(dbmtow({float(power_dBm):.12g}),{phase_deg:.12g})"


def emit_p1tone(source: P1ToneSource) -> str:
    """Return the hpeesofsim ``P_1Tone:`` line for one power source.

    Parameters
    ----------
    source : P1ToneSource
        A *resolved* source -- ``freq`` and ``power_dBm`` are literal
        numbers (the var-resolver substitutes any named reference
        before emission). A string value here is a bug upstream; it is
        coerced via ``float`` and would raise loudly.

    Returns
    -------
    str
        The single ``P_1Tone:`` netlist line. The positive terminal is
        ``source.port_node``; the negative terminal is the literal
        ground ``0``.
    """

    return (
        f"P_1Tone:{source.name} {source.port_node} {_GROUND_TOKEN} "
        f"Num={int(source.num)} "
        f"Z={format_p1tone_z(source.z_ohm)} "
        f"Freq={format_p1tone_freq(source.freq)} "
        f"P={format_p1tone_power(source.power_dBm, source.phase_rad)} "
        f"Noise={P1TONE_NOISE}"
    )


