"""Parity tolerances per level.

These are constants, not configuration. If a level's parity gate fails
above its declared tolerance, the failure is reported and the level
stops. The tolerance is never loosened to push a check through.
"""

from __future__ import annotations


PARITY_TOLERANCE_DB_S21_LEVEL_1: float = 0.5
"""Max permitted ``|delta_S21|`` (in dB) across the frequency grid for
the Level-1 canonical circuits (TermG + MLIN + TermG). A passive,
50-Ohm-matched line should agree with the analytical model to far
better than half a dB; if any canonical circuit exceeds 0.5 dB, the
emitter, parser, or analytical primitive has a bug."""


PARITY_TOLERANCE_DB_S21_LEVEL_2: float = 0.5
"""Max permitted ``|delta_S21|`` (in dB) for the Level-2 canonical
circuits (cascaded MLINs through shared internal nodes). The same
0.5 dB envelope as Level 1: cascading passive lines must not
introduce more than a passive line's own loss, and the cascade
junction itself contributes nothing to first order."""


PARITY_TOLERANCE_DB_S21_LEVEL_3: float = 0.5
"""Max permitted ``|delta_S21|`` (in dB) for the Level-3 canonical
circuits (MTEE T-junction with a shunt branch stub). The same 0.5 dB
envelope as Levels 1-2. The dominant |S21| contribution is the
shunt-stub input admittance; the MTEE junction parasitics are a small
correction. Whether the TVTL ideal-junction model stays inside this
envelope against ADS's real MTEE element is the Level-3 discovery
decision -- the tolerance is never loosened to force a pass."""


PARITY_TOLERANCE_DB_S21_LEVEL_4: float = 0.5
"""Max permitted ``|delta_S21|`` (in dB) for the Level-4 canonical
circuits (lumped R / L / C / RLC_SERIES, alone and combined with the
distributed MLIN / MTEE elements). The same 0.5 dB envelope as
Levels 1-3.

A lumped R/L/C is unambiguous between the two simulators: the TVTL
analytical engine evaluates the closed-form series impedance
``R + jwL + 1/(jwC)`` and hpeesofsim sums the same three primitives.
The three pure-lumped / MLIN+lumped circuits clear 0.5 dB comfortably
(observed Path B 0.000-0.009 dB). ``mtee_with_rlc_stub`` adds an MTEE
and inherits the Level-3 ideal-junction MTEE model gap: its observed
Path B is ~0.997 dB, so it is recorded **parity-blocked** in
``inspection/metadata.py`` -- the same binary classification the
Level-3 MTEE circuits use. The tolerance itself is never loosened to
force a pass; a circuit that genuinely cannot meet it is marked
parity-blocked, not given a wider numeric class."""


PARITY_TOLERANCE_DB_S21_LEVEL_5: float = 0.5
"""Max permitted ``|delta_S21|`` (in dB) for the Level-5 canonical
circuit (the ``SDD2P`` symbolically-defined-device varactor).

The same 0.5 dB envelope as Levels 1-4; it is **never loosened**. Note
that the SDD2P element is validated only *structurally* -- the
OA-vs-hpeesofsim netlist cross-check (Path A) plus the polynomial
content cross-check. There is deliberately no TVTL analytical SDD2P
model yet (a nonlinear varactor has no closed-form linear S-parameter),
so the analytical Path B leg does not run for the Level-5 circuit; the
constant is declared for ladder uniformity and used should a later
phase add an analytical SDD2P model."""


PARITY_TOLERANCE_DB_S21_LEVEL_6: float = 0.5
"""Max permitted ``|delta_S21|`` (in dB) for the Level-6 canonical
circuit (the single-tone Harmonic Balance controller).

The same 0.5 dB envelope as Levels 1-5; it is **never loosened**.
The HB controller is validated *structurally* -- the
OA-vs-hpeesofsim netlist cross-check (Path A) plus the HB controller
content check. The Level-6 circuit's signal path is a plain 50-Ohm
series resistor, so the analytical Path B leg, when it runs, sees an
ordinary frequency-flat S-parameter response; the HB controller itself
has no analytical model."""


PARITY_TOLERANCE_DB_S21_LEVEL_7: float = 0.5
"""Max permitted ``|delta_S21|`` (in dB) for the Level-7 canonical
circuits (``P_1Tone`` power sources, and the integrated TVTL unit cell
driven by two-tone HB).

The same 0.5 dB envelope as Levels 1-6; it is **never loosened**.
The ``P_1Tone`` source and the integrated circuit are validated
*structurally* -- the OA-vs-hpeesofsim netlist cross-check (Path A),
the ``P_1Tone`` source content check, and (for the integrated cell)
the SDD2P / VAR / two-tone-HB content checks. The integrated cell
carries an ``SDD2P`` nonlinear varactor, which has no closed-form
linear S-parameter model, so its analytical Path B leg does not run
(``nan dB``) -- the constant is declared for ladder uniformity."""


PARITY_TOLERANCE_DB_S21_LEVEL_8: float = 0.5
"""Max permitted ``|delta_S21|`` (in dB) for the Level-8 canonical
circuits (the topology-driven N-cell TVTL cascades built by the
design API).

The same 0.5 dB envelope as Levels 1-7; it is **never loosened**.
A Level-8 canonical is a cascade of N identical Level-7-style unit
cells generated from a single :class:`TVTLDesign`. It is validated
*structurally* -- the OA-vs-hpeesofsim netlist cross-check (Path A)
plus the SDD2P / VAR / two-tone-HB / P_1Tone content checks. Every
cell carries an ``SDD2P`` nonlinear varactor, which has no closed-form
linear S-parameter model, so the analytical Path B leg does not run
(``nan dB``), exactly as for Level 5 and Level 7 -- the constant is
declared for ladder uniformity."""


PARITY_TOLERANCE_DB_S21_LEVEL_9: float = 0.5
"""Max permitted ``|delta_S21|`` (in dB) for the Level-9 canonical
circuits (the probed TVTL cascades with boundary AND varactor I_Probes,
two-tone HB).

The same 0.5 dB envelope as Levels 1-8; it is **never loosened**. A
Level-9 canonical is structurally a Level-7-style TVTL unit cell
replicated N times (N = 2 or 3), with I_Probes spliced at every cell
boundary and in series with every varactor. Two refinements over
Level-8 make the deck HB-runnable end-to-end: the two P_1Tone sources
carry distinct ``Num`` values (signal Num=1, pump Num=2) so ADS's
S_Param controller does not error with "Duplicate Port or Term number
for SIGNAL_SRC", and the output port is Num=3.

Path A (OA <-> hpeesofsim structural cross-check) plus the SDD2P /
VAR / two-tone-HB / P_1Tone content checks are the deliverable. Every
cell carries an SDD2P nonlinear varactor, so the analytical Path B
leg does not run (``nan dB``), exactly as for Level 5, Level 7 and
Level 8 -- the constant is declared for ladder uniformity."""


PARITY_TOLERANCE_DB_S21_LEVEL_10: float = 0.5
"""Max permitted ``|delta_S21|`` (in dB) for the Level-10 canonical
(the paper-fidelity 15-cell TVTL cascade on Rogers 4003C).

The same 0.5 dB envelope as Levels 1-9; it is **never loosened**.
Level-10 is the publication's reference device: 15 cells, Rogers 4003C
substrate, SMV1245-011LF Skyworks SPICE Taylor-expansion varactor,
the paper's exact microstrip dimensions (main w=4.27 mm, period
6.11 mm; three-segment shunt stub 1.93/1.41/1.98 mm wide and
6.83/5.25/1.04 mm long), two parallel vias to ground modelled via the
Bahl simple inductance formula.

Path A (OA <-> hpeesofsim structural cross-check) is the parity gate
exercised by this constant. The Path B Figure 4 analytical-vs-ADS
S-parameter reproduction is exercised separately by the
Figure 4 dry-run test with its own empirical-residual tolerance
(separate from this 0.5 dB constant)."""


