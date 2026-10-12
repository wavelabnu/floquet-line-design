"""Shared helper for the boundary replay runners: load a relocated corrected-stub
Touchstone ``.s1p`` (1-port S-parameters) and expose the stub admittance.

The ``.s1p`` tables under ``results/stub_tables/`` are the R6/R9 element-accurate
ADS ``S_Param`` sweeps, relocated verbatim from the original ADS EEsof
``.ds`` dataset (validated Y_stub(3.1 GHz) to 1e-15 vs the recorded value in
``smv1245_mstar_arc_v2.json``). Reading them needs no ADS.
"""
from __future__ import annotations

import math
from pathlib import Path

import numpy as np

RESULTS = Path(__file__).resolve().parents[1] / "results"
STUB_55GHz = RESULTS / "stub_tables" / "smv1245_stub_55GHz.s1p"
STUB_80GHz = RESULTS / "stub_tables" / "smv1245_stub_80GHz.s1p"


def load_s1p(path: str | Path):
    """Return ``(f_hz, Y_stub)`` from a ``# Hz S RI R 50`` 1-port Touchstone file.

    ``Y_stub = Y0 (1 - S11) / (1 + S11)`` with ``Y0 = 1/50`` — the passive stub
    admittance looking in from the varactor's bottom terminal (plane ``n_s0``).
    """
    f, sre, sim = [], [], []
    for ln in Path(path).read_text(encoding="ascii").splitlines():
        ln = ln.strip()
        if not ln or ln[0] in "!#":
            continue
        a, b, c = ln.split()
        f.append(float(a))
        sre.append(float(b))
        sim.append(float(c))
    f = np.asarray(f)
    s11 = np.asarray(sre) + 1j * np.asarray(sim)
    y = (1.0 / 50.0) * (1 - s11) / (1 + s11)
    return f, y


def make_y_override(f_hz, y_stub):
    """Build the ``omega -> Y_stub`` callable used as ``stub_y_fn_override``.

    Interpolates the tabulated admittance; conjugates for negative omega (so the
    override is Hermitian, as the analytical engine requires); hard-errors
    outside the table (a missing harmonic must fail loudly, not extrapolate).
    """
    fre, fim = np.asarray(y_stub).real, np.asarray(y_stub).imag
    flo, fhi = float(f_hz[0]), float(f_hz[-1])

    def y_override(omega):
        ff = abs(float(omega)) / (2 * math.pi)
        if ff < flo or ff > fhi:
            raise ValueError(
                f"stub table hard-error: {ff / 1e9:.3f} GHz outside "
                f"[{flo / 1e9:.3f}, {fhi / 1e9:.3f}] GHz"
            )
        y = complex(float(np.interp(ff, f_hz, fre)), float(np.interp(ff, f_hz, fim)))
        return y.conjugate() if omega < 0 else y

    return y_override
