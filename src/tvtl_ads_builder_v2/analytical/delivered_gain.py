"""Delivered, spatially varying capacitance-linear gain (ANALYTICAL only).

All voltages are peak phasors; power is |V|**2/(2 Z0).  Frequencies are Hz,
phases are radians, and reference impedances are real positive ohms.  This
module neither starts a simulator nor certifies gain.  A terminal ABCD alone
cannot identify an internal junction voltage: use a BranchCell callable, or
adapt an existing ABCD callable with its physical branch decomposition.
"""
from __future__ import annotations

from dataclasses import dataclass
from numbers import Integral
from typing import Callable, Sequence
from functools import lru_cache

import numpy as np


class TeeModelOutOfRange(ValueError):
    """A cell sample needs the MTEE reference-plane model above its validity edge.

    The message starts with the refusal reason ``tee_model_out_of_range``. Gate and
    optimiser paths turn it into that reason instead of a number.
    """

    reason = "tee_model_out_of_range"


def cell_functions_from_spec(*, spec, varactor, stub_y_fn=None, use_mtee=True,
                             modulation_convention="derivative", input_abcd_fn=None,
                             output_abcd_fn=None, tee_reference_frequency_hz=None,
                             tee_out_of_range="raise"):
    """Adapt a Level-10 spec and palette device to BranchCell callables.

    Explicit stub admittance overrides (angular frequency -> siemens) support
    mid-shunts, RlcVia and optimizer COMBINED terminations. Supply one callable
    or a list per cell. Without overrides, the established spec tracer is used.
    Matching must be supplied as positive-frequency ABCD callables; silently
    dropping matching elements from a spec is rejected. ``r5`` explicitly
    reproduces M/VJ without CP/bias correction; ``derivative`` uses the total
    capacitance's physical fractional derivative at the declared DC bias.
    ``tee_reference_frequency_hz`` freezes only the line-length corrections,
    matching the legacy harmonic core convention; None retains R5's per-frequency
    corrections. Tee matrices and shunt admittance always use the actual frequency.

    The MTEE reference-plane model fails its validity guard above a width-dependent
    edge (11.64 GHz for the SMV1245 cell). ``tee_out_of_range="raise"`` (default)
    raises :class:`TeeModelOutOfRange` for a sample above the edge. ``"reduced"``
    is the 1.0.0 behaviour, requested explicitly: identity tee blocks, no junction
    admittance, line lengths still shortened by the model's corrections. Results
    computed in that mode must be labelled as using the reduced tee model.
    """
    from .floquet_stub import build_stub_y_fn_from_spec
    from .level10_sparams import _line_abcd_lossy, _tvtl_substrate_from_spec
    from .mtee_blocks import tee_reference_plane_single

    if modulation_convention not in ("derivative", "r5"):
        raise ValueError("modulation_convention must be derivative or r5")
    if tee_out_of_range not in ("raise", "reduced"):
        raise ValueError("tee_out_of_range must be 'raise' or 'reduced'")
    if tee_reference_frequency_hz is not None:
        _positive(tee_reference_frequency_hz, "tee_reference_frequency_hz")
    els = {e.id: e for e in spec.elements}
    keys = sorted(int(e.id.split("_C")[-1]) for e in spec.elements if e.id.startswith("VARACTOR_C"))
    if not keys:
        raise ValueError("No varactor cells in spec")
    if any("MATCH" in name.upper() for name in els) and input_abcd_fn is None and output_abcd_fn is None:
        raise ValueError("Spec contains matching: provide explicit matching ABCD callables")
    if stub_y_fn is None:
        stubs = [build_stub_y_fn_from_spec(spec, k-1) for k in keys]
    elif callable(stub_y_fn):
        stubs = [stub_y_fn]*len(keys)
    else:
        stubs = list(stub_y_fn)
        if len(stubs) != len(keys):
            raise ValueError("One stub override is required per cell")
    sub = _tvtl_substrate_from_spec(spec)
    c0 = float(varactor.C0_f)
    potential = float(varactor.VJ_v)
    grading = float(varactor.M)
    if modulation_convention == "derivative":
        potential += float(varactor.V_bias_v)
        cj = float(varactor.CJ0_f)/(1+float(varactor.V_bias_v)/float(varactor.VJ_v))**float(varactor.M)
        grading *= cj/c0
        if varactor.ideal_linear:
            grading = 0.0
    functions = []
    for position, (k, stub) in enumerate(zip(keys, stubs)):
        left = els[f"LINE_LEFT_C{k}"].parameters
        right = els[f"LINE_RIGHT_C{k}"].parameters
        tee = els.get(f"TEE_C{k}")
        lp = float(els[f"LPKG_C{k}"].parameters["l_h"])
        rp = float(els[f"RPKG_C{k}"].parameters["r_ohm"])
        def factory(left, right, tee, lp, rp, stub, position, k):
            @lru_cache(maxsize=128)
            def sample(f):
                _positive(f, "frequency_hz")
                a = b = np.eye(2, dtype=complex)
                j, la, lb = 0j, 0., 0.
                if use_mtee and tee is not None:
                    a, b, j, la, lb, valid = tee_reference_plane_single(float(left['w_m']), float(tee.parameters['w3_m']), f, sub)
                    if not valid and tee_out_of_range == "raise":
                        raise TeeModelOutOfRange(
                            f"tee_model_out_of_range: cell {k} MTEE reference-plane model fails its validity "
                            f"guard at {f/1e9:.6g} GHz (main width {float(left['w_m'])*1e3:.4g} mm, branch width "
                            f"{float(tee.parameters['w3_m'])*1e3:.4g} mm); tee_out_of_range='reduced' evaluates "
                            "without the tee correction")
                    if not valid:
                        a = b = np.eye(2, dtype=complex)
                        j = 0j
                    if tee_reference_frequency_hz is not None:
                        _, _, _, la, lb, _ = tee_reference_plane_single(
                            float(left['w_m']), float(tee.parameters['w3_m']),
                            tee_reference_frequency_hz, sub)
                l = _line_abcd_lossy(float(left['w_m']), float(left['l_m'])-la, f, sub) @ a
                r = b @ _line_abcd_lossy(float(right['w_m']), float(right['l_m'])-lb, f, sub)
                if position == 0 and input_abcd_fn is not None:
                    l = np.asarray(input_abcd_fn(f), complex) @ l
                if position == len(keys)-1 and output_abcd_fn is not None:
                    r = r @ np.asarray(output_abcd_fn(f), complex)
                y = complex(stub(2*np.pi*f))
                if y == 0:
                    raise ValueError("Open/infinite branch impedance needs a finite limiting realization")
                return BranchCell(l, r, rp+2j*np.pi*f*lp+1/y, c0, grading, potential, j)
            return sample
        functions.append(factory(left, right, tee, lp, rp, stub, position, k))
    return functions


@dataclass(frozen=True)
class BranchCell:
    """One frequency sample: left two-port, series branch, right two-port.

    ``series_impedance_ohm`` excludes the junction capacitor but includes
    package, return/stub and their losses.  ``junction_capacitance_f`` includes
    any parallel CP.  ``M / Vj`` is the explicitly supplied fractional
    capacitance slope per volt (use a bias/CP-corrected effective M when
    desired; the R5 convention supplies its original M unchanged).
    """

    left: np.ndarray
    right: np.ndarray
    series_impedance_ohm: complex
    junction_capacitance_f: float
    M: float
    Vj: float
    shunt_admittance_s: complex = 0j

    def __post_init__(self):
        for name in ("left", "right"):
            a = np.asarray(getattr(self, name), dtype=complex)
            if a.shape != (2, 2) or not np.all(np.isfinite(a)):
                raise ValueError(f"{name} must be a finite 2x2 ABCD")
            object.__setattr__(self, name, a.copy())
        for name in ("junction_capacitance_f", "Vj"):
            if not np.isfinite(getattr(self, name)) or getattr(self, name) <= 0:
                raise ValueError(f"{name} must be finite and positive")
        if not np.isfinite(self.M) or self.M < 0:
            raise ValueError("M must be finite and nonnegative")
        if not np.isfinite(self.series_impedance_ohm) or not np.isfinite(self.shunt_admittance_s):
            raise ValueError("branch impedances/admittances must be finite")

    def abcd(self, *, frequency_hz: float) -> np.ndarray:
        """Return the passive terminal ABCD at the sample frequency."""
        _positive(frequency_hz, "frequency_hz")
        zc = 1 / (2j * np.pi * frequency_hz * self.junction_capacitance_f)
        y = 1 / (self.series_impedance_ohm + zc) + self.shunt_admittance_s
        return self.left @ np.array([[1, 0], [y, 1]], complex) @ self.right


def _positive(value, name):
    if not np.isfinite(value) or value <= 0:
        raise ValueError(f"{name} must be finite and positive")


def _functions(cell_fn, N):
    if isinstance(N, bool) or not isinstance(N, Integral) or N < 1:
        raise ValueError("N must be a positive integer")
    fns = [cell_fn] * N if callable(cell_fn) else list(cell_fn)
    if len(fns) != N or not all(callable(f) for f in fns):
        raise ValueError("Provide one cell callable or exactly N cell callables")
    return fns


def _sample(fn, f):
    sample = fn(float(f))
    if not isinstance(sample, BranchCell):
        raise TypeError("Internal pump delivery requires BranchCell metadata; terminal ABCD alone is insufficient")
    return sample


def _voltage(power_dBm, Z0):
    if power_dBm == -np.inf:
        return 0.0
    if not np.isfinite(power_dBm):
        raise ValueError("Power must be finite dBm, or -inf for pump off")
    v = np.sqrt(2 * Z0 * 10 ** ((power_dBm - 30) / 10))
    if not np.isfinite(v):
        raise ValueError("Power is outside the representable range")
    return float(v)


def _passive(samples, f, incident, Z0):
    load = 1 / Z0
    factors, loads = [], []
    for cell in samples[::-1]:
        a, b, c, d = cell.abcd(frequency_hz=f).ravel()
        loads.append(load)
        den = a + b * load
        factors.append(1 / den)
        load = (c + d * load) / den
    reflection = (1 - Z0 * load) / (1 + Z0 * load)
    v = 2 * incident / (1 + Z0 * load)
    ends, junctions = [], []
    for cell, factor, load in zip(samples, factors[::-1], loads[::-1]):
        v *= factor
        branch_v = (cell.right[0, 0] + cell.right[0, 1] * load) * v
        zc = 1 / (2j * np.pi * f * cell.junction_capacitance_f)
        junctions.append(branch_v * zc / (cell.series_impedance_ohm + zc))
        ends.append(v)
    return np.asarray(junctions), np.asarray(ends), reflection


def delivered_pump_profile(*, cell_fn, N, f_p, P_inc_dBm, Z0=50):
    """Return N complex junction peak phasors from a board incident pump.

    The callable is evaluated at positive f_p. Disordered lines supply N
    callables, each returning its own BranchCell. No uniform-field assumption.
    """
    _positive(f_p, "f_p")
    _positive(Z0, "Z0")
    samples = [_sample(f, f_p) for f in _functions(cell_fn, N)]
    return _passive(samples, f_p, _voltage(P_inc_dBm, Z0), Z0)[0]


def modulation_profile(*, V_j, M, Vj, m_cap=0.9):
    """Return (depth, phase radians, capped mask), broadcasting M and Vj."""
    v, slope, potential = np.broadcast_arrays(np.asarray(V_j, complex), np.asarray(M, float), np.asarray(Vj, float))
    if not np.all(np.isfinite(v)) or not np.all(np.isfinite(slope)) or np.any(slope < 0):
        raise ValueError("Junction phasors and nonnegative M must be finite")
    if not np.all(np.isfinite(potential)) or np.any(potential <= 0):
        raise ValueError("Vj must be finite and positive")
    if not np.isfinite(m_cap) or not 0 < m_cap < 1:
        raise ValueError("m_cap must lie strictly between 0 and 1")
    raw = slope / potential * np.abs(v)
    return np.minimum(raw, m_cap), np.angle(v), raw > m_cap


def harmonic_cell(*, cell_fn, f_s, indices, coefficients):
    """Build one harmonic ABCD from explicit C Fourier coefficients.

    ``indices`` are signed odd multiples of fs, separated by fp=2fs.
    This lower-level entry point also accepts the historical asymmetric basis
    for direct comparison with realistic_pumped_unit_cell_abcd. Coefficients
    follow C(t)=sum C_d exp(i*d*fp*t), with Hermitian negative bands.
    """
    _positive(f_s, "f_s")
    idx = np.asarray(indices, int)
    if idx.ndim != 1 or len(idx) == 0 or len(set(idx)) != len(idx) or np.any(idx % 2 == 0):
        raise ValueError("indices must be distinct signed odd integers")
    samples = [_sample(cell_fn, abs(i) * f_s) for i in idx]
    coeff = np.asarray(coefficients, complex)
    if coeff.ndim != 1 or not len(coeff) or not np.all(np.isfinite(coeff)) or coeff[0].real <= 0 or coeff[0].imag != 0:
        raise ValueError("Finite C coefficients with positive real C0 required")
    ct = np.zeros((len(idx), len(idx)), complex)
    for r, ir in enumerate(idx):
        for c, ic in enumerate(idx):
            d = (ir - ic) // 2
            if abs(d) < len(coeff):
                ct[r, c] = coeff[d] if d >= 0 else coeff[-d].conjugate()
    def signed(values):
        return np.array([z.conjugate() if i < 0 else z for z, i in zip(values, idx)])
    def lift(name):
        return tuple(np.diag(signed([getattr(s, name)[i, j] for s in samples])) for i, j in [(0, 0), (0, 1), (1, 0), (1, 1)])
    al, bl, cl, dl = lift("left")
    ar, br, cr, dr = lift("right")
    yc = 2j * np.pi * f_s * idx[:, None] * ct
    yb = np.linalg.inv(np.linalg.inv(yc) + np.diag(signed([s.series_impedance_ohm for s in samples])))
    yb += np.diag(signed([s.shunt_admittance_s for s in samples]))
    return (al @ ar + bl @ (yb @ ar + cr), al @ br + bl @ (yb @ br + dr),
            cl @ ar + dl @ (yb @ ar + cr), cl @ br + dl @ (yb @ br + dr))


def loaded_harmonic_response(*, cells, Z0=50):
    """Return (input-to-output voltage transfer, per-cell transfers).

    Inputs are incident peak phasors. Loaded-admittance recursion avoids the
    exponentially large global ABCD product in deep stopbands.
    """
    _positive(Z0, "Z0")
    cells = list(cells)
    if not cells:
        raise ValueError("At least one cell is required")
    n = np.asarray(cells[0][0]).shape[0]
    eye = np.eye(n, dtype=complex)
    load = eye / Z0
    prop = []
    for blocks in cells[::-1]:
        if len(blocks) != 4 or any(np.shape(x) != (n, n) or not np.all(np.isfinite(x)) for x in blocks):
            raise ValueError("All harmonic blocks must be finite square matrices of equal size")
        a, b, c, d = blocks
        ei = np.linalg.inv(a + b @ load)
        prop.append(ei)
        load = (c + d @ load) @ ei
    total = 2 * np.linalg.inv(eye + Z0 * load)
    profiles = []
    for ei in prop[::-1]:
        total = ei @ total
        profiles.append(total.copy())
    if not np.all(np.isfinite(total)):
        raise FloatingPointError("Non-finite harmonic solution")
    return total, np.asarray(profiles)


def _db(value):
    return float(20 * np.log10(max(abs(value), 1e-300)))


def delivered_gain(*, cell_fn, N, f_s, P_pump_dBm, P_sig_dBm, harmonics=(1, 3), phase="best", Z0=50, m_cap=0.9):
    """Evaluate spatially delivered degenerate gain, never a certification.

    ``phase`` is a global pump phase in radians or 'best'.  ``harmonics`` is a
    consecutive positive odd basis starting at 1; (1,3) is the R5 baseline.
    Sensitivity is always reported between (1,3) and (1,3,5,7), independently
    of the requested output basis. A cell built with ``tee_out_of_range="raise"``
    refuses samples above its tee-model edge: the requested basis then raises
    :class:`TeeModelOutOfRange`, and a sensitivity basis beyond the edge gives
    ``harmonic_sensitivity_dB = None`` with the reason in
    ``meta['harmonic_sensitivity_unavailable']``. Per-cell signal is the complex endpoint
    voltage at +fs at the chosen global phase, not a power-generation profile.
    P_sig_dBm sets its scale; undepleted small-signal gain is scale independent.
    """
    _positive(f_s, "f_s")
    _positive(Z0, "Z0")
    if not np.isfinite(P_sig_dBm):
        raise ValueError("A finite, nonzero incident signal is required for gain")
    hs = tuple(harmonics)
    if len(hs) < 2 or hs != tuple(range(1, 2 * len(hs), 2)):
        raise ValueError("harmonics must be consecutive positive odd integers starting at 1, with at least (1,3)")
    if not (isinstance(phase, str) and phase == "best") and (isinstance(phase, str) or not np.isfinite(phase)):
        raise ValueError("phase must be 'best' or finite radians")
    fns = _functions(cell_fn, N)
    samples = [_sample(f, f_s) for f in fns]
    vj = delivered_pump_profile(cell_fn=fns, N=N, f_p=2*f_s, P_inc_dBm=P_pump_dBm, Z0=Z0)
    m, phi, capped = modulation_profile(V_j=vj, M=[c.M for c in samples], Vj=[c.Vj for c in samples], m_cap=m_cap)
    _, passive, _ = _passive(samples, f_s, 1.0, Z0)
    passive_db = _db(passive[-1])
    cache = {}
    def calculate(basis):
        indices = np.r_[-np.array(basis)[::-1], basis]
        blocks = [harmonic_cell(cell_fn=fn, f_s=f_s, indices=indices,
                  coefficients=[s.junction_capacitance_f, .5*s.junction_capacitance_f*mk*np.exp(1j*pk)])
                  for fn, s, mk, pk in zip(fns, samples, m, phi)]
        trans, profiles = loaded_harmonic_response(cells=blocks, Z0=Z0)
        pos, neg = indices.tolist().index(1), indices.tolist().index(-1)
        a, b = trans[pos, pos], trans[pos, neg]
        ph = float(np.angle(a)-np.angle(b)) if phase == "best" else float(phase)
        signal = (profiles[:, pos, pos] + profiles[:, pos, neg]*np.exp(1j*ph))*_voltage(P_sig_dBm, Z0)
        amplitude = abs(a)+abs(b) if phase == "best" else abs(a+b*np.exp(1j*ph))
        return dict(absolute=_db(amplitude), signal=signal, phase=ph, best=_db(abs(a)+abs(b)), worst=_db(abs(abs(a)-abs(b))))
    unavailable = None
    for basis in dict.fromkeys([hs, (1,3), (1,3,5,7)]):
        try:
            cache[basis] = calculate(basis)
        except TeeModelOutOfRange as exc:
            if basis == hs:
                raise
            unavailable = str(exc)
    chosen = cache[hs]
    # The sensitivity basis is a diagnostic: above the tee-model edge it is withheld, not estimated.
    sensitivity = None if unavailable else cache[(1,3,5,7)]["absolute"] - cache[(1,3)]["absolute"]
    return dict(onoff_dB=chosen["absolute"]-passive_db, absolute_outin_dB=chosen["absolute"],
                passive_S21_dB=passive_db, regime="large_m" if float(np.max(m)) > .3 else "small_m",
                max_m=float(np.max(m)), mean_elastance_shift=1+float(np.max(m))**2/2, per_cell_signal=chosen["signal"], per_cell_m=m,
                per_cell_phi=phi, junction_pump_V=vj, capped_cells=capped,
                harmonic_sensitivity_dB=sensitivity, phase_rad=chosen["phase"],
                best_absolute_outin_dB=chosen["best"], worst_absolute_outin_dB=chosen["worst"],
                meta=dict(tier="ANALYTICAL", label="analytical_proxy_only", N=int(N), Z0=float(Z0),
                          harmonics=hs, sensitivity_bases=((1,3),(1,3,5,7)),
                          needs_native_HB_order_check=True if sensitivity is None else abs(sensitivity)>1,
                          harmonic_sensitivity_unavailable=unavailable,
                          calibration_bias_dB=None, certified=False,
                          model="undepleted linear pump; first-order periodic capacitance; no conduction or DC rectification"))
