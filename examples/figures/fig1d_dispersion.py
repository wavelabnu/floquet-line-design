"""Fig 1d -- predicted pump-off Bloch dispersion of the fabricated SMV1245 cell.

Computes the passive (pump-off) Bloch dispersion from the framework analytical
leg at its CORRECTED_NOTEBOOK tee default, exports the curve as CSV + provenance,
and renders the panel in the measured-dispersion orientation
(x = beta_B * p / pi in [0, 1], y = frequency in GHz).

Framework path (single source of truth)
--------------------------------------
    spec  = real_part_designs.spec_smv1245()      # as-built paper geometry:
            main w = 4.27 mm, period p = 6.11 mm, MTEE branch w3 = 1.93 mm,
            3-section OPEN/MLEF stub [(1.93, 6.83), (1.41, 5.25), (1.98, 1.04)] mm
            + 2 mid-stub VIAGND
    dev   = get_entry("SMV1245-079LF").to_realistic_varactor(v_bias_v=0.0)
    cell  = realistic_pumped_unit_cell_abcd(spec, dev, m_branch_pump=0.0,
                truncation_N=1, use_mtee_parasitic=True, tee_model_variant=None,
                stub_y_fn_override=<tabulated stub>)
    gamma = tvtl.theory.dispersion.bloch_dispersion(central 2x2 (A, B, C, D),
                period = 6.11 mm)          # cosh(gamma*p) = (A + D)/2

``tee_model_variant=None`` selects the CORRECTED_NOTEBOOK tee (the
``tee_harmonic_blocks`` default). "Pump-off" is the drive ``m_branch_pump=0.0``:
at zero drive the harmonic blocks decouple, so the central (n = 0) 2x2 block is
the LTI passive cell and the result is independent of the truncation order. The
generator asserts that truncation-independence directly (N = 1 vs N = 6 at
3.1 GHz). ``truncation_N=1`` is used for the sweep so the stub table is only
queried at the harmonics {-f, f, 3f}, which stay inside the tabulated
0.05-80 GHz span across the whole 0.5-16 GHz sweep (N = 6 would reach 13*f and
hard-error above ~6.15 GHz).

Stub branch
-----------
The fabricated cell grounds the stub at two INTERMEDIATE nodes, which the
framework's single-chain stub tracer cannot represent, so the branch admittance
is the committed element-accurate tabulated stub
(``examples/real_parts/results/stub_tables/smv1245_stub_80GHz.s1p``) injected as
``stub_y_fn_override`` -- the same treatment the shipped boundary record uses.
The tee, through-line, package and static varactor C0 are framework analytical
models.

Overlay hook
------------
When ``fig1d_comsol.csv`` (columns ``f_GHz, betap_over_pi``) is present in the
output directory it is overlaid with distinct markers and the legend entry
"COMSOL eigenmode". No degenerate-pair markers are drawn on this panel.

Outputs (into ``$TVTL_FIG_OUTDIR`` if set, else ``results/figures/``):
``fig1d_dispersion.pdf`` / ``.png`` / ``.sidecar.json``, ``fig1d_analytical.csv``,
``fig1d_analytical.provenance.json``.

Usage:
    python examples/figures/fig1d_dispersion.py
"""
from __future__ import annotations

import csv
import json
import math
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
for _p in (REPO_ROOT / "src",
           REPO_ROOT / "examples" / "real_parts",
           REPO_ROOT / "examples" / "real_parts" / "boundary",
           Path(__file__).resolve().parent):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import figstyle  # noqa: E402

from tvtl.theory.dispersion import bloch_dispersion  # noqa: E402
from tvtl.varactor_palette import get_entry  # noqa: E402
from tvtl_ads_builder_v2.analytical.realistic_c_gain import (  # noqa: E402
    realistic_pumped_unit_cell_abcd,
)
from tvtl_ads_builder_v2.device_spec import with_var_overrides  # noqa: E402

from real_part_designs import spec_smv1245, _SMV1245_MAIN_PERIOD_M  # noqa: E402
from _stub_table import STUB_80GHz, load_s1p, make_y_override  # noqa: E402

PERIOD_M = float(_SMV1245_MAIN_PERIOD_M)   # 6.11 mm Bloch period
F_LO_HZ, F_HI_HZ = 0.5e9, 16.0e9
BASE_STEP_HZ = 4.0e6                        # base sweep grid
REFINE_HALFWIN_HZ = 0.12e9                  # +/-120 MHz refinement window
REFINE_STEP_HZ = 1.0e6                      # refined grid near band edges
MARKER_FREQS_GHZ = (3.1, 6.2, 5.3, 10.6)    # feed for the Fig 5a pair markers
VERIFY_FREQ_HZ = 3.1e9                      # truncation-independence check point

SRC_PATHS = (
    "examples/real_parts/real_part_designs.py",
    "src/tvtl_ads_builder_v2/analytical/realistic_c_gain.py",
    "src/tvtl_ads_builder_v2/analytical/mtee_blocks.py",
    "src/tvtl/theory/dispersion.py",
    "examples/real_parts/boundary/_stub_table.py",
    "examples/real_parts/results/stub_tables/smv1245_stub_80GHz.s1p",
    "src/tvtl/varactor_palette.py",
)

_CTX: tuple | None = None


def _context():
    """Lazily build (spec, device, stub override); cached across calls."""
    global _CTX
    if _CTX is None:
        dev = get_entry("SMV1245-079LF").to_realistic_varactor(v_bias_v=0.0)
        f_tab, y_tab = load_s1p(STUB_80GHz)
        _CTX = (spec_smv1245(), dev, make_y_override(f_tab, y_tab))
    return _CTX


def central_block(f_hz: float, truncation_N: int = 1,
                  fpump_factor: float = 2.0) -> np.ndarray:
    """Pump-off (drive 0) central-harmonic 2x2 unit-cell ABCD at ``f_hz``."""
    spec, dev, y_override = _context()
    spec = with_var_overrides(spec, {"f_signal": f_hz,
                                     "f_pump": fpump_factor * f_hz})
    a, b, c, d = realistic_pumped_unit_cell_abcd(
        spec, dev, m_branch_pump=0.0, truncation_N=truncation_N,
        use_mtee_parasitic=True, tee_model_variant=None,
        stub_y_fn_override=y_override,
    )
    n = truncation_N
    return np.array([[a[n, n], b[n, n]], [c[n, n], d[n, n]]], dtype=complex)


def dispersion_at(f_hz: float) -> tuple[float, float, complex]:
    """Return ``(beta_B*p/pi, alpha*p [Np/cell], half_trace)`` at ``f_hz``."""
    ab = central_block(f_hz)
    gamma = complex(np.asarray(bloch_dispersion(
        (ab[0, 0], ab[0, 1], ab[1, 0], ab[1, 1]), PERIOD_M)).ravel()[0])
    half = (ab[0, 0] + ab[1, 1]) / 2.0
    return gamma.imag * PERIOD_M / math.pi, gamma.real * PERIOD_M, half


def find_band_edges(f_hz: np.ndarray, re_half: np.ndarray) -> list[dict]:
    """Band edges: ``Re[(A+D)/2]`` crossing +1 (zone centre, beta = 0) or
    -1 (zone boundary, beta = pi). Each sign change is linearly interpolated."""
    edges: list[dict] = []
    for level, kind in ((1.0, "zone_center_beta0"), (-1.0, "zone_boundary_betaPi")):
        g = np.asarray(re_half) - level
        sign = np.sign(g)
        for i in np.where(np.diff(sign) != 0)[0]:
            f0, f1, y0, y1 = f_hz[i], f_hz[i + 1], g[i], g[i + 1]
            fx = f0 - y0 * (f1 - f0) / (y1 - y0) if y1 != y0 else f0
            edges.append({"f_GHz": float(fx / 1e9), "kind": kind, "level": level})
    edges.sort(key=lambda e: e["f_GHz"])
    return edges


def scan_band_edges(f_lo_hz: float = F_LO_HZ, f_hi_hz: float = F_HI_HZ,
                    step_hz: float = BASE_STEP_HZ) -> list[dict]:
    """Sweep ``[f_lo, f_hi]`` and return the interpolated band edges."""
    grid = np.arange(f_lo_hz, f_hi_hz + 0.5 * step_hz, step_hz)
    re_half = np.array([dispersion_at(f)[2].real for f in grid])
    return find_band_edges(grid, re_half)


def compute_dispersion() -> dict:
    """Base sweep + refinement near each band edge; returns the full curve."""
    base = np.arange(F_LO_HZ, F_HI_HZ + 0.5 * BASE_STEP_HZ, BASE_STEP_HZ)
    re_half = np.array([dispersion_at(f)[2].real for f in base])
    edges = find_band_edges(base, re_half)

    grid = set(base.tolist())
    for edge in edges:
        fc = edge["f_GHz"] * 1e9
        lo = max(F_LO_HZ, fc - REFINE_HALFWIN_HZ)
        hi = min(F_HI_HZ, fc + REFINE_HALFWIN_HZ)
        grid.update(np.arange(lo, hi + 0.5 * REFINE_STEP_HZ, REFINE_STEP_HZ).tolist())
    fs = np.array(sorted(grid))

    betap_pi = np.empty(fs.size)
    alphap = np.empty(fs.size)
    for i, f in enumerate(fs):
        betap_pi[i], alphap[i], _ = dispersion_at(f)
    return {"f_hz": fs, "betap_over_pi": betap_pi, "alphap": alphap,
            "band_edges": edges, "n_base": int(base.size)}


def _read_two_column(path: Path, cols: tuple[str, str]):
    xs, ys = [], []
    with open(path, newline="", encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            xs.append(float(row[cols[0]]))
            ys.append(float(row[cols[1]]))
    return xs, ys


def main() -> int:
    outdir = figstyle.resolve_outdir()
    result = compute_dispersion()
    fs, betap_pi, alphap = result["f_hz"], result["betap_over_pi"], result["alphap"]
    edges = result["band_edges"]

    print(f"base grid {result['n_base']} pts; {len(edges)} band edge(s) "
          f"(Re[(A+D)/2] crossing +-1):")
    for e in edges:
        print(f"  {e['f_GHz']:7.4f} GHz  ({e['kind']})")

    csv_path = outdir / "fig1d_analytical.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["f_GHz", "betap_over_pi", "alphap_Np_per_cell"])
        for f, b, a in zip(fs, betap_pi, alphap):
            w.writerow([f"{f / 1e9:.6f}", f"{b:.9f}", f"{a:.9e}"])
    print(f"wrote {figstyle.repo_relative(csv_path)}  ({fs.size} rows)")

    # Truncation-independence of the pump-off central block (public API only).
    d_trunc = float(np.max(np.abs(central_block(VERIFY_FREQ_HZ, 1)
                                  - central_block(VERIFY_FREQ_HZ, 6))))
    print(f"truncation check |N=1 - N=6| @ {VERIFY_FREQ_HZ / 1e9:g} GHz = {d_trunc:.2e}")

    markers = {}
    print("marker values (betap/pi, alphap Np/cell):")
    for fg in MARKER_FREQS_GHZ:
        b, a, half = dispersion_at(fg * 1e9)
        markers[f"{fg:g}GHz"] = {"betap_over_pi": b, "alphap_Np_per_cell": a,
                                 "Re_half_trace": half.real, "Im_half_trace": half.imag}
        print(f"  {fg:5.2f} GHz  betap/pi={b:+.6f}  alphap={a:+.6e}")

    provenance = {
        "panel_data": figstyle.repo_relative(csv_path),
        "generated_by": "examples/figures/fig1d_dispersion.py",
        "command": "python examples/figures/fig1d_dispersion.py",
        "framework_call": (
            "realistic_pumped_unit_cell_abcd(spec_smv1245(), SMV1245-079LF@0V, "
            "m_branch_pump=0.0, truncation_N=1, use_mtee_parasitic=True, "
            "tee_model_variant=None [=CORRECTED_NOTEBOOK], "
            "stub_y_fn_override=smv1245_stub_80GHz.s1p); central (n=0) 2x2 -> "
            "tvtl.theory.dispersion.bloch_dispersion(period=6.11mm)"
        ),
        "cell": ("fabricated SMV1245 board (spec_smv1245): main w=4.27mm, "
                 "period p=6.11mm, MTEE branch w3=1.93mm, 3-section OPEN/MLEF stub "
                 "[(1.93,6.83),(1.41,5.25),(1.98,1.04)]mm + 2 mid-stub VIAGND"),
        "varactor": ("SMV1245-079LF to_realistic_varactor(v_bias_v=0.0); "
                     "static C0=7.37 pF, CJ0=6.9 pF, CP=0.47 pF, L_pkg=0.7 nH, "
                     "R_series=2.0 ohm"),
        "pump_off_definition": ("m_branch_pump=0.0 (drive 0; harmonic blocks "
                                "decouple -> LTI passive cell)"),
        "period_m": PERIOD_M,
        "freq_range_GHz": [F_LO_HZ / 1e9, F_HI_HZ / 1e9],
        "base_step_Hz": BASE_STEP_HZ,
        "refine_step_Hz": REFINE_STEP_HZ,
        "refine_halfwin_Hz": REFINE_HALFWIN_HZ,
        "n_rows": int(fs.size),
        "band_edges": edges,
        "band_edge_definition": ("Re[(A+D)/2] crossing +1 (zone centre beta=0) "
                                 "or -1 (zone boundary beta=pi)"),
        "verification": {
            "central_block_N1_vs_N6_absdiff": d_trunc,
            "at_frequency_GHz": VERIFY_FREQ_HZ / 1e9,
            "note": ("the pump-off central block is truncation-independent "
                     "(drive 0 decouples the harmonics)"),
        },
        "markers_for_fig5a": markers,
        "sources": [{"path": s, "git_short_hash": figstyle.git_short_hash(s)}
                    for s in SRC_PATHS],
        "no_degenerate_pair_markers_on_this_panel": True,
    }
    prov_path = outdir / "fig1d_analytical.provenance.json"
    prov_path.write_text(json.dumps(provenance, indent=2, default=str), encoding="utf-8")
    print(f"wrote {figstyle.repo_relative(prov_path)}")

    # ---- panel ------------------------------------------------------------- #
    fig, ax = figstyle.new_panel(height_in=3.05)
    for e in edges:
        ax.axhline(e["f_GHz"], color=figstyle.C_GREY, lw=0.4, ls=(0, (2, 2)),
                   alpha=0.55, zorder=1)
    ax.plot(betap_pi, fs / 1e9, "-", color=figstyle.C_PRIMARY,
            lw=figstyle.DATA_LW, zorder=3, label="analytical (pump-off)")

    comsol = outdir / "fig1d_comsol.csv"
    overlaid = False
    n_overlay = 0
    if comsol.exists():
        cf, cb = _read_two_column(comsol, ("f_GHz", "betap_over_pi"))
        ax.plot(cb, cf, linestyle="none", marker="D", mfc="none",
                mec=figstyle.C_ACCENT, mew=1.0, ms=4.5, zorder=4,
                label="COMSOL eigenmode")
        overlaid = True
        n_overlay = len(cf)

    ax.set_xlabel(r"$\beta_B\,p/\pi$")
    ax.set_ylabel("frequency (GHz)")
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(F_LO_HZ / 1e9, F_HI_HZ / 1e9)
    ax.set_xticks([0.0, 0.25, 0.5, 0.75, 1.0])
    if overlaid:
        ax.legend(loc="lower right")
    fig.tight_layout()

    out_base = outdir / "fig1d_dispersion"
    paths = figstyle.save_panel(fig, out_base)
    sc = figstyle.write_sidecar(
        out_base, sources=list(SRC_PATHS),
        script="examples/figures/fig1d_dispersion.py",
        command="python examples/figures/fig1d_dispersion.py",
        key_values={
            "x_label": "beta_B p / pi (in [0,1])",
            "y_label": "frequency (GHz)",
            "orientation": "measured-dispersion (x=betap/pi, y=f)",
            "n_analytical_points": int(fs.size),
            "band_edges_GHz": [e["f_GHz"] for e in edges],
            "markers_for_fig5a": markers,
            "overlay_present": overlaid,
            "overlay_points": n_overlay,
            "no_degenerate_pair_markers": True,
        },
        extra={
            "analytical_csv": figstyle.repo_relative(csv_path),
            "analytical_provenance": figstyle.repo_relative(prov_path),
            "overlay_hook_file": figstyle.repo_relative(comsol),
        },
    )
    print(f"wrote {paths['pdf']} / {paths['png']}  "
          f"(overlay={'yes' if overlaid else 'no'})  {figstyle.repo_relative(sc)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
