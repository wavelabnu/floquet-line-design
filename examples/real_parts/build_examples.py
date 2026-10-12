"""Build, verify, and assemble the three real-part TVTL examples.

Regenerates the SMV1245 / SMV2020 / MAVR designs (see ``real_part_designs.py``)
through the hierarchical OpenAccess emitter -- a ``unit_cell`` subcircuit plus a
``unit_cell`` symbol, cascaded as ``unit_cell:Xk`` instances on a clean top-level
schematic -- and proves that the hierarchical design is electrically identical to
the flat reference deck (the parity gate).

Modes::

  python build_examples.py specs                 build all 3 specs, print structure (no ADS)
  <ADS_PYTHON> build_examples.py parity [cell]   emit flat + hier, HB both, parity gate (cell|all)
  <ADS_PYTHON> build_examples.py shots  [cell]   render unit-cell + top-level screenshots (cell|all)
  <ADS_PYTHON> build_examples.py assemble        emit all 3 hierarchically + assemble one workspace
  <ADS_PYTHON> build_examples.py all             parity + shots + assemble for all three

``specs`` runs under base Python. ``parity`` / ``shots`` / ``assemble`` require
Keysight ADS Python (the ``keysight.ads.de`` OpenAccess bridge and the
``hpeesofsim`` harmonic-balance engine). Generated workspaces and simulator runs
are written under ``examples/real_parts/outputs/`` (git-ignored); the parity
proofs and screenshots are written under ``examples/real_parts/results/``.
"""
from __future__ import annotations

import json
import math
import re
import shutil
import sys
from pathlib import Path

# Allow running as a plain script (``python build_examples.py``) without an
# editable install: make the sibling design module importable.
sys.path.insert(0, str(Path(__file__).resolve().parent))
from real_part_designs import (  # noqa: E402
    EXAMPLES as _EXAMPLES_REAL,
    REFERENCE_VOUT_SIGNAL as _REF_REAL,
    resolved_spec,
)
from proxy_designs import (  # noqa: E402
    EXAMPLES_PROXY as _EXAMPLES_PROXY,
    REFERENCE_VOUT_SIGNAL_PROXY as _REF_PROXY,
)

# All six examples: the three real-part designs + the three optimizer-picked proxy
# designs (proxy_designs.py). The proxy trio extends the cell coverage and is the
# first to carry input/output matching networks (the trio's edge impedances need
# none), so they exercise the hierarchical emitter's matching-network placement.
EXAMPLES = _EXAMPLES_REAL + _EXAMPLES_PROXY
REFERENCE_VOUT_SIGNAL = {**_REF_REAL, **_REF_PROXY}

_HERE = Path(__file__).resolve().parent
OUTPUTS = _HERE / "outputs"          # git-ignored (generated workspaces + sim runs)
RESULTS = _HERE / "results"          # committed (parity proofs + screenshots)
WORKSPACE = OUTPUTS / "TVTL_examples_workspace"  # six examples (3 real-part + 3 proxy)

# Expected resolved-spec structure (the frozen design fingerprint). A regenerated
# spec whose element/node counts or per-cell body differ from these has drifted
# from the recorded design.
EXPECTED_STRUCTURE = {
    "smv1245": {"n_elements": 244, "n_nodes": 212,
                "uc_body": {"MLIN": 5, "MTEE": 1, "L": 1, "R": 1, "SDD2P": 1,
                            "I_Probe": 1, "MLEF": 1, "VIAGND": 2}},
    # DOUBLE-via families emit two VIAGND per cell since v1.0.1 (1.0.0: one; 203 / 233 elements for the two
    # double proxies). smv2020 is a single-via family (n1_single_short).
    "smv2020": {"n_elements": 292, "n_nodes": 290,
                "uc_body": {"MLIN": 3, "MTEE": 1, "L": 1, "R": 1, "SDD2P": 1,
                            "I_Probe": 1, "VIAGND": 1}},
    "mavr": {"n_elements": 412, "n_nodes": 386,
             "uc_body": {"MLIN": 7, "MTEE": 1, "L": 1, "R": 1, "SDD2P": 1,
                         "I_Probe": 1, "MLEF": 1, "VIAGND": 1}},
    # Optimizer-picked proxy designs (SHORT / VIAGND, single-stub matching, 15 cells).
    # uc_body is the per-cell body; the MATCH_* matching networks are scaffolding
    # (counted in n_elements, not in the per-cell body).
    "proxy_n2_double_short": {"n_elements": 218, "n_nodes": 201,
                              "uc_body": {"MLIN": 4, "MTEE": 1, "L": 1, "R": 1, "SDD2P": 1,
                                          "I_Probe": 1, "VIAGND": 2}},
    "proxy_n3_single_short": {"n_elements": 218, "n_nodes": 216,
                              "uc_body": {"MLIN": 5, "MTEE": 1, "L": 1, "R": 1, "SDD2P": 1,
                                          "I_Probe": 1, "VIAGND": 1}},
    "proxy_n4_double_short": {"n_elements": 248, "n_nodes": 231,
                              "uc_body": {"MLIN": 6, "MTEE": 1, "L": 1, "R": 1, "SDD2P": 1,
                                          "I_Probe": 1, "VIAGND": 2}},
}

_BY_KEY = {key: (key, builder, lib, top, headline)
           for (key, builder, lib, top, headline) in EXAMPLES}


def _funds(spec):
    """(f_signal_Hz, f_pump_Hz) from the spec's two-tone HB directive."""
    tt = spec.hb_two_tone
    if tt is None:
        raise ValueError("spec carries no hb_two_tone directive")
    return (float(tt.fundamental_freq_1), float(tt.fundamental_freq_2))


def _per_cell_body(spec):
    """Cell-1 element multiset (the unit-cell body)."""
    body = {}
    for e in spec.elements:
        if e.type == "NAME":
            continue
        if re.search(r"_C1(?:_\d+)?$", e.id):
            body[e.type] = body.get(e.type, 0) + 1
    return body


# --------------------------------------------------------------------------- #
# specs -- ADS-free structural validation
# --------------------------------------------------------------------------- #
def mode_specs(_args):
    from tvtl_ads_builder_v2.netlist import emit_flat_hb_netlist

    print("=== real-part example specs (ADS-free) ===", flush=True)
    all_ok = True
    out = {}
    for key, builder, lib, top, headline in EXAMPLES:
        spec = resolved_spec(builder)
        body = _per_cell_body(spec)
        f_s, f_p = _funds(spec)
        deck = emit_flat_hb_netlist(spec)
        exp = EXPECTED_STRUCTURE[key]
        ok = (len(spec.elements) == exp["n_elements"]
              and len(spec.nodes) == exp["n_nodes"]
              and body == exp["uc_body"])
        all_ok = all_ok and ok
        rec = {"headline": headline, "f_s_GHz": f_s / 1e9, "f_pump_GHz": f_p / 1e9,
               "n_elements": len(spec.elements), "n_nodes": len(spec.nodes),
               "deck_lines": len(deck.splitlines()), "uc_body": body,
               "structure_ok": ok}
        out[key] = rec
        decks_dir = RESULTS / "decks"
        decks_dir.mkdir(parents=True, exist_ok=True)
        (decks_dir / f"{key}_flat_hb_deck.ckt").write_text(deck, encoding="utf-8")
        print(f"\n  -- {key}: {headline}", flush=True)
        print(f"     f_s={f_s/1e9:.3g} GHz  f_pump={f_p/1e9:.3g} GHz  "
              f"elements={len(spec.elements)} (expect {exp['n_elements']})  "
              f"nodes={len(spec.nodes)} (expect {exp['n_nodes']})  "
              f"deck_lines={len(deck.splitlines())}", flush=True)
        print(f"     unit-cell body: {body}", flush=True)
        print(f"     structure {'OK' if ok else 'MISMATCH vs recorded design'}", flush=True)
    RESULTS.mkdir(parents=True, exist_ok=True)
    (RESULTS / "specs.json").write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(f"\n  wrote {RESULTS / 'specs.json'}", flush=True)
    print(f"\n  >>> ALL SPECS {'OK' if all_ok else 'MISMATCH'}", flush=True)
    return 0 if all_ok else 1


# --------------------------------------------------------------------------- #
# parity -- the electrical-identity gate (requires ADS)
# --------------------------------------------------------------------------- #
def _flatten_hier(netlist_text: str) -> str:
    """Inline every ``define <uc> (...) ... end <uc>`` subckt into its instances.

    Produces a flat ADS-dialect deck equivalent to the hierarchical one: terminal
    nets are substituted from each instance's net list, the global net ``0`` is
    preserved, every other internal net is scoped per instance (``X1__n_c1_b``),
    and model lines are de-duplicated. Used only to feed the netlist comparator.
    """
    from tvtl_ads_builder_v2.oa.netlist_compare import (
        _logical_lines, _split_nets_and_params, _tokenize,
    )

    lines = _logical_lines(netlist_text)
    subckts: dict[str, dict] = {}
    cur = None
    top: list[str] = []
    models: dict[str, str] = {}
    for ln in lines:
        s = ln.strip()
        if not s or s.startswith(";"):
            continue
        toks = _tokenize(s)
        head = toks[0]
        if head == "define":
            name = toks[1]
            inside = s[s.find("(") + 1:s.rfind(")")].split()
            cur = {"name": name, "terms": inside, "body": [], "models": []}
            subckts[name] = cur
            continue
        if head == "end":
            cur = None
            continue
        if head == "model":
            models[toks[1]] = s
            continue
        if cur is not None:
            cur["body"].append(s)
            continue
        top.append(s)

    def _subst(line: str, term_map: dict, xname: str) -> str:
        toks = _tokenize(line)
        head = toks[0]
        type_token, _, name = head.partition(":")
        rest = toks[1:]
        nets, _params = _split_nets_and_params(rest)
        n = len(nets)

        def mp(t: str) -> str:
            if t in term_map:
                return term_map[t]
            if t == "0":
                return "0"
            return f"{xname}__{t}"

        new_nets = [mp(t) for t in nets]
        new_head = f"{type_token}:{xname}__{name}" if name else type_token
        return " ".join([new_head] + new_nets + rest[n:])

    out: list[str] = list(models.values())
    for s in top:
        toks = _tokenize(s)
        head = toks[0]
        type_token, _, _name = head.partition(":")
        if type_token in subckts:
            sub = subckts[type_token]
            inst_nets, _p = _split_nets_and_params(toks[1:])
            term_map = dict(zip(sub["terms"], inst_nets))
            xname = head.split(":", 1)[1] if ":" in head else head
            for body_line in sub["body"]:
                out.append(_subst(body_line, term_map, xname))
        else:
            out.append(s)
    return "\n".join(out) + "\n"


def _rename_boundary_nets(hier_nl: str) -> str:
    """Rename the hier deck's output (Port Num=3) / input (Num=1) nets to n_output / n_input.

    The hierarchical netlist auto-names its boundary nets; renaming them to the
    flat deck's names lets the output-node voltage be compared directly by name.
    """
    out_net = in_net = None
    for ln in hier_nl.splitlines():
        s = ln.strip()
        if s.startswith("Port:"):
            parts = s.split()
            net = parts[1] if len(parts) > 1 else None
            if "Num=3" in s:
                out_net = net
            elif "Num=1" in s:
                in_net = net
    text = hier_nl
    if out_net:
        text = re.sub(rf"\b{re.escape(out_net)}\b", "n_output", text)
    if in_net:
        text = re.sub(rf"\b{re.escape(in_net)}\b", "n_input", text)
    return text


def _run_and_read(deck_text, run_dir, label):
    from tvtl_ads_builder_v2.simulate.parser import read_raw_dataset
    from tvtl_ads_builder_v2.simulate.runner import run_hpeesofsim
    run_dir.mkdir(parents=True, exist_ok=True)
    deck = run_dir / f"{label}.ckt"
    deck.write_text(deck_text, encoding="utf-8")
    res = run_hpeesofsim(deck, timeout_s=600)
    info = {"status": res["status"], "dataset": res.get("dataset_path"),
            "stderr_tail": res.get("stderr_tail")}
    if res["status"] != "completed":
        return info, None
    return info, read_raw_dataset(res["dataset_path"])


def _sim_parity(spec, hier_nl, rundir):
    """HB on the flat deck + the hier deck; compare V(n_output) harmonic spectrum."""
    from tvtl_ads_builder_v2.analysis.hb_dataset import SIGNAL_MIX_2TONE, hb_dataset_from_raw
    from tvtl_ads_builder_v2.netlist import emit_flat_hb_netlist

    if rundir.exists():
        shutil.rmtree(rundir)
    flat_deck = emit_flat_hb_netlist(spec)
    hier_run_deck = _rename_boundary_nets(hier_nl)
    flat_info, flat_raw = _run_and_read(flat_deck, rundir / "flat", "flat_run")
    hier_info, hier_raw = _run_and_read(hier_run_deck, rundir / "hier", "hier_run")
    res = {"flat_run": flat_info["status"], "hier_run": hier_info["status"]}
    if flat_raw is None or hier_raw is None:
        res.update({"max_abs_dV": None, "verdict": "RUN_FAILED",
                    "flat_detail": flat_info, "hier_detail": hier_info})
        return res
    funds = _funds(spec)
    try:
        fv = hb_dataset_from_raw(flat_raw, funds).node_voltage("n_output")
        hv = hb_dataset_from_raw(hier_raw, funds).node_voltage("n_output")
    except KeyError as exc:
        res.update({"max_abs_dV": None, "verdict": "NO_OUTPUT_NODE", "detail": str(exc)})
        return res
    fmix = {h.mix: h.amplitude for h in fv.harmonics}
    hmix = {h.mix: h.amplitude for h in hv.harmonics}
    common = sorted(set(fmix) & set(hmix))
    max_d = max((abs(fmix[m] - hmix[m]) for m in common), default=float("nan"))
    vsig_f = fv.at_mix(SIGNAL_MIX_2TONE)
    vsig_h = hv.at_mix(SIGNAL_MIX_2TONE)
    gain_delta_dB = (20.0 * math.log10(abs(vsig_h) / abs(vsig_f))
                     if abs(vsig_f) > 0 and abs(vsig_h) > 0 else float("nan"))
    res.update({
        "n_mix_common": len(common),
        "Vout_signal_flat": f"{abs(vsig_f):.6e}", "Vout_signal_hier": f"{abs(vsig_h):.6e}",
        "gain_delta_dB": gain_delta_dB, "max_abs_dV": max_d,
        "verdict": "IDENTICAL" if max_d < 1e-9 else ("CLOSE" if max_d < 1e-6 else "DIFFER"),
    })
    return res


def _emit_hier(spec, key, lib, top):
    from tvtl_ads_builder_v2.oa.oa_hier_emitter import emit_hierarchical_workspace_from_spec
    ws = OUTPUTS / "emit" / key / "ws"
    if ws.exists():
        shutil.rmtree(ws)
    res = emit_hierarchical_workspace_from_spec(
        spec, ws, library_name=lib, top_cell_name=top, uc_cell_name="unit_cell")
    return res, Path(res["ads_netlist_path"]).read_text(encoding="utf-8")


def _parity_one(key):
    from tvtl_ads_builder_v2.netlist import emit_flat_hb_netlist
    from tvtl_ads_builder_v2.oa.netlist_compare import _type_counts, parse_netlist_text

    _key, builder, lib, top, headline = _BY_KEY[key]
    print(f"\n=== PARITY: {key} -- {headline} ===", flush=True)
    spec = resolved_spec(builder)
    f_s, f_p = _funds(spec)
    out = {"cell": key, "headline": headline, "f_s_GHz": f_s / 1e9, "f_pump_GHz": f_p / 1e9}

    try:
        res, hier_nl = _emit_hier(spec, key, lib, top)
        out["emit_status"] = "ok"
        out["n_cells"] = res["n_cells"]
    except Exception as exc:  # noqa: BLE001
        out["emit_status"] = f"{type(exc).__name__}: {exc}"
        out["verdict"] = "EMIT_FAILED"
        out["parity_pass"] = False
        print(f"  HIER EMIT FAILED: {out['emit_status']}", flush=True)
        return out

    flat_deck = emit_flat_hb_netlist(spec)
    flattened = _flatten_hier(hier_nl)
    counts_flat = _type_counts(parse_netlist_text(flat_deck).elements)
    counts_hier = _type_counts(parse_netlist_text(flattened).elements)
    out["element_counts_flat"] = counts_flat
    out["element_counts_hier"] = counts_hier
    out["element_counts_match"] = counts_flat == counts_hier

    sim = _sim_parity(spec, hier_nl, OUTPUTS / "sim" / key)
    out.update(sim)

    # Compare the flat HB result to the recorded reference value (6 sig figs).
    ref_v, _ref_fs = REFERENCE_VOUT_SIGNAL[key]
    out["Vout_signal_reference"] = f"{ref_v:.6e}"
    vflat = sim.get("Vout_signal_flat")
    matches_ref = (vflat == f"{ref_v:.6e}")
    if not matches_ref and vflat is not None:
        # tolerate the HB convergence floor at the regenerative edge
        try:
            matches_ref = abs(float(vflat) - ref_v) <= 1e-7 * max(1.0, ref_v)
        except ValueError:
            matches_ref = False
    out["matches_reference"] = matches_ref

    flat_vs_hier = (out.get("Vout_signal_flat") == out.get("Vout_signal_hier")
                    or sim.get("verdict") in ("IDENTICAL", "CLOSE"))
    out["parity_pass"] = bool(out["element_counts_match"] and flat_vs_hier and matches_ref)

    print(f"  element counts flat : {counts_flat}", flush=True)
    print(f"  element counts hier : {counts_hier}  "
          f"({'MATCH' if out['element_counts_match'] else 'MISMATCH'})", flush=True)
    print(f"  HB flat / hier      : {sim.get('flat_run')} / {sim.get('hier_run')}", flush=True)
    print(f"  Vout(sig) reference : {out['Vout_signal_reference']}", flush=True)
    print(f"  Vout(sig) flat      : {sim.get('Vout_signal_flat')}  "
          f"({'matches reference' if matches_ref else 'DIFFERS FROM REFERENCE'})", flush=True)
    print(f"  Vout(sig) hier      : {sim.get('Vout_signal_hier')}", flush=True)
    print(f"  max|dV| harmonics   : {sim.get('max_abs_dV')}  ({sim.get('verdict')})", flush=True)
    print(f"\n  >>> PARITY {key}: {'PASS' if out['parity_pass'] else 'REVIEW'}", flush=True)
    return out


def mode_parity(args):
    target = args[0] if args else "all"
    keys = [k for k, *_ in EXAMPLES] if target == "all" else [target]
    RESULTS.mkdir(parents=True, exist_ok=True)
    results = {}
    for k in keys:
        r = _parity_one(k)
        results[k] = r
        (RESULTS / f"parity_{k}.json").write_text(json.dumps(r, indent=2, default=str),
                                                  encoding="utf-8")
    print("\n=== PARITY SUMMARY ===", flush=True)
    all_pass = True
    for k in keys:
        r = results[k]
        all_pass = all_pass and r.get("parity_pass")
        print(f"  {k:9s}: emit={r.get('emit_status')} "
              f"counts={'MATCH' if r.get('element_counts_match') else 'MISMATCH'} "
              f"verdict={r.get('verdict')} pass={r.get('parity_pass')}", flush=True)
    print(f"\n  >>> ALL PARITY {'PASS' if all_pass else 'REVIEW'}", flush=True)
    return 0 if all_pass else 1


# --------------------------------------------------------------------------- #
# shots -- render the unit-cell + top-level drawings AND run the OA overlap scan
# (the drawing-quality gate; requires ADS)
# --------------------------------------------------------------------------- #
def mode_shots(args):
    import _render
    import oa_overlap_scan as scan

    target = args[0] if args else "all"
    keys = [k for k, *_ in EXAMPLES] if target == "all" else [target]
    shots_dir = RESULTS / "screenshots"
    shots_dir.mkdir(parents=True, exist_ok=True)
    overlap = {}
    all_clean = True
    for key in keys:
        _k, builder, lib, top, headline = _BY_KEY[key]
        spec = resolved_spec(builder)
        res, _ = _emit_hier(spec, key, lib, top)
        ws = res["workspace_path"]
        per_level = {}
        for tag, design, label_terms in (("unit_cell", f"{lib}:unit_cell:schematic", True),
                                         ("top_level", f"{lib}:{top}:schematic", False)):
            geo = _render.read_geometry(ws, design)
            rep = scan.scan(geo)
            per_level[tag] = rep["strict_total"]
            _render.draw(geo, shots_dir / f"{key}_{tag}.png",
                         f"{key}: {tag} -- {len(geo['insts'])} insts, {len(geo['wires'])} wires "
                         f"[strict overlaps={rep['strict_total']}]",
                         label_terms=label_terms)
            print(scan.summarize(rep, label=f"{key} {tag}"), flush=True)
        clean = per_level["unit_cell"] == 0 and per_level["top_level"] == 0
        all_clean = all_clean and clean
        overlap[key] = {**per_level, "clean": clean}
        print(f"  {key}: rendered + scanned (uc={per_level['unit_cell']} "
              f"top={per_level['top_level']}) -> {'CLEAN' if clean else 'OVERLAP'}", flush=True)
    RESULTS.mkdir(parents=True, exist_ok=True)
    (RESULTS / "overlap_scan.json").write_text(json.dumps(overlap, indent=2), encoding="utf-8")
    print(f"\n  wrote screenshots to {shots_dir}; overlap scan -> {RESULTS / 'overlap_scan.json'}",
          flush=True)
    print(f"  >>> ALL DRAWINGS {'CLEAN (0 strict overlaps)' if all_clean else 'REVIEW -- overlaps found'}",
          flush=True)
    return 0 if all_clean else 1


# --------------------------------------------------------------------------- #
# assemble -- one openable ADS workspace, one library per example (requires ADS)
# --------------------------------------------------------------------------- #
_GITIGNORE = """\
# Generated OpenAccess workspace internals. Regenerate this workspace with
#   <ADS_PYTHON> build_examples.py assemble
# The text deliverables -- each folder's README.md and flat_hb_deck.ckt -- are the
# reviewable record; the parity proofs live in ../results/.
*.oa
*.db
*.dm
*.tag
*.cfg
*.prf
*.log
*.opt
*.wrkstate
cds.lib
lib.defs
data/
"""


def _proxy_facts(key):
    """Operating-point + predicted/realized facts for a proxy design (analysis JSON)."""
    path = RESULTS / f"proxy_analysis_{key}.json"
    if not path.exists():
        return None
    d = json.loads(path.read_text(encoding="utf-8"))
    rz = d.get("realized", {})
    return {
        "op_dBm": d.get("operating_pump_dBm"),
        "pred_dB": d.get("prediction", {}).get("analytical_coherent_peak_dB"),
        "boost_dB": rz.get("boost_peak_dB"), "boost_phase": rz.get("boost_peak_phase_deg"),
        "abs_dB": rz.get("abs_out_in_dB"),
        "onset_dBm": d.get("nonconvergence_onset_dBm"),
        "bounded": d.get("stability", {}).get("bounded_lobe"),
    }


def _folder_readme(key, lib, top, headline):
    if key.startswith("proxy_"):
        return _proxy_folder_readme(key, lib, top, headline)
    return (
        f"# {lib}\n\n**{headline}**\n\n"
        f"A clean hierarchical design: a `unit_cell` subcircuit (one cell's elements plus an\n"
        f"IN/OUT symbol), cascaded as `unit_cell:Xk` instances on the `{top}` top-level\n"
        f"schematic, on the accurate distributed terminations (ADS `VIAGND` physical via and\n"
        f"`MLEF` open-end fringe).\n\n"
        f"## Open\n\n"
        f"Open the workspace `{WORKSPACE.name}` in ADS; in the library browser open\n"
        f"**{lib} > {top} > schematic** for the top-level cascade (the HarmonicBalance\n"
        f"controller is on it), or **{lib} > unit_cell > schematic** for one cell's internals.\n\n"
        f"`flat_hb_deck.ckt` is the equivalent standalone `hpeesofsim` deck (`Port:` sources)\n"
        f"for a headless run. The flat-vs-hierarchical parity proof is in the repository at\n"
        f"`examples/real_parts/results/parity_{key}.json`.\n\n"
        f"## Gain readout\n\n"
        f"The top schematic carries a `MEAS_GAIN` `MeasEqn` block: run Simulate, then plot\n"
        f"`G_mod_in_dB` (the degenerate full-family out/in) or `Gain_coherent_dB` (2-term).\n"
        f"The equations evaluate on the ADS harmonic-balance dataset — the realized record —\n"
        f"and reference the named boundary nets `n_input` / `n_output`.\n"
    )


def _proxy_folder_readme(key, lib, top, headline):
    f = _proxy_facts(key) or {}
    op = f.get("op_dBm")
    pred = f.get("pred_dB")
    boost = f.get("boost_dB")
    phase = f.get("boost_phase")
    absdb = f.get("abs_dB")
    onset = f.get("onset_dBm")
    netpos = (absdb is not None and absdb > 0.0)
    edge = (f"HB goes non-convergent at +{onset:g} dBm" if onset is not None
            else "the pump-on/off boost collapses one 3 dB step higher")
    extra = (
        "This design is net-**positive** in absolute out/in at its operating point -- a genuine "
        "stable parametric amplifier on the idealized proxy.\n"
        if netpos else
        "Absolute out/in stays net-negative (the idealized proxy still pays the cascade's "
        "passive loss); the pump-on/off boost is the parametric action.\n"
    )

    def _fmt(v, sign=True):
        if v is None:
            return "n/a"
        return f"{v:+.2f}" if sign else f"{v:.2f}"

    return (
        f"# {lib}\n\n**{headline}**\n\n"
        f"An **optimizer-picked** degenerate-parametric TVTL amplifier: the public optimizer\n"
        f"(`floquet-line-design optimize-amplifier --freq-ghz 2.0`, deterministic; this is one of\n"
        f"its three diversified picks) on the **low-parasitic proxy varactor**. A clean\n"
        f"hierarchical design: a `unit_cell` subcircuit (one cell's elements + an IN/OUT symbol)\n"
        f"cascaded as `unit_cell:Xk` on the `{top}` top schematic, on the accurate `VIAGND`\n"
        f"short termination, with single-stub input/output **matching networks** on the rail.\n\n"
        f"These matched designs are the real-world confirmation that the hierarchical emitter's\n"
        f"matching-network placement draws clean (the three real-part examples have edge\n"
        f"impedances that need no matching, so they never exercised that path): **0 strict\n"
        f"overlaps at unit cell and top level** -- input network left of `IPR_IN`, output network\n"
        f"right of `IPR_OUT`, load port rightmost, rail straight.\n\n"
        f"## Convergent operating point\n\n"
        f"Baked at pump **{_fmt(op)} dBm**, phase 0 -- a convergent, bounded operating point (HB\n"
        f"converges and the degenerate-PSA phase lobe is bounded; {edge}). Predicted and realized\n"
        f"are reported **separately**:\n\n"
        f"| quantity | value |\n|---|---|\n"
        f"| analytical PREDICTION (coherent peak, optimistic ranking proxy) | {_fmt(pred)} dB |\n"
        f"| realized pump-on/off boost (ADS HB, peak phase {_fmt(phase, sign=False)} deg) | {_fmt(boost)} dB |\n"
        f"| realized absolute out/in | {_fmt(absdb)} dB |\n"
        f"| stability | bounded phase lobe (not an oscillation-edge runaway) |\n\n"
        f"The proxy varactor is idealized, so the realized number is a **faithfulness record** of\n"
        f"the emit path, not a real-world device-gain claim; the analytical prediction is\n"
        f"optimistic (omits loss / mismatch / large-signal). {extra}\n"
        f"## Open\n\n"
        f"Open the workspace `{WORKSPACE.name}` in ADS; open **{lib} > {top} > schematic** for the\n"
        f"top-level cascade (the HarmonicBalance controller + `MEAS_GAIN` block are on it), or\n"
        f"**{lib} > unit_cell > schematic** for one cell. `flat_hb_deck.ckt` is the equivalent\n"
        f"standalone `hpeesofsim` deck. The flat-vs-hierarchical parity proof (bit-identical\n"
        f"`V(n_output)`) is at `examples/real_parts/results/parity_{key}.json`, the operating-point\n"
        f"analysis at `examples/real_parts/results/proxy_analysis_{key}.json`.\n\n"
        f"## Gain readout\n\n"
        f"Run Simulate, then plot `G_mod_in_dB` (degenerate full-family out/in) or `Gain_coherent_dB`\n"
        f"(2-term). To trace the phase lobe, edit the `PUMP_SRC` phase on the top schematic.\n"
    )


def _top_readme():
    real_rows = "\n".join(
        f"| `{lib}` | {headline} |" for (_k, _b, lib, _t, headline) in _EXAMPLES_REAL
    )
    proxy_rows = "\n".join(
        f"| `{lib}` | {headline} |" for (_k, _b, lib, _t, headline) in _EXAMPLES_PROXY
    )
    return (
        "# TVTL example workspace (six clean hierarchical designs)\n\n"
        "One openable ADS workspace, one library folder per example. Each example is a clean\n"
        "hierarchical design: a `unit_cell` subcircuit (one cell's elements + an IN/OUT symbol),\n"
        "cascaded as `unit_cell:Xk` instances on the top-level schematic, on the accurate\n"
        "distributed terminations (ADS `VIAGND` physical via + `MLEF` open-end fringe).\n\n"
        "## Three real-part designs (commercial varactor SPICE)\n\n"
        "| library | design |\n|---|---|\n" + real_rows + "\n\n"
        "## Three optimizer-picked proxy designs (low-parasitic proxy varactor + matching)\n\n"
        "The public optimizer (`floquet-line-design optimize-amplifier --freq-ghz 2.0`,\n"
        "deterministic) picks these three diversified designs -- distinct `(stub-count x\n"
        "termination)` cells the real-part trio does not cover. They are the FIRST examples to\n"
        "carry input/output single-stub **matching networks** (the trio's edge impedances need\n"
        "none), so they confirm the hierarchical emitter places matching networks cleanly\n"
        "outside the generalization test. Each is baked at its convergent, bounded operating\n"
        "point; predicted (analytical) and realized (ADS HB) gain are reported separately, and\n"
        "the idealized proxy's realized number is a faithfulness record, not a device-gain claim.\n\n"
        "| library | design |\n|---|---|\n" + proxy_rows + "\n\n"
        "## Gates\n\n"
        "Every example passes the electrical-identity parity gate: the hierarchical design's\n"
        "signal-frequency output voltage equals the flat reference deck's (bit-identical for all\n"
        "but the near-oscillation SMV2020 point, which agrees to the harmonic-balance convergence\n"
        "floor). The MAVR COMBINED parallel-trap fork renders correctly here even though it\n"
        "collapses in the flat OA schematic emitter, because the hierarchical unit-cell subcircuit\n"
        "carries no source and ADS does not recompute its connectivity from geometry. Every\n"
        "drawing -- unit cell and top level, including the three matched proxy designs -- passes\n"
        "the OA overlap scan with 0 strict overlaps. See `../results/parity_*.json`,\n"
        "`../results/proxy_analysis_*.json`, and `../results/overlap_scan.json`.\n\n"
        "Open this workspace in ADS, then a library's `<top> > schematic` for the cascade or\n"
        "`unit_cell > schematic` for one cell. Regenerate with `build_examples.py assemble`.\n"
    )


def _embed_gain_equations(ws, lib, top, *, p_avs_W):
    """Bake the MEAS_GAIN MeasEqn block onto an emitted hierarchical top schematic.

    The gain equations reference ``n_input`` / ``n_output``, but the hierarchical
    emitter auto-names its boundary nets. Renaming the Num=1 (signal) and Num=3
    (output) port nets to ``n_input`` / ``n_output`` carries those names into the
    generated netlist and the ``.ds`` dataset the MeasEqn evaluates on (the
    wire-label is added too, for the GUI node picker). A GUI ``Simulate`` then reads
    ``G_mod_in_dB`` / ``Gain_coherent_dB`` straight off the schematic. The equations
    evaluate on the ADS harmonic-balance dataset, i.e. the realized record.

    Per-cell branch-power equations are intentionally NOT added: in the hierarchical
    form the per-cell nodes live inside the ``unit_cell`` subcircuits, so they are not
    referenceable from the top level by a flat ``n_c<k>_b`` name.
    """
    import keysight.ads.de as de
    from keysight.ads.de import db_uu as db

    from tvtl_ads_builder_v2.oa.gain_equations import meas_gain_equations
    from tvtl_ads_builder_v2.oa.oa_emitter import _add_wire_label

    if de.workspace_is_open():
        de.close_workspace()
    wk = de.open_workspace(str(ws))
    wk.open_library(lib, str(Path(ws) / lib), de.LibraryMode.NON_SHARED)
    design = db.open_design(f"{lib}:{top}:schematic", mode=db.DesignMode.APPEND)

    # Ground nets to skip when choosing each source/output PORT pin to rename. The
    # literal global ground "0" is not enough: the hierarchical emitter grounds every
    # source RETURN through a coincident GROUND symbol whose net is auto-named (e.g.
    # "N__102"), not "0". A name=="0" test alone therefore lands on the return pin and
    # mislabels the source RETURN as n_input instead of the driven input node. Collect
    # every net a GROUND symbol sits on so the rename skips returns and names the real
    # port node (the output P_OUT already grounds via the literal node "0", so it was
    # always correct; this also fixes the input).
    ground_net_names = {"0"}
    for inst in design.instances:
        if str(getattr(inst, "cell_name", "")) == "GROUND":
            for pin in inst.inst_pins:
                net = getattr(pin, "net", None)
                if net is not None:
                    ground_net_names.add(str(getattr(net, "name", "")))

    targets = {"1": "n_input", "3": "n_output"}
    renamed: list[str] = []
    for inst in design.instances:
        params = getattr(inst, "parameters", {})
        if "Num" not in params:
            continue
        num = str(params["Num"].value).strip().strip('"')
        if num not in targets:
            continue
        for pin in inst.inst_pins:
            net = getattr(pin, "net", None)
            if net is None or str(getattr(net, "name", "")) in ground_net_names:
                continue
            try:
                net.name = targets[num]
                _add_wire_label(net, targets[num])
                renamed.append(targets[num])
            except Exception:  # noqa: BLE001
                pass
            break

    gain_eqs = list(meas_gain_equations(p_avs_W))
    m1 = design.add_instance(("ads_simulation", "MeasEqn", "symbol"), (40.0, 20.0), name="MEAS_GAIN")
    if "Meas" in m1.parameters:
        m1.parameters["Meas"].value = gain_eqs
    design.save_design()
    # MeasEqn is a measurement directive, not a circuit element, so it is verified on
    # the design instances (it does not appear in the circuit netlist); the named
    # boundary nets it references are verified on the generated netlist.
    n_measeqn = sum(1 for i in design.instances if getattr(i, "cell_name", None) == "MeasEqn")
    text = str(design.generate_netlist())
    if de.workspace_is_open():
        de.close_workspace()
    return {
        "renamed_nets": sorted(set(renamed)),
        "n_gain_eqs": len(gain_eqs),
        "n_output_named": "n_output" in text,
        "n_input_named": "n_input" in text,
        "measeqn_added": n_measeqn,
    }


def _signal_p_avs_W(spec) -> float:
    """Available signal power [W] from the Num=1 source (the absolute-gain denominator)."""
    sig_dbm = next((float(s.power_dBm) for s in spec.sources
                    if getattr(s, "num", None) == 1), -30.0)
    return 10.0 ** ((sig_dbm - 30.0) / 10.0)


def mode_assemble(_args):
    import keysight.ads.de as de

    from tvtl_ads_builder_v2.netlist import emit_flat_hb_netlist
    from tvtl_ads_builder_v2.oa.oa_hier_emitter import emit_hierarchical_workspace_from_spec

    scratch = OUTPUTS / "assemble_scratch"
    if scratch.exists():
        shutil.rmtree(scratch, ignore_errors=True)
    scratch.mkdir(parents=True, exist_ok=True)
    # ignore_errors: outputs/ is git-ignored regenerable scratch, and a stale ADS
    # bridge backend can hold a handle on a previously-opened workspace dir on
    # Windows. Tolerate that rather than hard-crash; create_workspace re-creates it.
    if WORKSPACE.exists():
        shutil.rmtree(WORKSPACE, ignore_errors=True)

    # STEP 1 -- emit each example hierarchically into its own throwaway workspace.
    # (the emitter closes the global workspace, so the master cannot stay open
    # across emits -- emit all first, then assemble the master.)
    print("=== STEP 1: emit each example hierarchically ===", flush=True)
    emitted = []
    for key, builder, lib, top, headline in EXAMPLES:
        spec = resolved_spec(builder)
        res = emit_hierarchical_workspace_from_spec(
            spec, scratch / f"emit_{key}" / "ws", library_name=lib,
            top_cell_name=top, uc_cell_name="unit_cell")
        eq = _embed_gain_equations(res["workspace_path"], lib, top,
                                   p_avs_W=_signal_p_avs_W(spec))
        lib_src = Path(res["workspace_path"]) / lib
        flat_deck = emit_flat_hb_netlist(spec)
        emitted.append((key, lib, top, headline, lib_src, flat_deck))
        print(f"  {lib}: hier-emitted (n_cells={res['n_cells']}); "
              f"gain eqns embedded {eq}", flush=True)

    # STEP 2 -- assemble the master workspace referencing all emitted libraries.
    print(f"\n=== STEP 2: assemble {WORKSPACE} ===", flush=True)
    if de.workspace_is_open():
        de.close_workspace()
    workspace = de.create_workspace(str(WORKSPACE))
    workspace.open()
    for key, lib, top, headline, lib_src, flat_deck in emitted:
        dest = WORKSPACE / lib
        shutil.copytree(lib_src, dest)
        (dest / "flat_hb_deck.ckt").write_text(flat_deck, encoding="utf-8")
        (dest / "README.md").write_text(_folder_readme(key, lib, top, headline), encoding="utf-8")
        workspace.add_library(lib, str(dest), de.LibraryMode.NON_SHARED)
        print(f"  add_library -> {lib}", flush=True)
    try:
        if hasattr(workspace, "save"):
            workspace.save()
    except Exception as exc:  # noqa: BLE001
        print(f"  workspace.save note: {type(exc).__name__}: {exc}", flush=True)
    de.close_workspace()

    (WORKSPACE / ".gitignore").write_text(_GITIGNORE, encoding="utf-8")
    (WORKSPACE / "README.md").write_text(_top_readme(), encoding="utf-8")
    libdefs = WORKSPACE / "lib.defs"
    print(f"\n=== lib.defs ===\n{libdefs.read_text() if libdefs.exists() else '(none)'}", flush=True)
    print(f"  assembled workspace at {WORKSPACE}", flush=True)
    for p in sorted(WORKSPACE.iterdir()):
        print(f"    {p.name}{'/' if p.is_dir() else ''}", flush=True)
    return 0


def mode_all(args):
    rc = mode_parity(args)
    mode_shots(args)
    mode_assemble(args)
    return rc


_MODES = {
    "specs": mode_specs,
    "parity": mode_parity,
    "shots": mode_shots,
    "assemble": mode_assemble,
    "all": mode_all,
}

if __name__ == "__main__":
    mode = sys.argv[1] if len(sys.argv) > 1 else "specs"
    args = sys.argv[2:]
    fn = _MODES.get(mode)
    if fn is None:
        print(f"unknown mode {mode!r}; use {sorted(_MODES)}")
        raise SystemExit(2)
    raise SystemExit(fn(args))
