"""Integrity guard for the imported 2.0 GHz length-sweep record.

Parses the imported primary table
``results/length_sweep/sweep_table_primary.csv`` (candidate
``proxy_n3_single_short``), resolving column names from the header, and pins the
study's headline classification so the imported record cannot silently drift:

  * exactly 9 N-points spanning 6..30 (step 3);
  * N=15 is the ROBUST optimum: boost 8.18 +/- 0.01 dB, absolute out/in
    1.18 +/- 0.01 dB, operating pump +15 dBm;
  * the higher-gain N=12 and N=18 rows are fragile SPIKEs.

Realized numbers are the idealized-proxy-varactor faithfulness record (not a
real-world device gain); see results/length_sweep/PROVENANCE.md.
"""

from __future__ import annotations

import csv
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
CSV_PATH = REPO_ROOT / "results" / "length_sweep" / "sweep_table_primary.csv"

# Roles mapped to the study CSV column names (resolved against the header below).
COL_N = "N"
COL_BOOST = "boost_peak_dB"
COL_ABS = "abs_out_in_dB"
COL_PUMP = "op_pump_dBm"
COL_CLASS = "sharpness_class"


def _load() -> tuple[list[str], list[dict[str, str]]]:
    with open(CSV_PATH, newline="", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        header = reader.fieldnames or []
        rows = [dict(r) for r in reader]
    return header, rows


def test_length_sweep_primary_record_integrity(capsys) -> None:
    header, rows = _load()

    # Resolve column names from the header and report the mapping.
    mapping = {
        "N": COL_N,
        "boost": COL_BOOST,
        "abs_out_in": COL_ABS,
        "op_pump": COL_PUMP,
        "class": COL_CLASS,
    }
    for role, col in mapping.items():
        assert col in header, f"column for {role!r} ({col!r}) not in header {header}"
    with capsys.disabled():
        print("[length-sweep] column mapping: "
              + ", ".join(f"{role}->{col}" for role, col in mapping.items()))

    by_n = {int(float(r[COL_N])): r for r in rows}

    # Exactly 9 N-points spanning 6..30 (step 3).
    ns = sorted(by_n)
    assert ns == [6, 9, 12, 15, 18, 21, 24, 27, 30], ns
    assert len(ns) == 9 and ns[0] == 6 and ns[-1] == 30

    # N=15 is the robust optimum with the recorded operating point.
    r15 = by_n[15]
    assert r15[COL_CLASS].strip() == "robust", r15[COL_CLASS]
    assert float(r15[COL_BOOST]) == pytest.approx(8.18, abs=0.01)
    assert float(r15[COL_ABS]) == pytest.approx(1.18, abs=0.01)
    assert float(r15[COL_PUMP]) == pytest.approx(15.0, abs=1e-9)

    # The higher-gain N=12 and N=18 lengths are fragile SPIKEs.
    assert by_n[12][COL_CLASS].strip() == "SPIKE", by_n[12][COL_CLASS]
    assert by_n[18][COL_CLASS].strip() == "SPIKE", by_n[18][COL_CLASS]
