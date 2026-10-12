"""Drift guard for the committed theoretical-leg text artifacts.

Recomputes the SHA256 of the four committed text artifacts under
``results/theoretical_leg/`` and asserts they equal the values pinned in
``results/theoretical_leg/DETERMINISM.md`` (the determinism digest). If either
the artifacts or the pinned digest drift out of sync, this test fails.

The four text artifacts reproduce byte-for-byte on re-run (seed 0); see
``DETERMINISM.md`` for the double-run identity check and the committed-tree
comparison. PNGs are environment-dependent and are intentionally not pinned.
"""

from __future__ import annotations

import hashlib
import re
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
LEG_DIR = REPO_ROOT / "results" / "theoretical_leg"
DETERMINISM_MD = LEG_DIR / "DETERMINISM.md"

TEXT_ARTIFACTS = ("report.md", "candidates.csv", "candidates.json", "summary.json")

_PIN_BLOCK = re.compile(
    r"<!-- DRIFT-GUARD-PINS-START -->(.*?)<!-- DRIFT-GUARD-PINS-END -->",
    re.DOTALL,
)


def _pinned_hashes() -> dict[str, str]:
    text = DETERMINISM_MD.read_text(encoding="utf-8")
    block = _PIN_BLOCK.search(text)
    assert block is not None, "DETERMINISM.md is missing the DRIFT-GUARD-PINS block"
    pins: dict[str, str] = {}
    for name in TEXT_ARTIFACTS:
        match = re.search(
            rf"^{re.escape(name)}\s+sha256\s*=\s*([0-9a-f]{{64}})\s*$",
            block.group(1),
            re.MULTILINE,
        )
        assert match is not None, f"DETERMINISM.md pin block has no SHA256 for {name}"
        pins[name] = match.group(1)
    return pins


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_committed_theoretical_leg_text_artifacts_match_pinned_digest() -> None:
    """Committed text artifacts must match the SHA256 pins in DETERMINISM.md."""

    pins = _pinned_hashes()
    mismatches = []
    for name in TEXT_ARTIFACTS:
        artifact = LEG_DIR / name
        assert artifact.is_file(), f"missing committed artifact: {artifact}"
        actual = _sha256(artifact)
        if actual != pins[name]:
            mismatches.append(f"{name}: committed={actual} pinned={pins[name]}")
    assert not mismatches, "theoretical-leg artifact drift detected:\n" + "\n".join(mismatches)
