"""Byte-identity gate: emitted flat-HB decks must match the committed goldens.

Each of the six worked examples (three real-part designs + three optimizer-picked
proxy designs) ships a canonical ``<key>_flat_hb_deck.ckt`` under
``examples/real_parts/results/decks/``. This asserts the current emitter still
reproduces every one of them byte-for-byte (SHA-256), so any drift in the flat
netlist -- element order, parameter values, spacing, encoding -- is caught in the
default suite. Deck emission is pure Python (``emit_flat_hb_netlist``); no Keysight
ADS is required, and all six specs build + emit in well under a second.

Line endings are canonicalized to LF before hashing. The repository is committed
under ``core.autocrlf`` normalization (the decks are stored as LF in git and
checked out as CRLF on Windows), while ``emit_flat_hb_netlist`` returns LF text.
Canonicalizing both sides to LF compares the git-canonical deck content and keeps
the gate platform-independent; every non-newline byte is still compared strictly.

This is the encoded form of the previously script-only invariant -- unlike
``build_examples.py specs`` (which *overwrites* the goldens in place), this test
only reads them, so a regression fails the suite instead of silently rewriting the
reference.
"""
from __future__ import annotations

import hashlib
import sys
from pathlib import Path

import pytest

_REPO_ROOT = Path(__file__).resolve().parents[1]
_EXAMPLES_DIR = _REPO_ROOT / "examples" / "real_parts"
_DECKS_DIR = _EXAMPLES_DIR / "results" / "decks"

# The design registries live beside ``build_examples.py`` (not an installed
# package); make them importable exactly as that script does.
if str(_EXAMPLES_DIR) not in sys.path:
    sys.path.insert(0, str(_EXAMPLES_DIR))

from real_part_designs import EXAMPLES as _EXAMPLES_REAL, resolved_spec  # noqa: E402
from proxy_designs import EXAMPLES_PROXY as _EXAMPLES_PROXY  # noqa: E402
from tvtl_ads_builder_v2.netlist import emit_flat_hb_netlist  # noqa: E402

# (key, spec-builder, ...) for all six examples -- the same registry, in the same
# order, that build_examples.py uses to write the committed decks.
_BUILDERS = {key: builder for (key, builder, *_rest) in (_EXAMPLES_REAL + _EXAMPLES_PROXY)}

_EXPECTED_KEYS = {
    "smv1245",
    "smv2020",
    "mavr",
    "proxy_n2_double_short",
    "proxy_n3_single_short",
    "proxy_n4_double_short",
}


def _sha256_lf(data: bytes) -> str:
    """SHA-256 of ``data`` with CRLF/CR line endings canonicalized to LF."""
    normalized = data.replace(b"\r\n", b"\n").replace(b"\r", b"\n")
    return hashlib.sha256(normalized).hexdigest()


def test_all_six_example_decks_registered():
    # Guard against a builder registry silently shrinking (which would make the
    # per-deck parametrization cover fewer examples without any test failing).
    assert set(_BUILDERS) == _EXPECTED_KEYS


@pytest.mark.parametrize("key", sorted(_EXPECTED_KEYS))
def test_flat_hb_deck_byte_identity(key):
    reference = _DECKS_DIR / f"{key}_flat_hb_deck.ckt"
    assert reference.is_file(), f"missing committed reference deck: {reference}"

    spec = resolved_spec(_BUILDERS[key])
    generated = emit_flat_hb_netlist(spec).encode("utf-8")

    assert _sha256_lf(generated) == _sha256_lf(reference.read_bytes()), (
        f"{key}: emitted flat-HB deck no longer matches committed golden "
        f"{reference.name} (byte-identity drift). If the change to the emitter is "
        f"intended, regenerate the deck with "
        f"'python examples/real_parts/build_examples.py specs' and review the diff."
    )
