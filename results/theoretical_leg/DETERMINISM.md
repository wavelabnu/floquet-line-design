# Theoretical-leg determinism digest

Determinism evidence for the flagship theoretical (predicted) leg artifact under `results/theoretical_leg/`.
It records the regeneration for release 1.0.1, a double-run byte-identity check and the comparison with the
1.0.0 tree. The drift-guard test `tests/test_theoretical_leg_digest.py` re-pins the committed text-artifact
SHA256 values from the block at the bottom of this file.

- **Date:** 2026-10-07 (release 1.0.1). This supersedes the digest of 2026-07-18.
- **Re-pinned:** 2026-10-12. The predicted-gain disclaimer (`PREDICTED_GAIN_QUALIFIER` in
  `src/tvtl/optimization/cli.py`) was reworded. `candidates.json`, `report.md` and `summary.json` change only
  in that string (5 lines); `candidates.csv` and the three PNGs are byte-identical. The double-run check
  below was repeated on 2026-10-12 with the same outcome.
- **Repo:** `floquet-line-design`, release 1.0.1.

## Exact command

This is the documented default from `docs/REPRODUCE.md`, run in place to regenerate the committed tree:

```powershell
.\.venv\Scripts\python.exe -m tvtl.optimization optimize-amplifier --freq-ghz 2.0 --out results\theoretical_leg
```

The double-run check ran the same command with `--out` pointed at two throwaway scratch directories.

Each run reported:

- `3247 screened, top-10 refined, 7 realizable, 3 selected`;
- best candidate `n2_double_short (coherent_peak=10.6 dB, Z_TL=61.3 ohm, main_width=2.49 mm)`.

## Environment

| | Python | numpy | scipy | matplotlib | pandas | PyYAML |
| --- | --- | --- | --- | --- | --- | --- |
| Base `.venv` (the recorded environment, `requirements-freeze.txt`) | 3.12.10 | 2.4.6 | 1.17.1 | 3.10.9 | 3.0.3 | 6.0.3 |

## Why 1.0.1 regenerates the tree

The 1.0.0 tree predated the gate fields that the optimizer has written since the delivery-core round. The
documented command had therefore stopped reproducing `candidates.csv`, `candidates.json` and `summary.json`.
Release 1.0.1 regenerates them.

- **Candidates.** The same three candidates are in the same order (`n2_double_short`, `n4_double_short`,
  `n3_single_short`), and every 1.0.0 value is unchanged.
- **Additions.** The added fields are listed in `CHANGELOG.md`. All three candidates pass the 1.0.1
  stage-0 gate.

## SHA256 per artifact (committed tree, checked out)

| Artifact | Bytes | SHA256 |
| --- | --- | --- |
| candidates.csv | 1635 | `57942633d71ee5691116c632b92f3a2b181b5546977a83b57d1a7652408c6755` |
| candidates.json | 62046 | `0674693f305b9fe4aadf023ef8fa358c81d54b64013897ce56ed1e6203a92bda` |
| report.md | 5779 | `d9bae82368e2fcc971d731a9fc30b850df91cb3f87c302da1c1692fd8fea4beb` |
| summary.json | 22953 | `f7e668c77e67f84685d9cb0334378508914f9070a75f0aac340d5c204551960c` |
| fig_predicted_nsweep.png | 70157 | `4c6191208073b1930ba9386a2e878fe4fd70f09461ad5c971a8880cd685a7e26` (unchanged) |
| fig_predicted_sparams.png | 55699 | `d015689fe6243069442329d947da4338547c17c3bd21de2128d0ea9e6226d7dd` (unchanged) |
| fig_topology.png | 22854 | `fee21726ce394f2d0d0c0799390772087a56f553eff72b72f84392630e91acde` (unchanged) |

The bytes the command writes equal the bytes git checks out. The `.gitattributes` `eol=crlf` pins cover
`candidates.csv`, `candidates.json`, `report.md` and `summary.json`.

## Double-run identity statement

Two independent runs went into two different scratch directories:

- `candidates.csv`, `candidates.json`, `report.md` and the three PNGs are **byte-identical** to each other
  and to the committed tree.
- `summary.json` differs between the two runs only in the four self-referential output-path fields
  (`files.candidates_csv`, `files.candidates_json`, `files.report_md`, `files.summary_json`). A structural
  JSON diff confirms that no other field differs.

**Conclusion:** the optimizer is deterministic (seed 0). Holding the output path fixed, every artifact
reproduces byte-for-byte.

## PNG note

PNG bytes depend on the environment (matplotlib and font rendering). Under the recorded environment they
reproduce byte-for-byte. The drift-guard test pins only the four text artifacts.

## Drift-guard pins

`tests/test_theoretical_leg_digest.py` recomputes the SHA256 of the four committed text artifacts and
asserts they equal the values pinned in the block below (machine-readable; do not reformat the lines
between the markers).

<!-- DRIFT-GUARD-PINS-START -->
```
candidates.csv sha256 = 57942633d71ee5691116c632b92f3a2b181b5546977a83b57d1a7652408c6755
candidates.json sha256 = 0674693f305b9fe4aadf023ef8fa358c81d54b64013897ce56ed1e6203a92bda
report.md sha256 = d9bae82368e2fcc971d731a9fc30b850df91cb3f87c302da1c1692fd8fea4beb
summary.json sha256 = f7e668c77e67f84685d9cb0334378508914f9070a75f0aac340d5c204551960c
```
<!-- DRIFT-GUARD-PINS-END -->
