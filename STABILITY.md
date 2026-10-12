# Public Stability Notes

The public API is intentionally small.

## Frozen paper surface

For the v1.0.1 paper record (the release the paper cites), the supported import and data surface is:

- `tvtl.design.feasibility.feasibility`, `candidate_cell_factory` and the constants `T_REQ`, `T_REQ_SCREEN_2026_09`, `ALPHA_MAX` for the passive Stage 0 gate (delivery, pump and signal transport; optional phase margin, depth ceiling);
- `tvtl.design.depth_ceiling` for the delivered-depth ceiling and `tvtl_ads_builder_v2.validation.dc_return` for the varactor DC-return check;
- `tvtl_ads_builder_v2.analytical.delivered_gain.delivered_gain`, `delivered_pump_profile`, `modulation_profile` and `TeeModelOutOfRange` for the stated analytical delivery quantities;
- `tvtl_ads_builder_v2.analytical.cascade_budget` for the cascade layer (phase mismatch, `kappa/m`, `m_EP`, `m*`);
- `tvtl_ads_builder_v2.analytical.floquet_stub.Y_stub_at_freq_with_shunts` and `build_stub_y_fn_with_mid_shunts` for the stub with mid-stub shunts;
- `tvtl_ads_builder_v2.oa.certification.GainRecord` for the ADS-realized certification fields and checks;
- the six frozen example records under `examples/real_parts/results/` (three real-part and three proxy cases), whose published JSON bytes are covered by drift guards;
- `scripts/` for the paper's figure inputs and numbers (`scripts/README.md`).

This freeze covers the documented signatures and recorded data for the paper. It does not turn an analytical
estimate into an ADS or measurement claim. Since 1.0.1 the release includes the intermediate-shunt stub
functions of `floquet_stub.py`, which the two-filter sweep and the dense map behind Figs. 2 and 7 use.

Stable analytical imports live under `tvtl` and cover ABCD primitives (`tvtl.line_abcd`, `tvtl.cascade_abcd`, `tvtl.series_impedance_abcd`, `tvtl.shunt_admittance_abcd`; the raw matrix helpers in `tvtl.abcd.matrices`), microstrip estimates, Floquet/harmonic utilities, staged design helpers, and varactor models.

Stable ADS-builder imports live under `tvtl_ads_builder_v2` and cover device specs, flat netlist emission, OpenAccess workspace emission, content checks, parser helpers, public model-registry loading, and selected parity utilities.

Internal modules may change between supplementary-material releases. Prefer the package-level APIs and documented subpackage exports when building new examples.

The legacy v1 ADS builder is not part of this public repository. Private diagnostics, generated workspaces, local scratch outputs, and unpublished reconstruction material are also intentionally excluded.

ADS/OpenAccess/HB validation requires a local Keysight ADS installation. Optional reference workspaces, surrogate tables, and generated outputs must be provided with environment variables or explicit command-line paths.

---

**Related:** building on these stable APIs to add a varactor, topology, or analysis → [docs/extending.md](docs/extending.md).


