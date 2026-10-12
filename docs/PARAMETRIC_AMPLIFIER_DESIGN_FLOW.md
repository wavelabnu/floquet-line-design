# Parametric Amplifier Design Flow

The paper-target flow is:

1. Use TVTL static/cascade evaluation to screen passive unit-cell structure.
2. Introduce a static passive `C0 + Rser` branch and compare observables for one UC.
3. Repeat the static passive branch across multiple UCs to check topology composition before nonlinear work.
4. Use the generic TVTL design-search layer to rank parametric unit-cell candidates by proxy metrics.
5. Represent the nonlinear capacitor mathematically as `C(V)` / `Q(V)` and select an ADS realization by analysis mode.
6. Represent native ADS `SDD2P` in a flat single-UC static/small-signal scaffold only when explicitly requested, then review the component convention.
7. Move to ADS SDD small-signal validation for component convention checks.
8. Run ADS-HB engagement diagnostics and report `Vcap`, `Icap`, `Pcap`, pump controls, and convergence controls.
9. Promote a fabrication candidate only when TVTL ranking and ADS validation both support the selection.

No full gain claim is included in this repository without validation artifacts.

The current TVTL theory, Floquet, and optimization layers are generic production code, while examples remain data. Objective metrics are proxies such as passive insertion loss, match penalty, local sideband coupling, phase matching, conditioning, voltage/current safety, and fabrication penalties. They are useful for search triage but are not full-device gain or amplifier proof.

A typical public search script samples unit-cell geometry, nonlinear-capacitor coefficients, pump/signal frequencies, and pump-voltage proxy deterministically, then writes ranking artifacts under an explicit output directory. TVTL computes gain proxies such as nonlinear coupling, phase matching, pump delivery, passive insertion loss, match penalties, conditioning, and safety penalties. These are optimization proxies only; ADS-HB gain and full-device gain require local validation artifacts.

For nonlinear capacitance, `sparam_static` uses native ADS `C=C0` plus optional `Rser`. Static dry-run manifests record `static_C_F`, `static_Rser_ohm`, `emitted_ads_components`, and `sdd2p_emitted=false` to make this realization auditable. `sdd_static_scaffold` and future `hb_nonlinear` modes use native `ads_datacmps:SDD2P` with the `I[1,1]=Q(_v1)` convention. The manual golden-cell path is unavailable until a reviewed ADS component is ingested and marked validated.

---

**Related:** the physics behind each stage → [theory.md](theory.md); the optimizer that implements the ranking → [methodology/OPTIMIZER_METHODOLOGY.md](methodology/OPTIMIZER_METHODOLOGY.md); running the flow → [workflow.md](workflow.md).


