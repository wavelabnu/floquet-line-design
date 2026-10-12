"""OA-workspace embedded ``MeasEqn`` gain/per-cell equation set.

GUI-evaluated ADS ``MeasEqn`` power-ratio equations baked into emitted OA
workspaces, so a human running *Simulate* reads the gain definitions directly off
the schematic. They are GUI dataflow expressions -- ``mix(net,{m,n})``,
``mag``/``real``/``conj``/``log10``, power ``= 0.5*mag(V)*mag(V)/50`` (never
``**`` / ``^``) -- evaluated on the manual Simulate, NOT headlessly.

Degenerate self-pump (``f_pump = 2 f_signal``): a product ``(m, n)`` lands on
``f_signal`` when ``m + 2n = 1``. The family in the default HB truncation
(Order 5/5/6) is :data:`FOLDING_FAMILY` = ``(1,0),(-1,1),(3,-1),(-3,2)``.

The equation set carries, all clearly labelled:

* **OURS** (the physically-complete on-``f_s`` family, SA-comparable):
  ``Pout_mod`` = phasor sum of the whole ``m+2n=1`` family; ``G_mod_in_dB`` =
  ``10*log10(Pout_mod / P_avs)``. The pump-on/off headline ``G_mod_unmod`` is a
  TWO-run quantity (pump-on at the baked optimal phase minus pump-off) -- see the
  per-workspace README; the single-dataset block here supplies ``G_mod_in_dB`` and
  ``Gain_sig_dB`` whose two-run difference IS ``G_mod_unmod``.
* **Reference/comparison convention** (the alternative gain convention, for
  direct comparison): ``Gain_ref_dB`` = the reference convention's ``P_out_tot``
  (signal+idler) over ``P_avs`` -- which equals OUR 2-term coh_pk numerator /
  ``Gain_coherent_dB``; and ``Gain_4f_dB`` = its ``P_out_tot_4f``
  (``(1,0)+(0,1)+(-1,1)+(1,-1)``) over ``Pin_tot_4f``. That 4-frequency total
  INCLUDES the **pump-frequency** term ``(0,1)`` at ``2 f_s`` (NOT on ``f_s``) and
  the ``(1,-1)`` term at ``-f_s`` -- the key gain-definition reconciliation point
  flagged in the README.
* the existing **1a/1b** signal-band transducer (``Gain_sig_dB``) and the 2-term
  coherent (``Gain_coherent_dB``) building blocks, plus ratio cross-checks.

``MeasEqn`` evaluates the list top-to-bottom and a later equation may reference an
earlier one, so the ordering below is dependency-safe.
"""

from __future__ import annotations

__all__ = ["FOLDING_FAMILY", "DEFAULT_P_AVS_W", "meas_gain_equations", "percell_equations"]

# The degenerate folding family on f_signal (m + 2n = 1) present in the default
# HB truncation (Order[1]=5 Order[2]=5 MaxOrder=6: |m|+|n| <= 6). Verified live in
# the full-harmonic-sum study -- the next folding term (5,-2) is order 7, truncated.
FOLDING_FAMILY: tuple[tuple[int, int], ...] = ((1, 0), (-1, 1), (3, -1), (-3, 2))

# Available signal power for the operating point the workspaces bake (signal at
# -30 dBm): dbmtow(-30) = 1e-6 W.
DEFAULT_P_AVS_W: float = 1.0e-6


def meas_gain_equations(p_avs_W: float = DEFAULT_P_AVS_W) -> list[str]:
    """Return the ``MEAS_GAIN`` ``MeasEqn`` equation list (``"name=expr"`` strings).

    ``p_avs_W`` is the available input signal power baked as the absolute-gain
    denominator (default ``dbmtow(-30)`` = 1e-6 W, the operating point).
    """

    return [
        # ----- available input signal power; the absolute-gain denominator -----
        f"P_avs={p_avs_W:.6e}",
        # ----- node phasors at every on-f_s folding mix (m+2n=1) + the input -----
        "Vout_sig=mix(n_output,{1,0})",        # signal  (1,0)
        "Vout_idler=mix(n_output,{-1,1})",     # idler   (-1,1)
        "Vout_3m1=mix(n_output,{3,-1})",       # folding (3,-1)
        "Vout_m32=mix(n_output,{-3,2})",       # folding (-3,2)
        "Vin_sig=mix(n_input,{1,0})",
        # ----- per-term + 2-term output powers (existing building blocks) -----
        "Pout_sig=0.5*mag(Vout_sig)*mag(Vout_sig)/50",
        "Pout_idler=0.5*mag(Vout_idler)*mag(Vout_idler)/50",
        "Pout_coherent=0.5*mag(Vout_sig+Vout_idler)*mag(Vout_sig+Vout_idler)/50",
        # ----- OURS: full m+2n=1 family sum on f_s (the SA-comparable signal power) -----
        "Vout_mod=Vout_sig+Vout_idler+Vout_3m1+Vout_m32",
        "Pout_mod=0.5*mag(Vout_mod)*mag(Vout_mod)/50",
        # ----- absolute gains vs P_avs -----
        "Gain_sig_dB=10*log10(Pout_sig/P_avs)",            # 1a/1b signal-band transducer
        "Gain_coherent_dB=10*log10(Pout_coherent/P_avs)",  # 2-term coherent absolute
        "G_mod_in_dB=10*log10(Pout_mod/P_avs)",            # OURS absolute full-sum
        # ----- Reference convention: "Gain" = (signal+idler) over P_avs (== Gain_coherent) -----
        "Pout_tot_aig=0.5*mag(Vout_sig+Vout_idler)*mag(Vout_sig+Vout_idler)/50",
        "Gain_ref_dB=10*log10(Pout_tot_aig/P_avs)",
        # ----- Reference convention: 4-frequency total (INCLUDES pump (0,1) at 2 f_s + (1,-1)) -----
        "Vpump_out=mix(n_output,{0,1})",       # pump (0,1) at 2 f_s -- NOT on f_s (README)
        "Vout_1m1=mix(n_output,{1,-1})",       # (1,-1) at -f_s
        "Vout_tot_4f=Vout_sig+Vpump_out+Vout_idler+Vout_1m1",
        "Pout_tot_4f=0.5*mag(Vout_tot_4f)*mag(Vout_tot_4f)/50",
        "Vin_pump=mix(n_input,{0,1})",
        "Vin_idler=mix(n_input,{-1,1})",
        "Vin_1m1=mix(n_input,{1,-1})",
        "Vin_tot_4f=Vin_sig+Vin_pump+Vin_idler+Vin_1m1",
        "Pin_tot_4f=0.5*mag(Vin_tot_4f)*mag(Vin_tot_4f)/50",
        "Gain_4f_dB=10*log10(Pout_tot_4f/Pin_tot_4f)",
        # ----- ratio / voltage-gain cross-checks (existing) -----
        "IdlerOverSignal=Pout_idler/Pout_sig",
        "Vgain_sig_dB=20*log10(mag(Vout_sig)/mag(Vin_sig))",
    ]


def percell_equations(n_cells: int) -> list[str]:
    """Return the ``MEAS_PERCELL`` per-cell branch-power equation list.

    Two equations per cell -- pump-band (0,1) and signal-band (1,0) real power
    delivered into each shunt branch -- referencing the named branch-top node
    ``n_c<k>_b`` and the per-cell varactor I_Probe current ``IPR_VAR_C<k>.i``.
    """

    out: list[str] = []
    for k in range(1, int(n_cells) + 1):
        out.append(f"Pcell_pump_{k}=0.5*real(mix(n_c{k}_b,{{0,1}})*conj(mix(IPR_VAR_C{k}.i,{{0,1}})))")
        out.append(f"Pcell_sig_{k}=0.5*real(mix(n_c{k}_b,{{1,0}})*conj(mix(IPR_VAR_C{k}.i,{{1,0}})))")
    return out


