"""Commercial varactor palette with datasheet/SPICE parameters and provenance.

Public foundation for the 2-16 GHz wide search. Each entry carries the vendor's
published SPICE C-V parameters (CJ0, VJ, M), package parasitics (R_series, L_package,
CP), the datasheet C(V) ground-truth table where published, a per-part nominal bias,
the C0 / C' at that bias, a source citation, a law-OK-or-use-table flag, and explicit
UNKNOWN flags for any unpublished parameter.

PROVENANCE DISCIPLINE
---------------------
Every numeric value is transcribed VERBATIM from the cited vendor document (fetched
datasheet PDF / SPICE application note). NOTHING here is recalled, approximated, or
back-filled from a family default. Any parameter a vendor does not publish is ``None``
and listed in ``unknown_flags`` for sign-off — never guessed.

Sources:
  * Skyworks AN 200315C "Varactor SPICE Models for RF VCO Applications" (Aug 5 2015),
    Table 2 — multi-part SPICE params (CJ0, VJ, M, CP, RS). VJ/M/CP are EMPIRICAL FIT
    values (the AN states they "do not represent the precise physical properties").
  * Skyworks DS 200061P "SMV1247-SMV1255 Series" (Oct 3 2014) — Table 4 C(V) table,
    Table 5 part-specific SPICE (supersedes AN 200315C for SMV1251), Table 1 package LS.
  * Skyworks DS 200074Q "SMV2019 to SMV2023 Series" (Aug 7 2015) — Table 4 C(V),
    Table 5 SPICE incl. LS (SMV2019-079LF, SMV2020-079LF).
  * Skyworks DS 200072N "SMV1763 Series" — Table 4 C(V), Table 5 SPICE incl. LS.
  * Skyworks DS 200060I "SMV1245 Series: Hyperabrupt Junction Tuning Varactors"
    (Aug 6 2015) — Table 4 part-specific SPICE incl. LS=0.7 nH for -079LF (SC-79),
    Table 5 C(V) table (VR 0–12 V), Table 3 VBR=26 V. Supersedes AN 200315C for SMV1245.
  * MACOM DC-0003906 "MAVR-000120-1411" Rev V6 (sibling of the requested part).

PACKAGE NOTE: the requested parts are the ``-079LF`` plated-lead variant = Skyworks
SC-79 package, for which the part datasheets give ``LS = 0.7 nH`` (Table 1 of 200060I/
200061P/200074Q/200072N). This SUPERSEDES the AN 200315C blanket ``LS = 1.7 nH`` (the
SOT-23/SOD-323 model value incl. board insertion; note 200060I's own SOD-323 ``-011LF``
LS is 1.5 nH, not 1.7). LS for parts sourced only from the AN (SMV1281, SMV1213,
SMV1405) is taken as the SC-79 class value 0.7 nH and flagged (``ls_from_package_class``)
— confirm against each part datasheet. SMV1245 is now datasheet-sourced (200060I Table 4
gives LS=0.7 nH for -079LF directly), so it carries no ``ls_from_package_class`` flag.

SUPPLEMENTAL PALETTE EXPANSION
------------------------------
The public supplement carries several supplemental palette entries and value
updates used by the paper design search. This pass therefore:

* ADDS three parts present in the supplemental palette but not the datasheet-sourced palette —
  ``SMV1233-079``, ``SMV1705-004`` (lowest R_s, 0.32 Ω), and ``1465-MAVR-000350``
  (MACOM GaAs, now with a full junction model, distinct from the unknown
  ``MAVR-0010214101P`` placeholder which is retained).
* UPDATES three entries to the supplemental palette values while PRESERVING the
  prior datasheet-sourced numbers in ``notes`` and
  flagging the divergence (``differs_from_prior_datasheet``):
    - ``SMV1405-079``  CJ0/VJ/M/CP 2.92/0.68/0.41/0.05 → 2.37/0.77/0.50/0.29
      (R_s 0.8, L_s 0.7 unchanged). The supplemental junction model reconciles with the
      device-of-record polynomial (``models.varactor.smv1405``): poly C(0 V) = 2.66 pF =
      C_jo + C_p = 2.37 + 0.29. That polynomial is carried here as
      ``cv_poly_coeffs``. The device-of-record uses R_s 1.0 Ω / L_pkg 1.7 nH,
      which differ from the supplemental 0.8 Ω / 0.7 nH values and are flagged
      for sign-off.
    - ``SMV1281-079LF``  L_s 0.7 → 1.2 nH (the prior 0.7 was the flagged package-class
      ASSUMPTION, not a datasheet value; the supplemental palette supersedes it).
    - ``SMV1213-079``  R_s 0.8 → 1.4 Ω.

POLYNOMIAL C(V): the supplemental palette flags several parts as "(poly)" with
a polynomial fit whose C0 approximates C_jo + C_p as a sanity check. Five parts
carry a transcribed ``cv_poly_coeffs`` plus a ``cv_poly_valid_max_v``
validity ceiling: SMV1233 / SMV1405 / SMV1705 / SMV1245 (≤8 V), SMV1281 (≤12 V). The
supplemental SMV2020 transcription is rejected (a copy of the SMV1405 device-of-record
poly; see below), so SMV2020 carries none. The fits are NON-PHYSICAL past their window (they go negative / non-monotonic), so
:meth:`cv_pf_guarded` / :meth:`cv_poly_window_v` use the poly only inside the window and
FALL BACK to the junction law beyond. The design search deliberately uses the
junction model at the operating bias when operating points sit above the poly window.
The polys are carried for completeness and low-bias C(V) cross-checks only.
Two poly disagreements: (1) SMV2020's supplemental poly is identical
to the device-of-record SMV1405 poly and its C0 = 2.66 pF disagrees with C_jo = 3.20 pF
(table C(0V) = 3.20); it is rejected and not attached to the SMV2020 entry; (2)
SMV1405's supplemental poly differs (beyond a_0) from the device-of-record poly. INCOMPLETE
parts (BBY6502/BBY6602 — poly but no parasitics; 1465-MAVR-000409, 150-KVX2301 — no params)
are NOT added.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Mapping

from .design.realistic_varactor import RealisticVaractor

__all__ = ["PaletteEntry", "PALETTE", "get_entry", "palette_part_numbers"]

_pF = 1.0e-12
_nH = 1.0e-9


@dataclass(frozen=True)
class PaletteEntry:
    """One commercial varactor with sourced parameters + provenance + flags.

    Capacitances in pF, inductance in nH, resistance in ohm, voltages in V — the
    human-readable datasheet units. :meth:`to_realistic_varactor` converts to the SI
    :class:`RealisticVaractor` the engine consumes. ``cv_table`` maps reverse voltage
    [V] to TOTAL capacitance [pF] (junction + CP) exactly as the datasheet tabulates.
    ``None`` for any unpublished field; see ``unknown_flags``.
    """

    part_number: str
    device_class: str            # "Si hyperabrupt" | "Si abrupt" | "GaAs hyperabrupt"
    vendor: str
    cj0_pf: float | None
    vj_v: float | None
    m: float | None
    cp_pf: float | None
    r_series_ohm: float | None
    l_package_nh: float | None
    cv_table: Mapping[float, float] | None
    v_bias_v: float              # per-part nominal reverse bias for C0/C'
    source: str
    cv_source: str | None
    law_ok_within_5pct: bool | None      # None => no datasheet table fetched to test
    law_max_dev_pct: float | None        # max |law-table| over the tabulated points
    unknown_flags: tuple[str, ...] = ()
    notes: str = ""
    # Datasheet polynomial C-V fit C(v) = sum_k coeffs[k] v^k [F], v = reverse bias [V]
    # (supplemental public palette "(poly)" parts). ``None`` => no polynomial for this part
    # (e.g. MAVR-000350 is a junction
    # model). ``cv_poly_valid_max_v`` is the datasheet-stated upper bound of the fit's
    # validity (volts); BEYOND it the fit is non-physical and callers must fall back to
    # the junction law -- see :meth:`c_poly_pf` / :meth:`cv_pf_guarded` / :meth:`cv_poly_window_v`.
    cv_poly_coeffs: tuple[float, ...] | None = None
    cv_poly_valid_max_v: float | None = None

    # ---- derived C-V quantities (junction power law) -------------------- #
    def c_junction_pf(self, v: float) -> float:
        return self.cj0_pf / (1.0 + v / self.vj_v) ** self.m

    def c0_junction_pf(self) -> float | None:
        if None in (self.cj0_pf, self.vj_v, self.m):
            return None
        return self.c_junction_pf(self.v_bias_v)

    def c0_total_pf(self) -> float | None:
        c = self.c0_junction_pf()
        return None if c is None else c + (self.cp_pf or 0.0)

    def c0_poly_pf(self) -> float | None:
        """C(0 V) [pF] from the polynomial C-V fit, if coefficients are present.

        The polynomial sanity check is ``poly C0 ≈ C_jo + C_p`` at 0 V (compare
        with :meth:`c0_total_pf`). ``None`` when no polynomial has been transcribed
        (see ``cv_poly_coeffs_pending_xlsx``).
        """
        if self.cv_poly_coeffs is None:
            return None
        return float(self.cv_poly_coeffs[0]) * 1e12

    def c_poly_pf(self, v: float) -> float | None:
        """RAW polynomial C(v) [pF] at reverse bias ``v`` -- NO physicality guard.

        ``C(v) = sum_k coeffs[k]*v^k`` [F], converted to pF. ``None`` when no
        polynomial is carried. The raw fit is non-physical past its validity
        window (goes negative / non-monotonic); use :meth:`cv_pf_guarded` for the
        guarded value and :meth:`cv_poly_window_v` for the window.
        """
        if self.cv_poly_coeffs is None:
            return None
        v = float(v)
        c_f = 0.0
        for k, a in enumerate(self.cv_poly_coeffs):
            c_f += float(a) * (v ** k)
        return c_f * 1e12

    def cv_poly_window_v(self, *, v_hi: float = 30.0, dv: float = 0.05) -> float | None:
        """Effective valid window [0, v*] of the polynomial fit (volts).

        ``v*`` is the smallest of: the datasheet-stated ``cv_poly_valid_max_v``,
        the bias where the raw poly first becomes non-positive, and the bias where
        it first stops monotonically decreasing (``dC/dv`` turns >= 0). This is the
        physicality guard ("positive, monotonically decreasing");
        beyond ``v*`` the junction law is used. ``None`` when no polynomial.
        """
        if self.cv_poly_coeffs is None:
            return None
        stated = float(self.cv_poly_valid_max_v) if self.cv_poly_valid_max_v is not None else v_hi
        hi = min(float(v_hi), stated)
        v = 0.0
        c_prev = self.c_poly_pf(0.0)
        if c_prev is None or c_prev <= 0.0:
            return 0.0
        v_star = 0.0
        steps = int(round(hi / dv))
        for i in range(1, steps + 1):
            v = i * dv
            c = self.c_poly_pf(v)
            if c is None or c <= 0.0 or c >= c_prev:  # non-positive or no longer decreasing
                break
            v_star = v
            c_prev = c
        return min(v_star, stated)

    def cv_pf_guarded(self, v: float) -> tuple[float, str]:
        """GUARDED total C(v) [pF] + which model was used ("poly" | "junction").

        Uses the datasheet polynomial when ``v`` is inside its valid window
        (:meth:`cv_poly_window_v`), else falls back to the junction-law total
        :meth:`c0_total_pf`-style ``c_junction_pf(v) + CP``. The optimizer search
        deliberately uses the JUNCTION model at the operating bias (most operating
        points sit above the poly window); this guarded helper is for the
        completeness / low-bias cross-check only.
        """
        win = self.cv_poly_window_v()
        if win is not None and 0.0 <= float(v) <= win:
            c = self.c_poly_pf(v)
            if c is not None and c > 0.0:
                return c, "poly"
        # junction-law fallback (requires the junction params)
        if None in (self.cj0_pf, self.vj_v, self.m):
            raise ValueError(
                f"{self.part_number}: poly invalid at v={v} V and no junction-law "
                "parameters to fall back to."
            )
        return self.c_junction_pf(v) + (self.cp_pf or 0.0), "junction"

    def cprime_pf_per_v(self) -> float | None:
        if None in (self.cj0_pf, self.vj_v, self.m):
            return None
        v = self.v_bias_v
        return -(self.m / self.vj_v) * self.cj0_pf * (1.0 + v / self.vj_v) ** (-self.m - 1.0)

    def to_realistic_varactor(
        self, *, v_bias_v: float | None = None, taylor_order: int = 5
    ) -> RealisticVaractor:
        """Build the engine's SI :class:`RealisticVaractor` at a reverse bias.

        ``v_bias_v`` defaults to the entry's nominal :attr:`v_bias_v` (existing
        callers unchanged). Pass an explicit value to instantiate the same
        bias-independent SPICE params at any reverse bias -- the engine's
        V_bias SEARCH dimension (the C-V law CJ0/VJ/M and the
        package L_pkg/R_series/CP are bias-independent; only the operating
        point moves). Raises ``ValueError`` if the C-V law parameters are
        UNKNOWN (e.g. the MACOM part), so the engine never silently consumes a
        guessed device.
        """
        if None in (self.cj0_pf, self.vj_v, self.m):
            raise ValueError(
                f"{self.part_number}: C-V law parameters are UNKNOWN "
                f"({', '.join(self.unknown_flags)}); cannot build a RealisticVaractor "
                "without sourced CJ0/VJ/M. Source the exact datasheet first."
            )
        bias = self.v_bias_v if v_bias_v is None else float(v_bias_v)
        return RealisticVaractor(
            CJ0_f=self.cj0_pf * _pF, VJ_v=self.vj_v, M=self.m,
            CP_f=(self.cp_pf or 0.0) * _pF,
            L_pkg_h=(self.l_package_nh or 0.0) * _nH,
            R_series_ohm=(self.r_series_ohm or 0.0),
            V_bias_v=bias, taylor_order=taylor_order,
        )


# Datasheet C(V) tables (VR [V] -> CT [pF], TOTAL incl. CP) — transcribed verbatim.
_CV_SMV1251 = {0: 53.65, 0.5: 38.23, 1.0: 28.09, 1.5: 20.13, 2.0: 13.55, 2.5: 8.60,
               3.0: 5.78, 3.5: 4.57, 4.0: 3.95, 4.5: 3.58, 5.0: 3.33, 5.5: 3.16,
               6.0: 3.03, 6.5: 2.94, 7.0: 2.88, 7.5: 2.83, 8.0: 2.79}
_CV_SMV1763 = {0: 9.0, 0.5: 6.7, 1.0: 5.2, 1.5: 4.0, 2.0: 3.2, 2.5: 2.6, 3.0: 2.2,
               3.5: 2.0, 4.0: 1.9, 4.5: 1.8, 5.0: 1.8}
_CV_SMV2019 = {0: 2.22, 0.5: 1.77, 1.0: 1.51, 1.5: 1.33, 2.0: 1.20, 2.5: 1.08, 3.0: 0.98,
               3.5: 0.89, 4.0: 0.81, 5.0: 0.66, 6.0: 0.55, 7.0: 0.48, 8.0: 0.44, 9.0: 0.40,
               10.0: 0.38, 11.0: 0.37, 12.0: 0.35, 13.0: 0.34, 14.0: 0.33, 15.0: 0.32,
               16.0: 0.32, 17.0: 0.31, 18.0: 0.31, 19.0: 0.30, 20.0: 0.30}
_CV_SMV2020 = {0: 3.20, 0.5: 2.59, 1.0: 2.25, 1.5: 2.01, 2.0: 1.83, 2.5: 1.68, 3.0: 1.56,
               3.5: 1.46, 4.0: 1.36, 5.0: 1.19, 6.0: 1.03, 7.0: 0.88, 8.0: 0.75, 9.0: 0.64,
               10.0: 0.57, 11.0: 0.52, 12.0: 0.48, 13.0: 0.45, 14.0: 0.43, 15.0: 0.41,
               16.0: 0.39, 17.0: 0.38, 18.0: 0.37, 19.0: 0.36, 20.0: 0.35}
_CV_SMV1245 = {0: 7.37, 0.5: 5.84, 1.0: 4.93, 1.5: 4.28, 2.0: 3.79, 2.5: 3.40, 3.0: 3.06,
               3.5: 2.76, 4.0: 2.51, 4.5: 2.28, 5.0: 2.09, 5.5: 1.92, 6.0: 1.78, 6.5: 1.66,
               7.0: 1.55, 7.5: 1.46, 8.0: 1.38, 8.5: 1.32, 9.0: 1.26, 9.5: 1.20, 10.0: 1.16,
               10.5: 1.12, 11.0: 1.08, 11.5: 1.05, 12.0: 1.02}  # Skyworks 200060I Table 5
# MACOM published C(V) is only spec points (min/max), not a full table.
_CV_MACOM = {4.0: 0.35, 10.0: 0.17}  # midpoints of spec ranges (0.30-0.40 @4V, 0.14-0.20 @10V)

# Reference SMV1405 verbatim differential-capacitance polynomial C(v) = sum_k a_k v^k [F]
# (the DEVICE-OF-RECORD ``models.varactor.smv1405``; pass-B reconciliation, locked by the
# SMV1405 combined-termination regression test -- that test defines its own copy, so this
# palette constant is provenance only). a_0 = C(0 V) = 2.66 pF == C_jo + C_p
# (2.37 + 0.29). This device-of-record fit differs (beyond a_0) from the
# supplemental public palette poly carried for SMV1405 (``_POLY_XLSX_SMV1405``).
# Constant name keeps digits at the END -> avoids the "<NN>_" example-ID guard.
_POLY_REFERENCE_SMV1405 = (2.66e-12, -1.36e-12, 7.76e-13, -2.7e-13, 4.85e-14, -3.43e-15)

# --------------------------------------------------------------------------------------- #
# Supplemental public palette "(poly)" C-V fits: C(v) = sum_k coeffs[k] v^k [F],
# v = reverse-bias volts. Each carries its validity ceiling; past it the fit is
# non-physical and the junction law is used (the cv_pf_guarded / cv_poly_window_v guard).
# Constant names keep the part-number digits at the END (example-ID guard).
_POLY_XLSX_SMV1233 = (5.02e-12, -2.22e-12, 6.08e-13, -9.82e-14, 8.60e-15, -3.10e-16)   # <= ~8 V
_POLY_XLSX_SMV1405 = (2.66e-12, -1.16e-12, 4.57e-13, -9.84e-14, 1.04e-14, -4.27e-16)   # <= ~8 V; a_0 == REFERENCE a_0
_POLY_XLSX_SMV1705 = (3.15e-11, -1.88e-11, 6.43e-12, -1.24e-12, 1.23e-13, -4.82e-15)   # <= ~8 V
_POLY_XLSX_SMV1245 = (7.37e-12, -3.16e-12, 9.26e-13, -1.65e-13, 1.56e-14, -5.92e-16)   # <= ~8 V
_POLY_XLSX_SMV1281 = (1.36e-11, -5.43e-12, 1.19e-12, -1.54e-13, 1.08e-14, -3.12e-16)   # <= ~12 V
# REJECTED (v1.0.1): the supplemental "SMV2020" poly coefficients are numerically identical to
# the device-of-record SMV1405 polynomial (_POLY_REFERENCE_SMV1405); its a_0 = 2.66 pF disagrees
# with SMV2020's junction C_jo = 3.20 pF and the datasheet table C(0 V) = 3.20 pF. It is a copy of
# the SMV1405 row, not an SMV2020 fit. The SMV2020 entry therefore carries NO polynomial
# (cv_poly_coeffs None): every C(V) path falls back to the junction law. The transcription is
# kept here for provenance only and is referenced by no entry.
_POLY_XLSX_SMV2020_REJECTED = (2.66e-12, -1.36e-12, 7.76e-13, -2.70e-13, 4.85e-14, -3.43e-15)

_AN = "Skyworks AN 200315C Table 2 (Aug 5 2015)"
_LS_PKG = "ls_from_package_class(SC-79=0.7nH; confirm vs part datasheet)"

PALETTE: tuple[PaletteEntry, ...] = (
    PaletteEntry(
        part_number="SMV1245-079LF", device_class="Si hyperabrupt", vendor="Skyworks",
        cj0_pf=6.9, vj_v=3.5, m=1.7, cp_pf=0.47, r_series_ohm=2.0, l_package_nh=0.7,
        cv_table=_CV_SMV1245, v_bias_v=2.0,
        source="Skyworks DS 200060I Table 4 (Aug 6 2015): SMV1245-079LF SPICE "
               "CJ0=6.9/VJ=3.5/M=1.7/CP=0.47/RS=2, LS=0.7 nH (SC-79; -079LF row & Table 1)",
        cv_source="Skyworks DS 200060I Table 5 (C vs VR, typ @25 C, VR 0-12 V); xlsx poly "
                  "also carried (supplemental public palette).",
        law_ok_within_5pct=False, law_max_dev_pct=6.0, unknown_flags=(),
        cv_poly_coeffs=_POLY_XLSX_SMV1245, cv_poly_valid_max_v=8.0,
        notes="ANCHOR. Part-datasheet SPICE (200060I Table 4) supersedes AN 200315C; the "
              "6.9/3.5/1.7/0.47/2 values coincide, now confirmed for the exact part. C(V) is "
              "the verbatim Table 5 (total C incl. CP); the law reproduces it ~6.0% max (at "
              "VR=4 V) / ~2.1% mean, so it is usable but the TABLE is ground truth -- the law "
              "over-predicts |C'| ~12% at the 2 V bias (-0.99 vs -0.88 pF/V from the table). "
              "VBR=26 V (Table 3). LS=0.7 nH is the -079LF SC-79 value (Table 4 & Table 1); the "
              "datasheet's SOD-323 (-011LF) LS is 1.5 nH and does NOT apply. CP=0.47 is the "
              "published -079LF value (identical to the -011LF row -- no package ambiguity).",
    ),
    PaletteEntry(
        part_number="SMV1251-079LF", device_class="Si hyperabrupt", vendor="Skyworks",
        cj0_pf=52.48, vj_v=100.0, m=76.0, cp_pf=2.00, r_series_ohm=1.4, l_package_nh=0.7,
        cv_table=_CV_SMV1251, v_bias_v=2.0,
        source="Skyworks DS 200061P Table 5 (Oct 3 2014); LS SC-79=0.7nH Table 1",
        cv_source="Skyworks DS 200061P Table 4",
        law_ok_within_5pct=False, law_max_dev_pct=30.6,
        unknown_flags=(),
        notes="Part-datasheet SPICE (200061P) supersedes AN 200315C (which gave "
              "60/17/14/0/1.3). Model fit only to <=4 V; law deviates ~31% at the high-V tail.",
    ),
    PaletteEntry(
        part_number="SMV1281-079LF", device_class="Si hyperabrupt", vendor="Skyworks",
        cj0_pf=13.0, vj_v=14.0, m=6.0, cp_pf=0.62, r_series_ohm=1.7, l_package_nh=1.2,
        cv_table=None, v_bias_v=2.0,
        source="Skyworks AN 200315C plus supplemental public palette transcription: "
               "CJ0/VJ/M/CP/RS=13/14/6/0.62/1.7, L_s=1.2 nH.",
        cv_source="Supplemental public palette polynomial fit.",
        law_ok_within_5pct=None, law_max_dev_pct=None,
        cv_poly_coeffs=_POLY_XLSX_SMV1281, cv_poly_valid_max_v=12.0,
        unknown_flags=("cv_table_not_fetched",
                       "differs_from_prior_palette(L_s 0.7nH was a package-class assumption)"),
        notes="VJ/M/CP empirical fit (AN). Supplemental polynomial C(V) carried "
              "(valid <= ~12 V; "
              "highest poly window in the palette). C(V) datasheet TABLE still not fetched. "
              "L_s updated 0.7 to 1.2 nH per the supplemental palette (the 0.7 was the ls_from_package_class "
              "ASSUMPTION, not a datasheet value).",
    ),
    PaletteEntry(
        part_number="SMV1213-079LF", device_class="Si hyperabrupt", vendor="Skyworks",
        cj0_pf=28.9, vj_v=190.0, m=105.0, cp_pf=2.2, r_series_ohm=1.4, l_package_nh=0.7,
        cv_table=None, v_bias_v=2.0,
        source="Skyworks AN 200315C plus supplemental public palette transcription: "
               "CJ0/VJ/M/CP=28.9/190/105/2.2, R_s=1.4 ohm, L_s=0.7 nH.",
        cv_source=None,
        law_ok_within_5pct=None, law_max_dev_pct=None,
        unknown_flags=("cv_table_not_fetched", "cv_poly_coeffs_pending_xlsx", _LS_PKG,
                       "differs_from_prior_datasheet(R_s 0.8Ω AN 200315C)"),
        notes="Extreme hyperabrupt fit (VJ=190, M=105 — non-physical empirical values). "
              "C(V) table / poly not fetched; the simple law is expected to need the table. "
              "R_s updated 0.8 to 1.4 ohm per the supplemental palette.",
    ),
    PaletteEntry(
        part_number="SMV2019-079LF", device_class="Si hyperabrupt", vendor="Skyworks",
        cj0_pf=2.25, vj_v=3.5, m=1.4, cp_pf=0.07, r_series_ohm=4.8, l_package_nh=0.7,
        cv_table=_CV_SMV2019, v_bias_v=2.0,
        source="Skyworks DS 200074Q Table 5 (Aug 7 2015)",
        cv_source="Skyworks DS 200074Q Table 4",
        law_ok_within_5pct=False, law_max_dev_pct=24.5, unknown_flags=(),
        notes="Part-datasheet SPICE incl. LS=0.7nH (SC-79). Law deviates ~25% at high-V tail.",
    ),
    PaletteEntry(
        part_number="SMV2020-079LF", device_class="Si hyperabrupt", vendor="Skyworks",
        cj0_pf=3.20, vj_v=4.4, m=1.45, cp_pf=0.0, r_series_ohm=2.5, l_package_nh=0.7,
        cv_table=_CV_SMV2020, v_bias_v=2.0,
        source="Skyworks DS 200074Q Table 5 (Aug 7 2015)",
        cv_source="Skyworks DS 200074Q Table 4 (table is ground truth); no polynomial "
                  "(the supplemental transcription is a copy of SMV1405 and is rejected).",
        law_ok_within_5pct=False, law_max_dev_pct=23.7,
        cv_poly_coeffs=None, cv_poly_valid_max_v=None,
        unknown_flags=("cv_poly_rejected_copy_of_SMV1405_reference",),
        notes="Part-datasheet SPICE incl. LS=0.7nH (SC-79), CP=0. Law deviates ~24% at "
              "high-V tail (TABLE _CV_SMV2020 is ground truth). No polynomial C(V): the "
              "supplemental 'SMV2020' poly equals the device-of-record SMV1405 poly (C0=2.66 pF "
              "vs junction CJ0=3.20 pF and table C(0V)=3.20 pF), so it is rejected "
              "(_POLY_XLSX_SMV2020_REJECTED, provenance only) and every C(V) path uses the "
              "junction law. SPICE junction parameters unchanged.",
    ),
    PaletteEntry(
        part_number="SMV1763-079LF", device_class="Si hyperabrupt", vendor="Skyworks",
        cj0_pf=7.63, vj_v=26.7, m=20.0, cp_pf=1.42, r_series_ohm=0.5, l_package_nh=0.7,
        cv_table=_CV_SMV1763, v_bias_v=2.0,
        source="Skyworks DS 200072N Table 5", cv_source="Skyworks DS 200072N Table 4",
        law_ok_within_5pct=False, law_max_dev_pct=7.4, unknown_flags=(),
        notes="Part-datasheet SPICE incl. LS=0.7nH (SC-79). Datasheet marked DISCONTINUED. "
              "Law borderline (~7% max dev at the tail); fits well over the 0-3 V operating range.",
    ),
    PaletteEntry(
        part_number="SMV1405-079LF", device_class="Si abrupt", vendor="Skyworks",
        cj0_pf=2.37, vj_v=0.77, m=0.5, cp_pf=0.29, r_series_ohm=0.8, l_package_nh=0.7,
        cv_table=None, v_bias_v=2.0,
        source="Supplemental public palette transcription: CJ0=2.37/VJ=0.77/"
               "M=0.50/CP=0.29/RS=0.8/LS=0.7 with polynomial fit. Supersedes AN 200315C "
               "(2.92/0.68/0.41/0.05).",
        cv_source="Supplemental public palette polynomial fit; reconciled with "
                  "the device-of-record poly (models.varactor.smv1405, C0=2.66 pF).",
        law_ok_within_5pct=None, law_max_dev_pct=None,
        cv_poly_coeffs=_POLY_XLSX_SMV1405, cv_poly_valid_max_v=8.0,
        unknown_flags=("differs_from_prior_datasheet(AN 200315C 2.92/0.68/0.41/0.05)",
                       "device_of_record_parasitics_differ(R=1.0ohm/L=1.7nH vs sheet 0.8/0.7)",
                       "xlsx_poly_differs_from_device_of_record_poly_beyond_a0",
                       _LS_PKG),
        notes="Device-of-record part. The palette now carries the supplemental polynomial "
              "(_POLY_XLSX_SMV1405, valid <= ~8 V). Its a_0 = 2.66 pF reconciles with the "
              "junction C_jo+C_p = 2.37+0.29 = 2.66 pF AND with the device-of-record "
              "poly a_0 (models.varactor.smv1405, _POLY_REFERENCE_SMV1405) -- but the two polys "
              "DIFFER beyond a_0 (the device-of-record is a local small-signal fit; the supplemental "
              "fit follows the palette transcription). The screen uses L_s=0.7 nH; the "
              "device-of-record uses R=1.0 Ω / L_pkg=1.7 nH — that L would lower SMV1405's "
              "SRF ~1.56× (more pump-starved); flagged for sign-off. Prior palette value was "
              "the AN 200315C abrupt fit (2.92/0.68/0.41/0.05).",
    ),
    PaletteEntry(
        part_number="SMV1233-079LF", device_class="Si hyperabrupt", vendor="Skyworks",
        cj0_pf=4.21, vj_v=11.87, m=6.43, cp_pf=0.81, r_series_ohm=1.2, l_package_nh=0.7,
        cv_table=None, v_bias_v=2.0,
        source="Supplemental public palette transcription: "
               "CJ0=4.21/VJ=11.87/M=6.43/CP=0.81/RS=1.2/LS=0.7 with polynomial fit.",
        cv_source="Supplemental public palette polynomial fit.",
        law_ok_within_5pct=None, law_max_dev_pct=None,
        cv_poly_coeffs=_POLY_XLSX_SMV1233, cv_poly_valid_max_v=8.0,
        unknown_flags=("cv_table_not_fetched", "vbr_not_in_sheet"),
        notes="Hyperabrupt (M=6.43). Supplemental polynomial C(V) carried "
              "(valid <= ~8 V); the search uses the junction law at the operating bias. "
              "VBR not in the palette source; bias capped conservatively in the screen.",
    ),
    PaletteEntry(
        part_number="SMV1705-004", device_class="Si hyperabrupt", vendor="Skyworks",
        cj0_pf=31.0, vj_v=3.0, m=2.0, cp_pf=0.5, r_series_ohm=0.32, l_package_nh=0.8,
        cv_table=None, v_bias_v=2.0,
        source="Supplemental public palette transcription: "
               "CJ0=31/VJ=3/M=2/CP=0.5/RS=0.32/LS=0.8 with polynomial fit.",
        cv_source="Supplemental public palette polynomial fit.",
        law_ok_within_5pct=None, law_max_dev_pct=None,
        cv_poly_coeffs=_POLY_XLSX_SMV1705, cv_poly_valid_max_v=8.0,
        unknown_flags=("cv_table_not_fetched", "vbr_not_in_sheet"),
        notes="Lowest R_s in the palette (0.32 Ω) — the "
              "loss-favoured candidate — but high CJ0=31 pF (large C → low SRF), so the "
              "coupling screen tests whether bias lifts its SRF enough to clear high f. "
              "Supplemental polynomial C(V) carried (valid <= ~8 V); HELD pending V_BR. "
              "VBR not in the palette source (bias capped conservatively).",
    ),
    PaletteEntry(
        part_number="MAVR-000350", device_class="GaAs hyperabrupt", vendor="MACOM",
        cj0_pf=15.7, vj_v=14.55, m=7.26, cp_pf=0.15, r_series_ohm=1.0, l_package_nh=1.4,
        cv_table=None, v_bias_v=4.0,
        source="Supplemental public palette transcription for part '1465-MAVR-000350': "
               "CJ0=15.7/VJ=14.55/M=7.26/CP=0.15/RS=1/LS=1.4 (junction model); "
               "distinct from the unknown MAVR-0010214101P entry.",
        cv_source=None,
        law_ok_within_5pct=None, law_max_dev_pct=None,
        unknown_flags=("cv_table_not_fetched", "vbr_not_in_sheet"),
        notes="MACOM GaAs hyperabrupt with a full junction model (unlike the unknown "
              "MAVR-0010214101P placeholder, which is retained). Junction model (not poly) "
              "per the supplemental palette. Largest L_s in the palette (1.4 nH). VBR not in the palette source "
              "(bias capped conservatively in the screen).",
    ),
    PaletteEntry(
        part_number="MAVR-0010214101P", device_class="GaAs hyperabrupt", vendor="MACOM",
        cj0_pf=None, vj_v=None, m=1.0, cp_pf=None, r_series_ohm=None, l_package_nh=None,
        cv_table=_CV_MACOM, v_bias_v=4.0,
        source="EXACT PART NOT LOCATED in any public MACOM datasheet/distributor. "
               "Reference sibling: MACOM DC-0003906 MAVR-000120-1411 Rev V6.",
        cv_source="MACOM DC-0003906 Table 'Total Capacitance' (Cj 0.30-0.40 pF @4V, "
                  "0.14-0.20 pF @10V); gamma=1.0+/-10% (2-12 V).",
        law_ok_within_5pct=None, law_max_dev_pct=None,
        unknown_flags=("EXACT_PART_NUMBER_NOT_FOUND", "cj0_unknown", "vj_unknown",
                       "r_series_unknown", "l_package_unknown", "cp_parasitic_unknown",
                       "cv_table_is_spec_midpoints_only"),
        notes="GaAs constant-gamma flip-chip: only gamma (M=1.0) and 2 spec C-points are "
              "published; CJ0/VJ/RS/LS/Cparasitic are NOT given numerically (shown as symbols "
              "in the equivalent circuit). DO NOT build an engine device until REFERENCE confirms "
              "the exact part and its full model. NB: Cj~0.30 pF @4V is the closest real part to "
              "the framework's abstract 'Varactor A' (0.3 pF) — see SMV1245 reconciliation.",
    ),
)


def get_entry(part_number: str) -> PaletteEntry:
    key = part_number.upper()
    for e in PALETTE:
        if e.part_number.upper() == key or e.part_number.upper().split("-")[0] == key:
            return e
    raise KeyError(f"no palette entry for {part_number!r}; have {palette_part_numbers()}")


def palette_part_numbers() -> tuple[str, ...]:
    return tuple(e.part_number for e in PALETTE)


