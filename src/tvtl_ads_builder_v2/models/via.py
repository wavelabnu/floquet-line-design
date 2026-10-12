"""Via (stub-to-ground termination) models for the chain model registry.

A :class:`ViaModel` is the **single source** for a
SHORT-terminated cell's stub-to-ground via, consumed identically by the two render
sites that matter:

* the **emitter** -- :meth:`ViaModel.emit` renders the via into the device spec
  (the cell's stub end -> ground termination);
* the **analytical leg** -- :meth:`ViaModel.analytical_termination` returns the
  ``Z_load(omega)`` the harmonic-ABCD stub recursion anchors at, computed from the
  **same** parameters the emitter rendered.

Three registered variants:

* :class:`LumpedLVia` -- the current reduced model: a single lumped ``L`` inductor
  to ground carrying the geometry-derived Bahl inductance (``via_inductance_bahl_H``).
  Its :meth:`emit` reproduces the historical ``LVIA_C{k}`` element byte-for-byte and
  its :meth:`analytical_termination` returns ``None`` so the analytical leg keeps its
  unchanged spec-extraction path (``j*omega*L`` from the emitted ``L``) -- the
  recovery-gate behaviour is preserved exactly.
* :class:`RlcVia` -- the *Mathematica* convention: an inductive via rendered with
  explicit lumped ``R`` / ``L`` (series to ground) and an optional shunt pad ``C`` to
  ground, from ``{Z0via, Lvia, Rvia, Cvia}``. The analytical termination is the matching
  ``Z_via(omega) = (Rvia + j*omega*Lvia) || (1/(j*omega*Cvia))`` -- emitter and
  analytical computed from one parameter set, never two.
* :class:`ViagndVia` -- the *ADS* convention: the ADS ``VIAGND`` physical
  via-to-ground (a 1-port that grounds internally). :meth:`emit` renders
  ``VIAGND:<id> <end> Subst="…" D=… W=… T=… Rho=…`` from the via barrel radius +
  metal/resistivity. :meth:`analytical_termination` has two modes:

  - ``"effective_L"`` (the **default**) -- a constant effective inductance matching the
    ADS ``VIAGND`` element itself. Characterizing the ADS element (1-port reflection
    probe, ``internal diagnostics/model_convention/viagnd_characterize.py``) shows it is a
    *frequency-flat* inductor: on Rogers 4003C (h=1.524 mm) with the default D=1.6 mm
    barrel and the nominal connecting width W=0.945 mm it presents L_eff=0.1949 nH with
    0.00 % spread across 2-16 GHz (+ a negligible R~0.006 Ω). The analytical termination
    returns ``R_eff + jωL_eff`` so it tracks the ADS element *exactly across the band*.
  - ``"goldfarb_pucel"`` -- the prior physical-via input impedance
    (:func:`tvtl.microstrip.discontinuities.via_impedance_ohm`); it presents ~0.43 nH
    in-band, i.e. ~2x the ADS ``VIAGND`` element -> the ~1.5 dB ADS↔analytical gap the
    model-convention pass measured. Retained for provenance / comparison.

  The effective-L default **closes** the VIAGND ADS↔analytical consistency gap (from
  ~1.5 dB to ~0.3 dB single-cell |S21| @3.1 GHz, matching the RLC-via-at-0.19 nH leg).
  The W-sensitivity of the ADS element (L_eff 0.236 nH @W=0.6 mm .. 0.154 nH @W=1.6 mm)
  is the residual a single ``effective_L_h`` carries; the exact via geometry (or a
  re-characterization) sets it for a different D/W.

Adding a new via variant = one ``@register_via("kind")`` registration (see
:mod:`.registry`); no call site changes.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Callable

from tvtl.microstrip.discontinuities import via_impedance_ohm

from ..device_spec import Element, ElementPin, Node
from .deprecation import warn_deprecated_termination

__all__ = ["ViaModel", "LumpedLVia", "RlcVia", "ViagndVia"]


class ViaModel(ABC):
    """Interface for a stub-to-ground via termination (SHORT cells)."""

    #: registry key (set by each concrete variant)
    kind: str = ""

    def extra_nodes(self, k: int, end_node: str) -> list[Node]:
        """Internal node declarations this via introduces for cell ``k``.

        Default: none (a single two-terminal element between ``end_node`` and the
        cell ground needs no extra net). :class:`RlcVia` overrides this for its
        series R-L internal node.
        """

        return []

    @abstractmethod
    def emit(
        self, k: int, *, end_node: str, ground_node: str, l_via_H: float,
        width_m: float | None = None, substrate_name: str | None = None,
        n_parallel: int = 1,
    ) -> list[Element]:
        """Render the via termination elements for cell ``k``.

        ``end_node`` is the cell's stub far end; ``ground_node`` is the cell's
        per-cell ground net. ``l_via_H`` is the geometry-derived Bahl inductance the
        emitter pre-computed (used by :class:`LumpedLVia`; ignored by variants that
        carry their own explicit inductance). ``width_m`` (the stub conductor width at
        the via) and ``substrate_name`` are supplied by the cell for physical-via
        variants (:class:`ViagndVia`); the lumped variants ignore them.
        ``n_parallel`` is the topology's via count: :class:`ViagndVia` emits that many
        physical vias; the lumped variants carry the parallel combination in their
        inductance (``l_via_H`` is already the Bahl value for ``n_parallel`` vias).
        """

    def analytical_termination(
        self, *, substrate: object | None = None
    ) -> Callable[[float], complex] | None:
        """``omega -> Z_load(omega)`` for the analytical stub recursion.

        ``None`` (the default, used by :class:`LumpedLVia`) means *use the analytical
        leg's unchanged spec-extraction path* -- i.e. the inductive ``jωL`` termination
        read back from the emitted ``L`` element. A non-``None`` callable overrides it
        (e.g. :class:`RlcVia`'s full ``Z_via`` or :class:`ViagndVia`'s physical-via
        impedance). ``substrate`` (a :class:`tvtl.materials.Substrate`) is supplied by
        the analytical leg for variants that need it (:class:`ViagndVia`); the lumped
        variants ignore it.
        """

        return None


@dataclass(frozen=True)
class LumpedLVia(ViaModel):
    """Reduced model: a single lumped inductor to ground (Bahl ``L_via``).

    ``emit`` reproduces the historical ``LVIA_C{k}`` element exactly (same id, type,
    pins, ``l_h`` parameter) so the reduced config is byte-identical. The analytical
    termination is left to the spec-extraction path (returns ``None``).

    .. deprecated::
        A4 standardization. This lumped, non-distributed via is DEPRECATED in favour
        of the standardized :class:`ViagndVia` (kind ``viagnd``). It stays functional
        for the byte-identical anchor and legacy decks (and the cell's inline lumped-L
        *default* path, which does not instantiate this class, is unchanged).
    """

    kind: str = "lumped_L"

    def __post_init__(self) -> None:
        warn_deprecated_termination(
            "LumpedLVia", self.kind,
            standard_class="ViagndVia", standard_kind="viagnd",
        )

    def emit(
        self, k: int, *, end_node: str, ground_node: str, l_via_H: float,
        width_m: float | None = None, substrate_name: str | None = None,
        n_parallel: int = 1,
    ) -> list[Element]:
        return [
            Element(
                id=f"LVIA_C{k}", type="L",
                pins=(ElementPin(pin=1, node=end_node),
                      ElementPin(pin=2, node=ground_node)),
                parameters={"l_h": float(l_via_H)},
            )
        ]


@dataclass(frozen=True)
class RlcVia(ViaModel):
    """RLC via: series ``R`` + ``L`` to ground, optional shunt pad ``C``.

    Parameters (SI): ``Lvia_h`` series inductance, ``Rvia_ohm`` series resistance,
    ``Cvia_f`` shunt pad capacitance (to ground at the stub end; ``0`` -> no pad),
    ``Z0via_ohm`` the via line's characteristic impedance (carried for provenance /
    future distributed rendering; the lumped rendering uses R/L/C).

    Emitter rendering for cell ``k`` (``end_node`` = stub end, ``g`` = ground):

        R(end_node -> n_c{k}_viaL = Rvia) ; L(n_c{k}_viaL -> g = Lvia)
        [C(end_node -> g = Cvia) if Cvia > 0]

    Analytical termination:

        Z_via(omega) = Zser || Zc , Zser = Rvia + j*omega*Lvia ,
                       Zc = 1/(j*omega*Cvia)   (= open when Cvia == 0)

    .. deprecated::
        A4 standardization. This lumped, non-distributed via is DEPRECATED in favour
        of the standardized :class:`ViagndVia` (kind ``viagnd``). It stays functional
        (the Mathematica-convention configs still build it).
    """

    Lvia_h: float
    Rvia_ohm: float = 0.0
    Cvia_f: float = 0.0
    Z0via_ohm: float = 0.0
    kind: str = "rlc"

    def __post_init__(self) -> None:
        warn_deprecated_termination(
            "RlcVia", self.kind,
            standard_class="ViagndVia", standard_kind="viagnd",
        )

    def extra_nodes(self, k: int, end_node: str) -> list[Node]:
        return [Node(id=f"n_c{k}_viaL")]

    def emit(
        self, k: int, *, end_node: str, ground_node: str, l_via_H: float,
        width_m: float | None = None, substrate_name: str | None = None,
        n_parallel: int = 1,
    ) -> list[Element]:
        mid = f"n_c{k}_viaL"
        els: list[Element] = [
            Element(
                id=f"RVIA_C{k}", type="R",
                pins=(ElementPin(pin=1, node=end_node),
                      ElementPin(pin=2, node=mid)),
                parameters={"r_ohm": float(self.Rvia_ohm)},
            ),
            Element(
                id=f"LVIA_C{k}", type="L",
                pins=(ElementPin(pin=1, node=mid),
                      ElementPin(pin=2, node=ground_node)),
                parameters={"l_h": float(self.Lvia_h)},
            ),
        ]
        if float(self.Cvia_f) > 0.0:
            els.append(Element(
                id=f"CVIA_C{k}", type="C",
                pins=(ElementPin(pin=1, node=end_node),
                      ElementPin(pin=2, node=ground_node)),
                parameters={"c_f": float(self.Cvia_f)},
            ))
        return els

    def analytical_termination(
        self, *, substrate: object | None = None
    ) -> Callable[[float], complex] | None:
        r = float(self.Rvia_ohm)
        lv = float(self.Lvia_h)
        cv = float(self.Cvia_f)

        def _z(omega: float) -> complex:
            w = float(omega)
            z_ser = complex(r, w * lv)
            if cv <= 0.0 or w == 0.0:
                return z_ser
            y = (1.0 / z_ser) + 1j * w * cv
            if abs(y) <= 1.0e-300:
                return complex("inf")
            return 1.0 / y

        return _z


# Characterized effective inductance of the ADS ``VIAGND`` element (1-port
# reflection-probe characterization). On Rogers 4003C (h=1.524 mm)
# with the default D=1.6 mm barrel and the nominal connecting width W=0.945 mm the
# element is a FREQUENCY-FLAT inductor: L_eff=0.1949 nH (0.00 % spread 2-16 GHz) with a
# negligible series R_eff~0.006 Ω. (Goldfarb-Pucel presents ~0.43 nH -> the ~1.5 dB
# ADS↔analytical gap the effective-L mode closes.) W-sensitivity: L_eff ranges 0.236 nH
# @W=0.6 mm to 0.154 nH @W=1.6 mm -- the residual a single effective-L carries.
_VIAGND_NOMINAL_EFFECTIVE_L_H = 0.1949e-9
_VIAGND_NOMINAL_EFFECTIVE_R_OHM = 0.006


@dataclass(frozen=True)
class ViagndVia(ViaModel):
    """ADS ``VIAGND`` physical via-to-ground; analytical = ADS-characterized effective L.

    **Standard SHORT-end termination (A4).** This distributed/physical via is the
    standardized via model; the lumped :class:`LumpedLVia` / :class:`RlcVia` are
    deprecated in its favour. It does not warn.

    Geometry is carried on the model (a physical via is a fixed structure, like
    :class:`RlcVia`'s lumped values): ``via_radius_m`` (the barrel radius -> emitted
    ``D = 2*via_radius``), ``metal_thickness_m`` (emitted ``T``), and ``rho_relative``
    (emitted ``Rho``, the resistivity relative to gold; ``None`` -> omit). The connecting
    conductor width ``W`` is supplied by the cell at :meth:`emit` time (the stub width at
    the via), falling back to ``2*via_radius_m`` when the cell does not pass one.

    Emitter (1-port, grounds internally; ``g`` = the cell ground net is unused):

        VIAGND:VIAGND_C{k} <end_node> Subst="<name>" D=2r W=<w> T=<t> [Rho=<rho>]

    A topology with ``n_parallel`` vias (``DOUBLE`` families: 2) gets that many identical
    elements on the same node, ``VIAGND_C{k}_2`` ... (1.0.0 emitted one regardless).

    Analytical termination -- two modes (``analytical_mode``):

    * ``"effective_L"`` (**default**): a constant inductive load matching the ADS
      ``VIAGND`` element, which is frequency-flat (characterized above):

          Z_load(omega) = effective_R_ohm + j*omega*effective_L_h

      ``effective_L_h`` defaults to the nominal-geometry characterization (0.1949 nH);
      set it for a different via D / connecting W (or to a measured exact value). This
      makes the analytical leg track the ADS element *exactly across band* -> closes the
      VIAGND ADS↔analytical gap.
    * ``"goldfarb_pucel"``: the prior physical-via input impedance
      ``via_impedance_ohm(omega/2pi, via_radius_m, substrate)`` (~0.43 nH in-band, ~2x
      the ADS element). Needs the analytical leg's ``substrate``; returns ``None`` when
      no substrate is supplied. Retained for provenance / the gap-closure comparison.

    The defaults (``via_radius_m`` = the paper's 0.8 mm barrel; ``rho_relative`` =
    copper-vs-gold 0.7066) reproduce the emitter-accuracy-pass VIAGND (~0.19 nH on
    Rogers 4003C). First pass uses physical estimates; exact values to be finalized
    from the measured via geometry.
    """

    via_radius_m: float = 0.8e-3
    metal_thickness_m: float = 17.5e-6
    rho_relative: float | None = 0.706613906162
    analytical_mode: str = "effective_L"
    effective_L_h: float = _VIAGND_NOMINAL_EFFECTIVE_L_H
    effective_R_ohm: float = _VIAGND_NOMINAL_EFFECTIVE_R_OHM
    kind: str = "viagnd"

    def __post_init__(self) -> None:
        if self.analytical_mode not in ("effective_L", "goldfarb_pucel"):
            raise ValueError(
                f"analytical_mode must be 'effective_L' or 'goldfarb_pucel' "
                f"(got {self.analytical_mode!r})."
            )
        if float(self.effective_L_h) < 0.0 or float(self.effective_R_ohm) < 0.0:
            raise ValueError("effective_L_h / effective_R_ohm must be non-negative.")

    def emit(
        self, k: int, *, end_node: str, ground_node: str, l_via_H: float,
        width_m: float | None = None, substrate_name: str | None = None,
        n_parallel: int = 1,
    ) -> list[Element]:
        """``n_parallel`` identical VIAGND on ``end_node``: ``VIAGND_C{k}``, then
        ``VIAGND_C{k}_2`` ... ``_{n}``. A single via is emitted exactly as in 1.0.0."""
        if isinstance(n_parallel, bool) or int(n_parallel) != n_parallel or n_parallel < 1:
            raise ValueError(f"n_parallel must be a positive integer (got {n_parallel!r}).")
        w = float(width_m) if width_m is not None else 2.0 * float(self.via_radius_m)
        params: dict = {
            "d_m": 2.0 * float(self.via_radius_m),
            "w_m": w,
            "t_m": float(self.metal_thickness_m),
        }
        if self.rho_relative is not None:
            params["rho"] = float(self.rho_relative)
        if substrate_name is not None:
            params["substrate_ref"] = substrate_name
        return [
            Element(
                id=f"VIAGND_C{k}" if i == 1 else f"VIAGND_C{k}_{i}", type="VIAGND",
                pins=(ElementPin(pin=1, node=end_node),),
                parameters=dict(params),
            )
            for i in range(1, int(n_parallel) + 1)
        ]

    def analytical_termination(
        self, *, substrate: object | None = None, n_parallel: int = 1
    ) -> Callable[[float], complex] | None:
        """``n_parallel`` identical vias in parallel (no mutual coupling): ``Z / n_parallel``,
        the analytical counterpart of the ``n_parallel`` elements :meth:`emit` renders."""
        if isinstance(n_parallel, bool) or int(n_parallel) != n_parallel or n_parallel < 1:
            raise ValueError(f"n_parallel must be a positive integer (got {n_parallel!r}).")
        n = int(n_parallel)
        if self.analytical_mode == "effective_L":
            # Constant inductive load matching the ADS VIAGND element (frequency-flat).
            r_eff = float(self.effective_R_ohm) / n
            l_eff = float(self.effective_L_h) / n

            def _z_eff(omega: float) -> complex:
                # j*omega*L is odd in omega (correct at the Floquet negative harmonics);
                # at DC the via is a near-short (R_eff ~ 0).
                return complex(r_eff, float(omega) * l_eff)

            return _z_eff

        # goldfarb_pucel mode
        if substrate is None:
            # No substrate supplied -> fall back to the spec-extraction path (the
            # walker classifies VIAGND as a short). In practice the analytical leg
            # always passes the substrate.
            return None
        r = float(self.via_radius_m)
        sub = substrate
        two_pi = 2.0 * 3.141592653589793

        def _z(omega: float) -> complex:
            w = float(omega)
            f = w / two_pi
            if f <= 0.0:
                return complex(0.0, 0.0)  # DC short (via to ground)
            return complex(via_impedance_ohm(f, r, sub)) / n

        return _z


