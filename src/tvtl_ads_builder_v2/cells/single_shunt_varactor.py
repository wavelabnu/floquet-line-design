"""The single-shunt-varactor unit cell (the paper structural class).

The ONE :class:`~.cell_spec.CellSpec` implementation that
exists today: per cell, a main microstrip line split by an MTEE, with a single
shunt branch carrying the package ``L_pkg`` / ``R_series`` divider, a charge-SDD
varactor (pins 3/4 tied to a per-cell return net for KCL closure), an I_Probe,
the optimized stub chain, and the family termination (an inductive via to ground
for SHORT; floating open end for OPEN).

This is a verbatim lift of the per-cell node/element construction that used to
live inline in
:mod:`tvtl_ads_builder_v2.parity.self_pumped_design` (``_self_pumped_cell_nodes``
/ ``_self_pumped_cell_elements``). Making the cell the single source of truth --
and routing the cascade assembler through the :class:`CellSpec` interface --
keeps the A.3.1 anchor byte-identical (recovery gate) while opening the door for
enriched PT-paper cells as additional ``CellSpec`` implementations.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from tvtl.optimization.topology_generator import (
    Termination,
    TopologyFamily,
    combined_open_segment,
    stub_segments,
)
from tvtl.varactors import RealisticVaractor

from ..device_spec import Element, ElementPin, Node
from .cell_spec import CellSpec

if TYPE_CHECKING:
    from ..models.open_end import OpenEndModel
    from ..models.via import ViaModel

__all__ = ["SingleShuntVaractorCell", "DEFAULT_BRANCH_WIDTH_M"]

# Paper structural-class branch width (was ``_BRANCH_WIDTH_M`` in self_pumped_design).
DEFAULT_BRANCH_WIDTH_M: float = 1.93e-3


@dataclass(frozen=True)
class SingleShuntVaractorCell(CellSpec):
    """Paper structural-class cell: main MLIN + MTEE + one shunt varactor branch.

    Built once per design. ``params`` is the stub-geometry parameter vector for
    ``topology``; ``q_poly`` is the varactor's charge-polynomial body (CP folded);
    ``l_via_H`` is the inductive-via value for SHORT terminations (0 for OPEN).
    """

    topology: TopologyFamily
    params: object
    varactor: RealisticVaractor
    substrate_name: str
    main_width_m: float
    main_period_m: float
    q_poly: tuple
    q_poly_reference: str
    l_via_H: float
    branch_width_m: float = DEFAULT_BRANCH_WIDTH_M
    # Model-registry hooks: optional via / open-end model objects (the
    # chain model registry's single source). When BOTH are ``None`` (the default)
    # the cell renders the historical lumped ``L`` via / ideal-open termination
    # byte-for-byte (the recovery path). When a model is supplied, the cell delegates
    # the SHORT via (via_model) / OPEN far end (open_end_model) rendering to it.
    via_model: "ViaModel | None" = None
    open_end_model: "OpenEndModel | None" = None
    # Bias-return element: a resistor from the varactor's stub-side terminal (n_c{k}_v) to the cell
    # ground (RDCRET_C{k}), the DC path an OPEN stub lacks. None (default) emits nothing; an HB deck of an
    # OPEN family without it is refused (no_dc_return_for_varactor).
    bias_return_ohm: float | None = None

    # ------------------------------------------------------------------ #
    # Derived structural flags / internal node names                     #
    # ------------------------------------------------------------------ #
    @property
    def _emit_lpkg(self) -> bool:
        return float(self.varactor.L_pkg_h) > 0.0

    @property
    def _emit_rseries(self) -> bool:
        return float(self.varactor.R_series_ohm) > 0.0

    @property
    def _short(self) -> bool:
        return self.topology.termination is Termination.SHORT

    @property
    def _open(self) -> bool:
        return self.topology.termination is Termination.OPEN

    @property
    def _combined(self) -> bool:
        # COMBINED: the MLEF open-stub IN PARALLEL with a VIAGND short at the
        # stub end node (the reference circuit2 resonant termination). Neither the via-only
        # (SHORT) nor the open-only (OPEN) family has both at one node.
        return self.topology.termination is Termination.COMBINED

    def _end_node(self, k: int) -> str:
        if self._short:
            return f"n_c{k}_via_top"
        if self._combined:
            return f"n_c{k}_term"
        return f"n_c{k}_open"

    def _sdd_top_node(self, k: int) -> str:
        """The SDD in-line pin node, after the package divider (post-divider)."""
        if self._emit_rseries:
            return f"n_c{k}_rp"
        if self._emit_lpkg:
            return f"n_c{k}_lp"
        return f"n_c{k}_b"

    # ------------------------------------------------------------------ #
    # CellSpec interface                                                 #
    # ------------------------------------------------------------------ #
    def output_terminal_node(self, k: int) -> str:
        return f"n_c{k}_out"

    def varactor_terminal_nodes(self, k: int) -> tuple[str, str]:
        return (self._sdd_top_node(k), f"n_c{k}_v")

    def ground_nodes(self, k: int) -> list[str]:
        return [f"n_c{k}_g"]

    def name_nodes(self, k: int) -> list[str]:
        return [f"n_c{k}_b", f"n_c{k}_v"]

    def cell_nodes(self, k: int) -> list[Node]:
        """Per-cell node list (verbatim from ``_self_pumped_cell_nodes``).

        Adds the ``n_c{k}_ret`` per-cell return net used by the charge SDD's
        pin 3/4 self-tie (KCL closure).
        """
        end_node = self._end_node(k)
        nodes = [
            Node(id=f"n_c{k}_l"),
            Node(id=f"n_c{k}_r"),
            Node(id=f"n_c{k}_out"),
            Node(id=f"n_c{k}_b"),
            Node(id=f"n_c{k}_v"),
            Node(id=f"n_c{k}_ret"),  # SDD pins 3/4 self-tie (self-pumped specific)
        ]
        if self._emit_lpkg:
            nodes.append(Node(id=f"n_c{k}_lp"))
        if self._emit_rseries:
            nodes.append(Node(id=f"n_c{k}_rp"))
        for i in range(self.topology.n_segments):
            nodes.append(Node(id=f"n_c{k}_s{i}"))
        nodes.append(Node(id=end_node))
        nodes.append(Node(id=f"n_c{k}_g"))
        # Model-registry path: a non-default via / open-end model may introduce
        # internal nodes (e.g. the RLC via's series R-L node). The default models
        # (or no model) add none, so the reduced node list is unchanged.
        if self._short and self.via_model is not None:
            nodes.extend(self.via_model.extra_nodes(k, end_node))
        elif self._open and self.open_end_model is not None:
            nodes.extend(self.open_end_model.extra_nodes(k, end_node))
        elif self._combined:
            # COMBINED: the open stublet's far end (its open-end fringe / dangling end)
            # plus any extra nodes the via and open-end models introduce.
            ostub_end = f"n_c{k}_ostub"
            nodes.append(Node(id=ostub_end))
            if self.via_model is not None:
                nodes.extend(self.via_model.extra_nodes(k, end_node))
            if self.open_end_model is not None:
                nodes.extend(self.open_end_model.extra_nodes(k, ostub_end))
        return nodes

    def cell_elements(self, k: int, *, input_net: str, output_net: str) -> list[Element]:
        """Per-cell element list (verbatim from ``_self_pumped_cell_elements``).

        MLIN halves + MTEE + branch ``[L_pkg] -> [R_series] -> charge SDD (pins
        3/4 self-tied) -> I_Probe -> stub chain -> termination``.
        """
        end_node = self._end_node(k)
        half_period = float(self.main_period_m) / 2.0
        els: list[Element] = []

        els.append(Element(
            id=f"LINE_LEFT_C{k}", type="MLIN",
            pins=(ElementPin(pin=1, node=input_net),
                  ElementPin(pin=2, node=f"n_c{k}_l")),
            parameters={"w_m": float(self.main_width_m), "l_m": half_period,
                        "substrate_ref": self.substrate_name},
        ))
        els.append(Element(
            id=f"TEE_C{k}", type="MTEE",
            pins=(ElementPin(pin=1, node=f"n_c{k}_l"),
                  ElementPin(pin=2, node=f"n_c{k}_r"),
                  ElementPin(pin=3, node=f"n_c{k}_b")),
            parameters={"w1_m": float(self.main_width_m), "w2_m": float(self.main_width_m),
                        "w3_m": float(self.branch_width_m), "substrate_ref": self.substrate_name},
        ))
        els.append(Element(
            id=f"LINE_RIGHT_C{k}", type="MLIN",
            pins=(ElementPin(pin=1, node=f"n_c{k}_r"),
                  ElementPin(pin=2, node=output_net)),
            parameters={"w_m": float(self.main_width_m), "l_m": half_period,
                        "substrate_ref": self.substrate_name},
        ))

        # Branch: [L_pkg] -> [R_series] -> SDD (charge body, pins 3/4 self-tied)
        # -> I_Probe -> stub -> termination.
        cur = f"n_c{k}_b"
        if self._emit_lpkg:
            els.append(Element(
                id=f"LPKG_C{k}", type="L",
                pins=(ElementPin(pin=1, node=cur),
                      ElementPin(pin=2, node=f"n_c{k}_lp")),
                parameters={"l_h": float(self.varactor.L_pkg_h)},
            ))
            cur = f"n_c{k}_lp"
        if self._emit_rseries:
            els.append(Element(
                id=f"RPKG_C{k}", type="R",
                pins=(ElementPin(pin=1, node=cur),
                      ElementPin(pin=2, node=f"n_c{k}_rp")),
                parameters={"r_ohm": float(self.varactor.R_series_ohm)},
            ))
            cur = f"n_c{k}_rp"
        els.append(Element(
            id=f"VARACTOR_C{k}", type="SDD2P",
            pins=(
                ElementPin(pin=1, node=cur),
                ElementPin(pin=2, node=f"n_c{k}_v"),
                ElementPin(pin=3, node=f"n_c{k}_ret"),
                ElementPin(pin=4, node=f"n_c{k}_ret"),
            ),
            parameters={
                "q_poly_coefficients": tuple(self.q_poly),
                "q_poly_reference": self.q_poly_reference,
            },
        ))
        els.append(Element(
            id=f"IPR_VAR_C{k}", type="I_Probe",
            pins=(ElementPin(pin=1, node=f"n_c{k}_v"),
                  ElementPin(pin=2, node=f"n_c{k}_s0")),
            parameters={},
        ))

        segs = stub_segments(self.topology, self.params)
        n_seg = len(segs)
        for i, seg in enumerate(segs):
            seg_in = f"n_c{k}_s{i}"
            seg_out = end_node if i == n_seg - 1 else f"n_c{k}_s{i + 1}"
            els.append(Element(
                id=f"STUB_C{k}_{i}", type="MLIN",
                pins=(ElementPin(pin=1, node=seg_in),
                      ElementPin(pin=2, node=seg_out)),
                parameters={"w_m": float(seg.width_m), "l_m": float(seg.length_m),
                            "substrate_ref": self.substrate_name},
            ))

        # Termination. Default (no model): the historical lumped-L via (SHORT) /
        # ideal dangling open (OPEN), byte-for-byte. Model-registry path: delegate
        # to the via model (SHORT) / open-end model (OPEN). COMBINED: an open
        # stublet (MLIN + open-end fringe) IN PARALLEL with a via, both at end_node.
        last_w = float(segs[-1].width_m) if segs else float(self.branch_width_m)
        if self._short:
            els.extend(self._emit_via(k, end_node, f"n_c{k}_g", last_w))
        elif self._combined:
            els.extend(self._emit_combined_termination(k, end_node, last_w))
        elif self.open_end_model is not None:
            # OPEN family with a non-default open-end model (the default ideal open
            # renders nothing -- the stub MLIN simply dangles).
            els.extend(self.open_end_model.emit(
                k, end_node=end_node, width_m=last_w,
                substrate_name=self.substrate_name,
            ))
        if self.bias_return_ohm is not None:
            if not float(self.bias_return_ohm) > 0.0:
                raise ValueError("bias_return_ohm must be positive")
            els.append(Element(
                id=f"RDCRET_C{k}", type="R",
                pins=(ElementPin(pin=1, node=f"n_c{k}_v"),
                      ElementPin(pin=2, node=f"n_c{k}_g")),
                parameters={"r_ohm": float(self.bias_return_ohm)},
            ))
        return els

    # ------------------------------------------------------------------ #
    # Termination renderers                                              #
    # ------------------------------------------------------------------ #
    def _emit_via(
        self, k: int, end_node: str, ground_node: str, width_m: float
    ) -> list[Element]:
        """The via-to-ground termination for cell ``k``.

        Default (``via_model is None``): the historical lumped ``L`` via
        (``LVIA_C{k}``), byte-for-byte. Model-registry path: delegate to the via
        model (e.g. the ADS ``VIAGND``). Shared by the SHORT and COMBINED branches.
        """

        if self.via_model is None:
            return [Element(
                id=f"LVIA_C{k}", type="L",
                pins=(ElementPin(pin=1, node=end_node),
                      ElementPin(pin=2, node=ground_node)),
                parameters={"l_h": float(self.l_via_H)},
            )]
        return list(self.via_model.emit(
            k, end_node=end_node, ground_node=ground_node,
            l_via_H=float(self.l_via_H),
            width_m=width_m, substrate_name=self.substrate_name,
            n_parallel=max(1, self.topology.n_via_parallel),
        ))

    def _emit_combined_termination(
        self, k: int, end_node: str, last_w: float
    ) -> list[Element]:
        """The COMBINED MLEF-open-stub-IN-PARALLEL-WITH-VIAGND resonant termination.

        At the stub end node ``end_node`` two branches hang in parallel (the reference
        circuit2 resonant trap):

        * the **open stublet** -- a series ``MLIN`` of the COMBINED open-segment
          geometry (``combined_open_segment``: width/length) from ``end_node`` to its
          own far end ``n_c{k}_ostub``, terminated by the open-end model's fringe
          (``MlefOpen`` -> an ``MLEF`` end-effect element; the ideal open dangles).
          This MLIN+MLEF decomposition is the framework's standard open-stub rendering
          (the MLIN carries the line length, the MLEF the open-end fringe);
        * the **via to ground** at ``end_node`` (``self._emit_via``; ``ViagndVia`` ->
          the ADS ``VIAGND`` physical via, or the lumped ``L`` default).

        Together they form the high-Q ``MLEF ∥ VIAGND`` parallel trap this design uses --
        the structural feature neither the via-only nor the open-only family has.
        """

        els: list[Element] = []
        stublet = combined_open_segment(self.topology, self.params)
        # combined_open_segment is non-None for COMBINED families (n_params carries the
        # extra (w, l)); guard defensively against a degenerate params vector.
        ostub_end = f"n_c{k}_ostub"
        if stublet is not None:
            els.append(Element(
                id=f"OSTUB_C{k}", type="MLIN",
                pins=(ElementPin(pin=1, node=end_node),
                      ElementPin(pin=2, node=ostub_end)),
                parameters={"w_m": float(stublet.width_m),
                            "l_m": float(stublet.length_m),
                            "substrate_ref": self.substrate_name},
            ))
            if self.open_end_model is not None:
                els.extend(self.open_end_model.emit(
                    k, end_node=ostub_end, width_m=float(stublet.width_m),
                    substrate_name=self.substrate_name,
                ))
        els.extend(self._emit_via(k, end_node, f"n_c{k}_g", last_w))
        return els


