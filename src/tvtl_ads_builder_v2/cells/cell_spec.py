"""The cell-agnostic unit-cell interface.

A :class:`CellSpec` is a *pluggable unit-cell architecture*. The cascade
assembler builds an N-cell self-pumped Level-10 cascade by walking cells purely
through this interface; it owns only the cascade-level scaffolding (input port,
inter-cell connective I_Probes, output port + termination, matching networks,
sources, HB controller) and never hardcodes a cell's internal node names or
element topology.

A concrete ``CellSpec`` is constructed once per design -- it carries the
per-design geometry + device context -- and answers, for each 1-based cell
index ``k``:

* :meth:`cell_nodes` -- the ``Node`` declarations this cell introduces.
* :meth:`cell_elements` -- the cell's ``Element`` list (its ABCD assembly),
  given the assembler-provided ``input_net`` and the cell's own
  :meth:`output_terminal_node`.
* :meth:`ground_nodes` -- node ids the assembler must declare as ground.
* :meth:`name_nodes` -- nodes to tag with ``NAME`` elements (diagnostic probes).
* :meth:`varactor_terminal_nodes` -- the ``(top, bottom)`` node pair across the
  varactor junction; the pump-delivery (kappa) extraction and the standing-wave
  diagnostic read the junction voltage as ``V(top) - V(bottom)`` here.
* :meth:`output_terminal_node` -- the cell's downstream terminal node (the
  assembler chains cell ``k``'s output terminal to cell ``k+1`` via an
  inter-cell I_Probe).

Only :class:`~.single_shunt_varactor.SingleShuntVaractorCell` implements this
today. Enriched PT-paper cells (coupled lines, multi-varactor, ...) implement
the same five-method contract and drop straight into the assembler.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

from ..device_spec import Element, Node

__all__ = ["CellSpec"]


class CellSpec(ABC):
    """Interface for one pluggable unit-cell architecture (see module docstring)."""

    @abstractmethod
    def cell_nodes(self, k: int) -> list[Node]:
        """Node declarations introduced by cell ``k`` (1-based).

        Excludes the cell's *input* node (assembler-owned connective tissue);
        includes the cell's output terminal, branch, varactor-middle, return,
        per-segment stub, end, and per-cell ground nodes.
        """

    @abstractmethod
    def cell_elements(self, k: int, *, input_net: str, output_net: str) -> list[Element]:
        """The ``Element`` list (ABCD assembly) of cell ``k``.

        ``input_net`` is supplied by the assembler (the upstream node). Pass
        the cell's own :meth:`output_terminal_node` as ``output_net``.
        """

    @abstractmethod
    def ground_nodes(self, k: int) -> list[str]:
        """Node ids the assembler must add to the spec's ground set for cell ``k``."""

    @abstractmethod
    def name_nodes(self, k: int) -> list[str]:
        """Nodes to tag with ``NAME`` elements for cell ``k`` (diagnostic probes)."""

    @abstractmethod
    def varactor_terminal_nodes(self, k: int) -> tuple[str, str]:
        """``(top, bottom)`` extraction node pair across cell ``k``'s varactor junction.

        The junction voltage at the pump (and signal) is ``V(top) - V(bottom)``;
        ``top`` is the post-package-divider node on the SDD's in-line pin and
        ``bottom`` is the SDD's stub-side pin.
        """

    @abstractmethod
    def output_terminal_node(self, k: int) -> str:
        """The cell's downstream terminal node (chained to cell ``k+1``)."""


