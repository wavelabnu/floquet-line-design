"""Pluggable unit-cell architectures for the self-pumped Level-10 cascade.

A :class:`~.cell_spec.CellSpec` describes ONE unit-cell
architecture; the cascade assembler
(:func:`tvtl_ads_builder_v2.parity.self_pumped_design.optimized_self_pumped_spec`)
walks cells purely through this interface and never reaches into a cell's
internal node naming or element topology. Only
:class:`~.single_shunt_varactor.SingleShuntVaractorCell` exists today (the paper
structural class); enriched PT-paper cells become additional implementations of
the same interface -- the assembler does not change. This is the bridge to the
PT paper: enriched cells are plugins.
"""

from .cell_spec import CellSpec
from .single_shunt_varactor import SingleShuntVaractorCell

__all__ = ["CellSpec", "SingleShuntVaractorCell"]


