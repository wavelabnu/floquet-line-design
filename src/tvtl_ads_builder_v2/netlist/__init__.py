"""Netlist emitters for ADS Builder v2 device specs.

Two output forms are provided:

* :func:`emit_hpeesofsim_netlist` -- the flat hpeesofsim ``.ckt`` deck,
  the simulator input the parity gate uses.
* :func:`emit_ads_native_netlist` -- the ADS-native ``.net`` form
  (``MLIN2``/``MTEE_ADS`` model names, full parameter sets), mirroring
  the netlist ADS writes for a drawn schematic.
"""

from .ads_native import emit_ads_native_netlist
from .emit import emit_flat_hb_netlist, emit_hpeesofsim_netlist

__all__ = [
    "emit_ads_native_netlist",
    "emit_flat_hb_netlist",
    "emit_hpeesofsim_netlist",
]


