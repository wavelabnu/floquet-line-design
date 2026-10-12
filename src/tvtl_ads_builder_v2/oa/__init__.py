"""OA-Python backend: build ADS workspaces via ``keysight.ads.de``.

This package is *additive* -- it sits alongside the hpeesofsim
flat-netlist emitter (:mod:`..netlist`) and the AEL inspection layer
(:mod:`..inspection`) and never modifies them. Its job is to build ADS
workspaces with the Keysight Python API and to cross-check every such
workspace against the hpeesofsim netlist of the same device spec.

Scope: the full D0-D4 diagnostic ladder, with automated ADS
netlist generation.

* :mod:`.oa_emitter`      -- spec-driven D0-D4 ADS workspace builder
                             (needs ADS).
* :mod:`.netlist_compare` -- ADS-free netlist equivalence check.
* :mod:`.oa_diagnostic_cli` -- the ``oa_diagnostic_cli`` entry point.
"""


