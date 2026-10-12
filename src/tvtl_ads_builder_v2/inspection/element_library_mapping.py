"""Map TVTL element types to ADS AEL symbol library paths.

Each path is the argument ADS's ``de_init_item`` takes -- the
``library:Model:symbol`` triple. Every entry is taken verbatim from
the ``de_init_item`` calls in the recorded ADS 2026 macros
(``recorded.dem`` / ``recorded2.dem`` in the reference workspace).

The map covers the element vocabulary of Levels 1-10, so the inspection
deck never needs a discovery step: TVTL's ``MTEE`` is ADS's
``MTEE_ADS``; ports are ``TermG``; the substrate / analysis controllers
(``MSUB``, ``S_Param``, ``SweepPlan``) and a circuit-ground symbol
(``GROUND``) are also placeable items.
"""

from __future__ import annotations

# TVTL element type  ->  ADS "library:Model:symbol" path for de_init_item.
ELEMENT_AEL_PATHS: dict[str, str] = {
    "MLIN": "ads_tlines:MLIN:symbol",
    "MTEE": "ads_tlines:MTEE_ADS:symbol",
    "MLOC": "ads_tlines:MLOC:symbol",
    "VIAGND": "ads_tlines:VIAGND:symbol",
    "MSUB": "ads_tlines:MSUB:symbol",
    "TermG": "ads_simulation:TermG:symbol",
    "S_Param": "ads_simulation:S_Param:symbol",
    "SweepPlan": "ads_simulation:SweepPlan:symbol",
    "GROUND": "ads_rflib:GROUND:symbol",
    "R": "ads_rflib:R:symbol",
    "L": "ads_rflib:L:symbol",
    "C": "ads_rflib:C:symbol",
    "SDD2P": "ads_datacmps:SDD2P:symbol",
    "VAR": "ads_datacmps:VAR:symbol",
    "I_Probe": "ads_rflib:I_Probe:symbol",
}


def ael_path_for_element(element_type: str) -> str:
    """Return the ADS AEL symbol path for a TVTL element type.

    Parameters
    ----------
    element_type : str
        A TVTL element type (``MLIN``, ``MTEE``, ``TermG``, ...) or one
        of the placeable controllers / ground symbol (``MSUB``,
        ``S_Param``, ``SweepPlan``, ``GROUND``).

    Returns
    -------
    str
        The ``library:Model:symbol`` path for ``de_init_item``.

    Raises
    ------
    KeyError
        If ``element_type`` has no recorded AEL symbol path.
    """

    try:
        return ELEMENT_AEL_PATHS[element_type]
    except KeyError:
        raise KeyError(
            f"no ADS AEL symbol path recorded for element type "
            f"{element_type!r}; known types: {sorted(ELEMENT_AEL_PATHS)}"
        ) from None


