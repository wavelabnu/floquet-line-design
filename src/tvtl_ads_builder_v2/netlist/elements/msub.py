"""Emit the ``MSUB`` substrate definition line for hpeesofsim.

The parameter spellings follow Keysight ADS's documented MSUB model:
``H`` (substrate height, mm), ``Er`` (relative permittivity), ``Mur``
(relative permeability), ``Cond`` (conductor conductivity, S/m), ``T``
(strip thickness, mm), ``TanD`` (dielectric loss tangent), ``Rough``
(conductor surface roughness, mm). The unit suffixes (``mm``) are
emitted explicitly so the simulator never has to guess.
"""

from __future__ import annotations

from ...device_spec import Substrate


def emit_msub(substrate: Substrate) -> str:
    """Return a single-line MSUB directive for ``substrate``.

    Parameters
    ----------
    substrate : Substrate
        The Level-1 device-spec substrate. Only the
        ``hammerstad_kirschning`` model is supported here.
    """

    if substrate.model != "hammerstad_kirschning":
        raise ValueError(
            f"emit_msub does not implement substrate model '{substrate.model}'."
        )
    params = substrate.parameters
    h_mm = float(params["h_m"]) * 1.0e3
    er = float(params["eps_r"])
    mur = float(params.get("mur", 1.0))
    cond = float(params["sigma_S_per_m"])
    t_mm = float(params["t_m"]) * 1.0e3
    tan_d = float(params["tan_delta"])
    rough_mm = float(params.get("rough_m", 0.0)) * 1.0e3
    # hpeesofsim expects ``model <name> MSUB <params>`` (a model
    # declaration) rather than ``MSUB:<name>`` (which it would parse as
    # an instance reference to a non-existent MSUB model). The MLIN
    # references the substrate by name via its ``Subst="<name>"`` field.
    return (
        f"model {substrate.name} MSUB "
        f"H={h_mm:.12g} mm "
        f"Er={er:.12g} "
        f"Mur={mur:.12g} "
        f"Cond={cond:.12g} "
        f"T={t_mm:.12g} mm "
        f"TanD={tan_d:.12g} "
        f"Rough={rough_mm:.12g} mm"
    )


