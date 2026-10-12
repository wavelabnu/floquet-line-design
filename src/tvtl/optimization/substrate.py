"""Substrate configuration for the decoupled design workflow.

Lightweight extensibility hook: substrate parameters are a
single immutable configuration object threaded through every optimization
stage, rather than Rogers-4003C constants hardcoded throughout the stage code.
Swapping substrates means changing this object; multi-substrate support
is out of scope.

The default :data:`ROGERS_4003C` matches the fabricated paper stack-up used by
:class:`tvtl_ads_builder_v2.parity.canonical_circuits.Level10SubstrateSpec`
(``eps_r = 3.38``, ``h = 1.54 mm``, ``tan_delta = 0.0027``, copper 17.5 um,
``sigma = 5.8e7 S/m``), so an optimization run on the default substrate is
apples-to-apples with the existing Level-10 canonicals.
"""

from __future__ import annotations

from dataclasses import dataclass

from scipy.optimize import brentq

from tvtl.materials import Substrate as _TvtlSubstrate
from tvtl.microstrip.models import hammerstad_kirschning

__all__ = ["SubstrateConfig", "ROGERS_4003C", "microstrip_width_for_z0"]


@dataclass(frozen=True)
class SubstrateConfig:
    """Microstrip substrate parameters (SI units).

    Field names and defaults mirror
    :class:`...parity.canonical_circuits.Level10SubstrateSpec` so the two are
    interchangeable; this class is the optimization layer's dependency-light
    handle (it imports nothing from the v2 builder).

    Parameters
    ----------
    eps_r : float
        Relative permittivity. Default 3.38 (Rogers 4003C).
    h_m : float
        Dielectric thickness [m]. Default 1.54 mm.
    tan_delta : float
        Loss tangent. Default 0.0027.
    sigma_S_per_m : float
        Conductor conductivity [S/m]. Default 5.8e7 (copper).
    t_m : float
        Conductor thickness [m]. Default 17.5 um (1/2 oz).
    rough_m : float
        Conductor surface roughness [m]. Default 0.0.
    mur : float
        Relative permeability. Default 1.0.
    name : str
        Substrate label used in emitted device specs. Default ``"msub1"``.
    """

    eps_r: float = 3.38
    h_m: float = 1.54e-3
    tan_delta: float = 0.0027
    sigma_S_per_m: float = 5.8e7
    t_m: float = 17.5e-6
    rough_m: float = 0.0
    mur: float = 1.0
    name: str = "msub1"

    def __post_init__(self) -> None:
        if float(self.eps_r) <= 0.0:
            raise ValueError(f"eps_r must be positive (got {self.eps_r}).")
        if float(self.h_m) <= 0.0:
            raise ValueError(f"h_m must be positive (got {self.h_m}).")
        if float(self.t_m) < 0.0:
            raise ValueError(f"t_m must be non-negative (got {self.t_m}).")
        if float(self.sigma_S_per_m) <= 0.0:
            raise ValueError(
                f"sigma_S_per_m must be positive (got {self.sigma_S_per_m})."
            )

    def stub_substrate_mapping(self) -> dict[str, float]:
        """Substrate mapping consumed by the analytical stub helper.

        Returns the dict shape
        :func:`tvtl_ads_builder_v2.analytical.floquet_stub.Y_stub_at_freq`
        reads (``eps_r``, ``h_m``, and the optional ``t_m`` copper thickness);
        the remaining keys mirror the v2 ``Substrate.parameters`` schema and
        are harmless extras.
        """

        return {
            "eps_r": float(self.eps_r),
            "h_m": float(self.h_m),
            "t_m": float(self.t_m),
            "tan_delta": float(self.tan_delta),
            "sigma_S_per_m": float(self.sigma_S_per_m),
            "rough_m": float(self.rough_m),
            "mur": float(self.mur),
        }

    def to_tvtl_substrate(self) -> _TvtlSubstrate:
        """Return the :class:`tvtl.materials.Substrate` for the line/MTEE models.

        Consumed by the scalar passive-cell helpers (Stage 2/3:
        ``tee_reference_plane_single``, ``calc_gamma_legacy``). Mirrors
        :func:`...analytical.parametric_gain._tvtl_substrate_from_spec`.
        """

        return _TvtlSubstrate(
            epsilon_r=float(self.eps_r),
            thickness_m=float(self.h_m),
            copper_thickness_m=float(self.t_m),
            loss_tangent=float(self.tan_delta),
            conductivity_s_per_m=float(self.sigma_S_per_m),
        )


ROGERS_4003C = SubstrateConfig()


def microstrip_width_for_z0(
    z0_target_ohm: float,
    substrate: SubstrateConfig,
    frequency_hz: float,
    *,
    w_lo_m: float = 5.0e-5,
    w_hi_m: float = 2.0e-2,
) -> float:
    """Inverse Hammerstad-Kirschning: microstrip width [m] for a target Z0.

    Numerically inverts the (monotone-decreasing) ``Z0(width)`` of
    :func:`tvtl.microstrip.models.hammerstad_kirschning` at ``frequency_hz`` by
    Brent root-find on ``Z0(w) - z0_target`` over ``[w_lo_m, w_hi_m]``. Used by
    Stage 2 (Z_TL -> main width) and Stage 3 (Z_qw / Z_stub -> width).

    Raises
    ------
    ValueError
        If ``z0_target_ohm`` is not achievable in ``[w_lo_m, w_hi_m]`` (the
        target is outside the substrate's realizable microstrip Z0 band); the
        message reports the achievable band so the caller can record a
        framework boundary.
    """

    m = substrate.stub_substrate_mapping()

    def _z0(w_m: float) -> float:
        z0_arr, _ = hammerstad_kirschning(
            float(w_m), m["t_m"], m["h_m"], m["eps_r"], [float(frequency_hz)]
        )
        return float(z0_arr[0])

    z0_at_hi = _z0(w_hi_m)  # widest -> lowest Z0
    z0_at_lo = _z0(w_lo_m)  # narrowest -> highest Z0
    if not (z0_at_hi <= float(z0_target_ohm) <= z0_at_lo):
        raise ValueError(
            f"Z0 target {z0_target_ohm:.3g} ohm not realizable on this substrate "
            f"at {frequency_hz/1e9:.3g} GHz; achievable band ~[{z0_at_hi:.1f}, "
            f"{z0_at_lo:.1f}] ohm for widths [{w_lo_m*1e3:.3g}, {w_hi_m*1e3:.3g}] mm."
        )
    return float(
        brentq(lambda w: _z0(w) - float(z0_target_ohm), w_lo_m, w_hi_m, xtol=1e-8, rtol=1e-10)
    )


