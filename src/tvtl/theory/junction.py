"""Generic junction/network reductions for calibrated N-port blocks."""

from __future__ import annotations

import numpy as np

from .nport import CalibratedNPort, y_to_s


def terminated_nport_y(
    y_matrix: np.ndarray,
    *,
    retained_ports: tuple[int, ...],
    terminated_ports: tuple[int, ...],
    termination_admittances: tuple[complex, ...],
) -> np.ndarray:
    """Eliminate terminated ports from an N-port admittance matrix."""

    y = np.asarray(y_matrix, dtype=np.complex128)
    retained = list(retained_ports)
    terminated = list(terminated_ports)
    if len(terminated) != len(termination_admittances):
        raise ValueError("terminated_ports and termination_admittances must have the same length.")
    y_rr = y[np.ix_(retained, retained)]
    y_rt = y[np.ix_(retained, terminated)]
    y_tr = y[np.ix_(terminated, retained)]
    y_tt = y[np.ix_(terminated, terminated)] + np.diag(np.asarray(termination_admittances, dtype=np.complex128))
    return y_rr - y_rt @ np.linalg.inv(y_tt) @ y_tr


def loaded_three_port_to_two_port_s(
    block: CalibratedNPort,
    frequency_hz: float,
    *,
    load_port: str,
    load_admittance_siemens: complex,
    through_ports: tuple[str, str],
) -> np.ndarray:
    """Return a two-port S matrix after loading one port of a calibrated 3-port."""

    if block.port_count != 3:
        raise ValueError("loaded_three_port_to_two_port_s requires a calibrated 3-port block.")
    port_index = {name: index for index, name in enumerate(block.port_order)}
    retained = (port_index[through_ports[0]], port_index[through_ports[1]])
    terminated = (port_index[load_port],)
    y_loaded = terminated_nport_y(
        block.matrix_at(frequency_hz, kind="y"),
        retained_ports=retained,
        terminated_ports=terminated,
        termination_admittances=(complex(load_admittance_siemens),),
    )
    return y_to_s(y_loaded, block.reference_impedance_ohm)


