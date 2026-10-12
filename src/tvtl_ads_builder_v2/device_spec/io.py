"""Device-spec on-disk I/O.

Device specs are stored as JSON, the canonical on-disk form. The file
extension is ``.device_spec.json``; the schema matches the in-memory
:class:`DeviceSpec` one-to-one.

``dump_device_spec`` is deterministic: the same input produces a
byte-identical file (keys sorted, two-space indentation, trailing
newline).
"""

from __future__ import annotations

import json
from pathlib import Path

from .schema import DeviceSpec
from .validator import validate_device_spec


DEVICE_SPEC_FILE_EXTENSION = ".device_spec.json"


def load_device_spec(path: str | Path) -> DeviceSpec:
    """Load and validate a device spec from disk.

    Parameters
    ----------
    path : str or Path
        Path to a JSON file produced by :func:`dump_device_spec` (or
        equivalently authored by hand).

    Returns
    -------
    DeviceSpec

    Raises
    ------
    DeviceSpecError
        Propagated from :func:`validate_device_spec` if the on-disk
        document violates an invariant.
    """

    path = Path(path)
    payload = json.loads(path.read_text(encoding="utf-8"))
    spec = DeviceSpec.from_dict(payload)
    validate_device_spec(spec)
    return spec


def dump_device_spec(spec: DeviceSpec, path: str | Path) -> Path:
    """Validate and write a device spec to disk in deterministic JSON.

    Parameters
    ----------
    spec : DeviceSpec
    path : str or Path
        Destination path. Parent directories are created if necessary.

    Returns
    -------
    Path
        The resolved destination path.
    """

    validate_device_spec(spec)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(spec.to_dict(), indent=2, sort_keys=True) + "\n"
    path.write_text(text, encoding="utf-8")
    return path


