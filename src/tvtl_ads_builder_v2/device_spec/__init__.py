"""Device-spec layer: schema dataclasses, validator, JSON/YAML I/O.

The device spec is the architectural contract between TVTL and the ADS
builder. Reading, writing, and validating it lives here.
"""

from .schema import (
    Analysis,
    DeviceSpec,
    Element,
    ElementPin,
    FrequencyGrid,
    HBSingleToneSpec,
    HBTwoToneSpec,
    Node,
    P1ToneSource,
    Port,
    Substrate,
    VarBlockSpec,
    VarEntry,
)
from .validator import (
    DeviceSpecError,
    validate_device_spec,
    validate_hb_single_tone,
    validate_hb_two_tone,
    validate_p1tone_source,
    validate_var_block,
)
from .var_resolver import resolve_spec
from .sweep import SweepSpec, with_var_overrides
from .io import (
    DEVICE_SPEC_FILE_EXTENSION,
    dump_device_spec,
    load_device_spec,
)

__all__ = [
    "Analysis",
    "DEVICE_SPEC_FILE_EXTENSION",
    "DeviceSpec",
    "DeviceSpecError",
    "Element",
    "ElementPin",
    "FrequencyGrid",
    "HBSingleToneSpec",
    "HBTwoToneSpec",
    "Node",
    "P1ToneSource",
    "Port",
    "Substrate",
    "SweepSpec",
    "VarBlockSpec",
    "VarEntry",
    "dump_device_spec",
    "load_device_spec",
    "resolve_spec",
    "with_var_overrides",
    "validate_device_spec",
    "validate_hb_single_tone",
    "validate_hb_two_tone",
    "validate_p1tone_source",
    "validate_var_block",
]


