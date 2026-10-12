"""YAML I/O for chain model configs.

PyYAML is a declared project dependency (``PyYAML>=6.0`` in ``pyproject.toml``), so
chain model configs are authored as ``.yaml`` -- the canonical form for the
design-selectable registry.
"""

from __future__ import annotations

from pathlib import Path

import yaml

from .registry import ChainModelConfig

__all__ = ["load_chain_config", "dump_chain_config"]


def load_chain_config(path: str | Path) -> ChainModelConfig:
    """Load a :class:`ChainModelConfig` from a YAML file."""

    payload = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"chain config {path!r} must be a YAML mapping.")
    return ChainModelConfig.from_dict(payload)


def dump_chain_config(config: ChainModelConfig, path: str | Path) -> Path:
    """Write a :class:`ChainModelConfig` to a YAML file (round-trips ``load``)."""

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        yaml.safe_dump(config.to_dict(), sort_keys=False), encoding="utf-8",
    )
    return path


