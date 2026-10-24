"""Configuration helpers."""

from __future__ import annotations

from pathlib import Path

import yaml


def load_yaml(path) -> dict:
    file_path = Path(path)
    with file_path.open("r", encoding="utf-8") as handle:
        payload = yaml.safe_load(handle)
    if not isinstance(payload, dict):
        raise ValueError("configuration root must be a mapping")
    return payload
