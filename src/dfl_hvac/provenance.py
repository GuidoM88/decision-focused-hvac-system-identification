"""Minimal provenance records for external simulation inputs."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path


def sha256(path) -> str:
    file_path = Path(path)
    digest = hashlib.sha256()
    with file_path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


@dataclass(frozen=True)
class ExternalFile:
    path: str
    sha256: str
    source: str | None = None


def describe_external_file(path, *, source: str | None = None) -> ExternalFile:
    file_path = Path(path).expanduser().resolve()
    if not file_path.is_file():
        raise FileNotFoundError(file_path)
    return ExternalFile(path=str(file_path), sha256=sha256(file_path), source=source)


def write_run_manifest(path, *, files=(), metadata=None) -> None:
    payload = {
        "created_at_utc": datetime.now(UTC).isoformat(),
        "files": [asdict(item) for item in files],
        "metadata": {} if metadata is None else dict(metadata),
    }
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
