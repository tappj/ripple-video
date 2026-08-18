"""Minimal, non-executing loader for owner-protected local secrets."""

from __future__ import annotations

import os
import re
import stat
from pathlib import Path

_NAME = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")


class EnvironmentFileError(ValueError):
    """Raised when a local environment file is unsafe or malformed."""


def _parse_value(raw: str, *, path: Path, line_number: int) -> str:
    value = raw.strip()
    if not value:
        return ""
    if value[0] in {"'", '"'}:
        if len(value) < 2 or value[-1] != value[0]:
            raise EnvironmentFileError(f"{path}:{line_number}: unmatched quote")
        return value[1:-1]
    return value


def load_env_file(path: str | Path = ".env") -> Path | None:
    """Load KEY=VALUE records without shell expansion or command execution.

    Existing process variables win, allowing CI and explicit shell exports to
    override the local file. On POSIX, any group/world access is rejected.
    """
    source = Path(path).expanduser().resolve()
    if not source.exists():
        return None
    if not source.is_file():
        raise EnvironmentFileError(f"environment path is not a file: {source}")
    if os.name == "posix":
        mode = stat.S_IMODE(source.stat().st_mode)
        if mode & 0o077:
            raise EnvironmentFileError(f"unsafe permissions on {source}; run: chmod 600 {source}")
    for line_number, raw_line in enumerate(source.read_text(encoding="utf-8").splitlines(), 1):
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("export "):
            line = line[7:].lstrip()
        name, separator, raw_value = line.partition("=")
        name = name.strip()
        if not separator or not _NAME.fullmatch(name):
            raise EnvironmentFileError(f"{source}:{line_number}: expected KEY=VALUE")
        os.environ.setdefault(name, _parse_value(raw_value, path=source, line_number=line_number))
    return source
