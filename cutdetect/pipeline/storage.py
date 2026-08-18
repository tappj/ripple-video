"""Storage boundary with a local-disk implementation."""

from __future__ import annotations

import json
import shutil
import urllib.request
from pathlib import Path
from typing import Protocol


class Storage(Protocol):
    """Operations needed by the pipeline independent of its backing store."""

    def path(self, key: str) -> Path: ...

    def copy_in(self, source: Path, key: str) -> Path: ...

    def write_json(self, key: str, value: object) -> Path: ...

    def download_https(self, url: str, key: str) -> Path: ...


class LocalDiskStorage:
    """Durable local storage rooted at one configured directory."""

    def __init__(self, root: str | Path) -> None:
        self.root = Path(root).expanduser().resolve()
        self.root.mkdir(parents=True, exist_ok=True)

    def path(self, key: str) -> Path:
        candidate = (self.root / key).resolve()
        if not candidate.is_relative_to(self.root):
            raise ValueError(f"storage key escapes root: {key}")
        candidate.parent.mkdir(parents=True, exist_ok=True)
        return candidate

    def copy_in(self, source: Path, key: str) -> Path:
        destination = self.path(key)
        shutil.copy2(source, destination)
        return destination

    def write_json(self, key: str, value: object) -> Path:
        destination = self.path(key)
        destination.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")
        return destination

    def download_https(self, url: str, key: str) -> Path:
        if not url.startswith("https://"):
            raise ValueError("Runway output URL must use HTTPS")
        destination = self.path(key)
        temporary = destination.with_suffix(destination.suffix + ".part")
        request = urllib.request.Request(url, headers={"User-Agent": "cutdetect/0.1"})
        with urllib.request.urlopen(request, timeout=180) as response, temporary.open("wb") as out:
            shutil.copyfileobj(response, out)
        temporary.replace(destination)
        return destination
