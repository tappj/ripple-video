"""Environment-driven entry point for container hosting."""

from __future__ import annotations

import os
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

from cutdetect.environment import load_env_file
from cutdetect.pipeline.app import PipelineStudioConfig, run_pipeline_studio


@dataclass(frozen=True, slots=True)
class HostingSettings:
    """Validated settings shared by Docker, Render, and other container hosts."""

    host: str
    port: int
    output_root: Path
    cache_dir: Path
    max_upload_bytes: int
    access_password: str | None


def _positive_int(value: str, name: str) -> int:
    try:
        result = int(value)
    except ValueError as error:
        raise ValueError(f"{name} must be an integer") from error
    if result <= 0:
        raise ValueError(f"{name} must be greater than zero")
    return result


def hosting_settings(environ: Mapping[str, str] | None = None) -> HostingSettings:
    """Read hosting configuration without making filesystem or network changes."""
    values = os.environ if environ is None else environ
    max_upload_mib = _positive_int(
        values.get("RIPPLE_MAX_UPLOAD_MIB", "256"), "RIPPLE_MAX_UPLOAD_MIB"
    )
    password = values.get("RIPPLE_ACCESS_PASSWORD", "").strip() or None
    return HostingSettings(
        host=values.get("RIPPLE_HOST", "0.0.0.0"),
        port=_positive_int(values.get("PORT", "10000"), "PORT"),
        output_root=Path(values.get("RIPPLE_DATA_DIR", ".cutdetect/pipeline_studio")),
        cache_dir=Path(values.get("RIPPLE_CACHE_DIR", ".cutdetect/cache")),
        max_upload_bytes=max_upload_mib * 1024 * 1024,
        access_password=password,
    )


def main() -> None:
    """Start Ripple on the host-provided address and writable data paths."""
    load_env_file()
    settings = hosting_settings()
    run_pipeline_studio(
        config=PipelineStudioConfig(
            host=settings.host,
            port=settings.port,
            max_upload_bytes=settings.max_upload_bytes,
            access_password=settings.access_password,
        ),
        output_root=settings.output_root,
        cache_dir=settings.cache_dir,
        open_browser=False,
    )


if __name__ == "__main__":
    main()
