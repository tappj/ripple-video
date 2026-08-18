from __future__ import annotations

import pytest

from cutdetect.pipeline.hosting import hosting_settings


def test_hosting_settings_use_container_environment() -> None:
    settings = hosting_settings(
        {
            "PORT": "7860",
            "RIPPLE_DATA_DIR": "/data/jobs",
            "RIPPLE_CACHE_DIR": "/data/cache",
            "RIPPLE_MAX_UPLOAD_MIB": "128",
            "RIPPLE_ACCESS_PASSWORD": "  shared-pass  ",
        }
    )

    assert settings.host == "0.0.0.0"
    assert settings.port == 7860
    assert str(settings.output_root) == "/data/jobs"
    assert settings.max_upload_bytes == 128 * 1024 * 1024
    assert settings.access_password == "shared-pass"


@pytest.mark.parametrize("name", ["PORT", "RIPPLE_MAX_UPLOAD_MIB"])
def test_hosting_settings_reject_nonpositive_numbers(name: str) -> None:
    with pytest.raises(ValueError, match="greater than zero"):
        hosting_settings({name: "0"})
