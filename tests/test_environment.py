import os
from pathlib import Path

import pytest

from cutdetect.environment import EnvironmentFileError, load_env_file


def test_loads_owner_only_env_without_executing_shell(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = tmp_path / ".env"
    path.write_text("# secrets\nRUNWAYML_API_SECRET='safe value; $(false)'\n", encoding="utf-8")
    path.chmod(0o600)
    monkeypatch.delenv("RUNWAYML_API_SECRET", raising=False)

    assert load_env_file(path) == path
    assert os.environ["RUNWAYML_API_SECRET"] == "safe value; $(false)"


def test_process_environment_takes_precedence(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = tmp_path / ".env"
    path.write_text("RUNWAYML_API_SECRET=file-value\n", encoding="utf-8")
    path.chmod(0o600)
    monkeypatch.setenv("RUNWAYML_API_SECRET", "process-value")

    load_env_file(path)

    assert os.environ["RUNWAYML_API_SECRET"] == "process-value"


def test_rejects_group_or_world_readable_env(tmp_path: Path) -> None:
    path = tmp_path / ".env"
    path.write_text("RUNWAYML_API_SECRET=value\n", encoding="utf-8")
    path.chmod(0o644)

    with pytest.raises(EnvironmentFileError, match="unsafe permissions"):
        load_env_file(path)
