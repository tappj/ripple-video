import json
from pathlib import Path

import pytest

from cutdetect.export import _validate_boundaries, load_cut_frames


def test_validate_boundaries_produces_fourteen_segments_for_thirteen_cuts() -> None:
    cuts = tuple(range(10, 140, 10))

    boundaries = _validate_boundaries(cuts, 150)

    assert len(cuts) == 13
    assert len(boundaries) - 1 == 14
    assert boundaries == (0, *cuts, 150)


def test_validate_boundaries_rejects_duplicates_and_edges() -> None:
    with pytest.raises(ValueError):
        _validate_boundaries((10, 10), 20)
    with pytest.raises(ValueError):
        _validate_boundaries((0, 10), 20)


def test_load_cut_frames_reads_prediction_contract(tmp_path: Path) -> None:
    path = tmp_path / "predictions.json"
    path.write_text(json.dumps({"cuts": [{"frame": 10}, {"frame": 20}]}))

    assert load_cut_frames(path) == (10, 20)
