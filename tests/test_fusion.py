import numpy as np

from cutdetect.config import FusionConfig
from cutdetect.fusion import (
    fuse_normalized,
    local_robust_zscore,
    pick_boundaries,
    squash_zscores,
)


def test_local_robust_zscore_excludes_cut_from_own_baseline() -> None:
    values = np.zeros(11)
    values[5] = 10.0

    zscores = local_robust_zscore(values, window_radius=5, center_exclusion=2)

    assert zscores[5] > 1_000_000
    assert squash_zscores(zscores, 8.0)[5] == 1.0


def test_fusion_renormalizes_over_available_signals() -> None:
    normalized = {
        "first": np.asarray([1.0, np.nan]),
        "second": np.asarray([np.nan, 0.5]),
    }

    fused, agreement, available = fuse_normalized(normalized, {"first": 2.0, "second": 1.0})

    assert fused.tolist() == [1.0, 0.5]
    assert agreement.tolist() == [1, 0]
    assert available.tolist() == [2.0, 1.0]


def test_peak_selection_applies_hysteresis_and_high_score_nms() -> None:
    config = FusionConfig(
        tau_high=0.8,
        tau_low=0.5,
        agreement_count=2,
        min_shot_frames=3,
    )
    fused = np.asarray([0.0, 0.9, 0.0, 0.6, 0.0])
    agreement = np.asarray([0, 2, 0, 3, 0])

    assert pick_boundaries(fused, agreement, config) == (1,)
