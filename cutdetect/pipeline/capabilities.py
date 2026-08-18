"""Single source of truth for regeneration model constraints and pricing."""

from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class ModelCaps:
    """Capabilities that affect grouping, validation, and UI controls."""

    model_id: str
    endpoint: str
    min_duration_s: float
    max_duration_s: float
    duration_is_integer: bool
    supports_reference_audio: bool
    supports_internal_cuts: bool
    supported_ratios: tuple[str, ...]
    supported_resolutions: tuple[str, ...]
    notes: str


SEEDANCE_RATIOS = (
    "992:432",
    "864:496",
    "752:560",
    "640:640",
    "560:752",
    "496:864",
    "1470:630",
    "1280:720",
    "1112:834",
    "960:960",
    "834:1112",
    "720:1280",
    "2206:946",
    "1920:1080",
    "1664:1248",
    "1440:1440",
    "1248:1664",
    "1080:1920",
    "3840:1646",
    "3840:2160",
    "3840:2880",
    "3840:3840",
    "2880:3840",
    "2160:3840",
)

HAILUO_RATIOS = ("adaptive", "21:9", "16:9", "4:3", "1:1", "3:4", "9:16")

# The Model Router accepts model-agnostic aspect ratios. Keep these separate from
# the direct Seedance pixel dimensions so old direct jobs remain resumable.
ROUTER_ASPECT_RATIOS = ("21:9", "16:9", "4:3", "1:1", "3:4", "9:16")
ROUTER_RESOLUTIONS: dict[str, tuple[str, ...]] = {
    "seedance2": ("480p", "720p", "1080p", "4K"),
    "hailuo3": ("768P", "2K"),
}

# CHECKPOINT A found that Seedance retained only one of two known hard cuts.
# The capability remains False as an honest QC signal. The user-directed 4-10 second
# grouping policy accepts that risk and retains every hard boundary in the source media.
MODEL_CAPABILITIES: dict[str, ModelCaps] = {
    "seedance2": ModelCaps(
        model_id="seedance2",
        endpoint="/v1/text_to_video",
        min_duration_s=4.0,
        max_duration_s=15.0,
        duration_is_integer=True,
        supports_reference_audio=True,
        supports_internal_cuts=False,
        supported_ratios=SEEDANCE_RATIOS,
        supported_resolutions=("480p", "720p", "1080p", "4K"),
        notes=(
            "Internal cuts were not fully reliable in Checkpoint A. Current policy groups "
            "complete source sections anyway and requires boundary QC during review."
        ),
    ),
    "hailuo3": ModelCaps(
        model_id="hailuo3",
        endpoint="/v1/text_to_video",
        min_duration_s=5.0,
        max_duration_s=15.0,
        duration_is_integer=True,
        supports_reference_audio=True,
        supports_internal_cuts=False,
        supported_ratios=HAILUO_RATIOS,
        supported_resolutions=("768P", "2K"),
        notes=(
            "Direct four-reference generation was proven with an unpositioned face image. "
            "A grouped source hard cut was retained with measurable timing drift."
        ),
    ),
}


def credit_cost(
    model_id: str,
    duration_s: int,
    ratio: str,
    *,
    reference_video_duration_s: float = 0.0,
) -> int:
    """Return the current static direct-model estimate in Runway credits."""
    if model_id == "seedance2":
        if ratio.lower() in {"480p", "720p"}:
            rate = 36
        elif ratio.lower() == "1080p":
            rate = 40
        elif ratio.lower() == "4k":
            rate = 150
        else:
            width, height = map(int, ratio.split(":", maxsplit=1))
            short_edge = min(width, height)
            rate = 150 if short_edge >= 2160 else 40 if short_edge >= 1080 else 36
        return duration_s * rate
    if model_id == "hailuo3":
        rate = 15 if ratio.upper() == "2K" else 10
        # Hailuo bills output and reference video at the same rate, plus two
        # credits for the one target-face reference used by this pipeline.
        return math.ceil((duration_s + reference_video_duration_s) * rate + 2)
    raise ValueError(f"unsupported model: {model_id}")


def closest_seedance_ratio(width: int, height: int) -> str:
    """Choose a supported output size with the closest source aspect ratio."""
    if width <= 0 or height <= 0:
        raise ValueError("source dimensions must be positive")
    source = width / height
    candidates = [ratio for ratio in SEEDANCE_RATIOS if max(map(int, ratio.split(":"))) <= 1920]
    return min(
        candidates,
        key=lambda ratio: abs(int(ratio.split(":")[0]) / int(ratio.split(":")[1]) - source),
    )
