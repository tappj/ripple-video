"""Versioned prompt records used by both CLI and future UI."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class PromptTemplate:
    """An editable, versioned generation prompt."""

    id: str
    label: str
    version: int
    body: str
    editable_by_user: bool


UGC_CLONE_V1 = PromptTemplate(
    id="ugc_clone_v1",
    label="UGC Clone",
    version=3,
    body=(
        "Clone Video 1 exactly at the same length. Keep its framing, camera, background, "
        "motion, timing, cuts, dialogue, and background audio unchanged. Replace only the "
        "person's face with Image 1. If Audio 1 is provided, replace only the speaking voice "
        "with Audio 1 while preserving the exact words and timing. Change nothing else."
    ),
    editable_by_user=True,
)

PROMPT_TEMPLATES: dict[str, PromptTemplate] = {UGC_CLONE_V1.id: UGC_CLONE_V1}
