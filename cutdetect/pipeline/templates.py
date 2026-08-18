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
    version=2,
    body=(
        "Use Video 1 as the reference video. Preserve the original video's camera angle, "
        "framing, background, timing, body movements, gestures, facial expressions, and "
        "pacing. Replace the person in Video 1 with the person from Image 1. Use Audio 1 "
        "only as the voice reference; keep the exact spoken words, timing, and pacing from "
        "Video 1."
    ),
    editable_by_user=True,
)

PROMPT_TEMPLATES: dict[str, PromptTemplate] = {UGC_CLONE_V1.id: UGC_CLONE_V1}
