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
    version=5,
    body=(
        "Recreate Video 1 exactly at its original duration. Video 1 is the only source for "
        "dialogue, wording, timing, motion, cuts, framing, background, and background audio. "
        "Do not slow, extend, loop, or add footage. Use Image 1 only for facial identity. "
        "Do not invent or alternate faces, glasses, hair, clothing, or accessories. If Audio "
        "1 is provided, use only its voice identity and tone. Ignore its words completely; "
        "speak only the exact words from Video 1 at the same timestamps. Change nothing else."
    ),
    editable_by_user=True,
)

PROMPT_TEMPLATES: dict[str, PromptTemplate] = {UGC_CLONE_V1.id: UGC_CLONE_V1}


def strict_generation_prompt(direction: str) -> str:
    """Keep clone constraints mandatory while preserving optional user direction."""
    cleaned = direction.strip()
    if not cleaned or cleaned == UGC_CLONE_V1.body:
        return UGC_CLONE_V1.body
    return f"{UGC_CLONE_V1.body}\nAdditional direction: {cleaned}"
