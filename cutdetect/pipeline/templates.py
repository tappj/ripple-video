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
    version=6,
    body=(
        "Clone Video 1 exactly at the same length. Keep its framing, camera, background, "
        "motion, timing, cuts, dialogue, and background audio unchanged. Replace only the "
        "person's face with Image 1. If Audio 1 is provided, replace only the speaking voice "
        "with Audio 1 while preserving the exact words and timing. Change nothing else."
    ),
    editable_by_user=True,
)

UGC_PRODUCT_CLONE_V1 = PromptTemplate(
    id="ugc_product_clone_v1",
    label="UGC Product Clone",
    version=1,
    body=(
        "Keep Video 1 exactly the same, but switch the avatar to Image 1 and make them "
        "hold the product from Image 2."
    ),
    editable_by_user=True,
)

_SUPERSEDED_DEFAULT_PROMPTS = {
    (
        "Recreate Video 1 exactly at its original duration. Video 1 is the only source for "
        "dialogue, wording, timing, motion, cuts, framing, background, and background audio. "
        "Do not slow, extend, loop, or add footage. Use Image 1 only for facial identity. "
        "Do not invent or alternate faces, glasses, hair, clothing, or accessories. If Audio "
        "1 is provided, use only its voice identity and tone. Ignore its words completely; "
        "speak only the exact words from Video 1 at the same timestamps. Change nothing else."
    )
}

PROMPT_TEMPLATES: dict[str, PromptTemplate] = {
    UGC_CLONE_V1.id: UGC_CLONE_V1,
    UGC_PRODUCT_CLONE_V1.id: UGC_PRODUCT_CLONE_V1,
}


def strict_generation_prompt(direction: str) -> str:
    """Keep clone constraints mandatory while preserving optional user direction."""
    cleaned = direction.strip()
    if not cleaned or cleaned == UGC_CLONE_V1.body or cleaned in _SUPERSEDED_DEFAULT_PROMPTS:
        return UGC_CLONE_V1.body
    return f"{UGC_CLONE_V1.body}\nAdditional direction: {cleaned}"


def generation_prompt(template_id: str, direction: str) -> str:
    """Apply the mandatory base prompt for the selected Ripple experience."""
    if template_id == UGC_CLONE_V1.id:
        return strict_generation_prompt(direction)
    if template_id != UGC_PRODUCT_CLONE_V1.id:
        raise ValueError(f"unknown prompt template: {template_id}")
    cleaned = direction.strip()
    if not cleaned or cleaned == UGC_PRODUCT_CLONE_V1.body:
        return UGC_PRODUCT_CLONE_V1.body
    return f"{UGC_PRODUCT_CLONE_V1.body}\nAdditional direction: {cleaned}"
