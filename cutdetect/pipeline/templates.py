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


# Seedance regenerates audio natively and treats a reference clip as a style and
# cadence hint rather than a source of words, so every reference gets one
# exclusive job and the clip states what it carries instead of what to avoid.
# The person belongs entirely to Image 1; the frame around them belongs entirely
# to Video 1.
_CLONE_OPENING = (
    "Recreate Video 1 as one continuous take at 1.0x speed: the person from Image 1 performing "
    "Video 1 exactly - the same place, the same movements, the same words, at the same moments."
)

_CLONE_IDENTITY = (
    "PERSON. Image 1 provides the whole person: facial features, head shape, hairline, hair, "
    "skin tone, build, and everything they wear in Image 1 - the same clothing in the same cut "
    "and color, the same accessories, eyewear, and jewelry. Every part of the person that "
    "Image 1 shows carries over exactly and stays identical in every frame, from the first to "
    "the last. Where Image 1 stops short of the full body, continue the same outfit plainly "
    "and consistently with what Image 1 shows, and hold it fixed for the whole clip."
)

_CLONE_FRAME = (
    "FRAME. Everything that is not the person comes from Video 1 and only from Video 1: the "
    "same background with the same objects in the same positions, the same held objects, the "
    "same lighting, shadows, and color grade, and the person standing or sitting at the same "
    "place and scale within the shot. What Video 1 shows around the person stays exactly as it "
    "is; what Video 1 does not show never appears."
)

_CLONE_MOTION = (
    "MOTION. Match Video 1 beat for beat - the same gestures at the same timestamps, the same "
    "head turns, blinks, and body position at every moment. Match its camera exactly: same "
    "framing, distance, angle, lens, and handheld drift. Cut only where Video 1 cuts."
)

_CLONE_MOUTH = (
    "MOUTH. Reproduce Video 1's speech articulation exactly - the same mouth and jaw movement, "
    "syllable for syllable, on the same timings, with the same pauses and breaths."
)

# Seedance 2 rejects any promptText longer than this.
MAX_PROMPT_CHARS = 3500

_CLONE_EXCLUSIONS = "No on-screen text, no subtitles, no second person."

_CLONE_MUSIC_EXCLUSION = " No new, added, or replacement music."


UGC_CLONE_V1 = PromptTemplate(
    id="ugc_clone_v1",
    label="UGC Clone",
    version=9,
    body="\n\n".join(
        (
            _CLONE_OPENING,
            _CLONE_IDENTITY,
            _CLONE_FRAME,
            _CLONE_MOTION,
            (
                f"{_CLONE_MOUTH[:-1]}, so the performance stays in sync with Video 1's "
                "original dialogue."
            ),
            (
                "ENDING. The performance finishes before the final frame, then hold the "
                "closing moment naturally - no invented gestures or repeated movement to fill "
                "time."
            ),
            _CLONE_EXCLUSIONS,
        )
    ),
    editable_by_user=True,
)

# Audio is now mastered outside the video model, so both legacy call paths use the
# same visual/lip-sync-only prompt and all provider-generated audio is discarded.
UGC_CLONE_NO_VOICE_V1 = PromptTemplate(
    id="ugc_clone_v1_no_voice",
    label="UGC Clone (source audio)",
    version=9,
    body=UGC_CLONE_V1.body,
    editable_by_user=False,
)

UGC_PRODUCT_CLONE_V1 = PromptTemplate(
    id="ugc_product_clone_v1",
    label="UGC Product Clone",
    version=3,
    body="\n\n".join(
        (
            f"{_CLONE_OPENING[:-1]}, holding the product from Image 2.",
            _CLONE_IDENTITY,
            (
                "PRODUCT. Image 2 provides the held product only. Reproduce its exact shape, "
                "proportions, color, material, label, and text, unchanged in every frame, held "
                "in the same hand and the same position as the object the person holds in "
                "Video 1, moving with that hand on Video 1's timings. It stays the same single "
                "object from the first frame to the last."
            ),
            _CLONE_FRAME.replace("the same held objects, ", ""),
            _CLONE_MOTION,
            _CLONE_MOUTH,
            (
                "ENDING. The performance finishes before the final frame, then hold the "
                "closing moment naturally."
            ),
            _CLONE_EXCLUSIONS,
        )
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
    ),
    (
        "Clone Video 1 exactly at the same length. Keep its framing, camera, background, "
        "motion, timing, cuts, dialogue, and background audio unchanged. Replace only the "
        "person's face with Image 1. If Audio 1 is provided, replace only the speaking voice "
        "with Audio 1 while preserving the exact words and timing. Change nothing else."
    ),
    (
        "Keep Video 1 exactly the same, but switch the avatar to Image 1 and make them "
        "hold the product from Image 2."
    ),
}

PROMPT_TEMPLATES: dict[str, PromptTemplate] = {
    UGC_CLONE_V1.id: UGC_CLONE_V1,
    UGC_PRODUCT_CLONE_V1.id: UGC_PRODUCT_CLONE_V1,
}


def _apply_base(base: str, direction: str, *, defaults: tuple[str, ...]) -> str:
    """Keep the base constraints mandatory while preserving optional user direction."""
    cleaned = direction.strip()
    if not cleaned or cleaned in defaults or cleaned in _SUPERSEDED_DEFAULT_PROMPTS:
        return base
    prefix = f"{base}\n\nADDITIONAL DIRECTION. "
    # Seedance rejects a prompt over its character limit outright, so the base
    # constraints keep their room and only the user's direction gives way.
    return f"{prefix}{cleaned}"[:MAX_PROMPT_CHARS].rstrip()


def strict_generation_prompt(direction: str, *, has_voice: bool = True) -> str:
    """Select the clone base that matches the supplied references, then add direction."""
    base = UGC_CLONE_V1.body if has_voice else UGC_CLONE_NO_VOICE_V1.body
    return _apply_base(
        base,
        direction,
        defaults=(UGC_CLONE_V1.body, UGC_CLONE_NO_VOICE_V1.body),
    )


def generation_prompt(template_id: str, direction: str, *, has_voice: bool = True) -> str:
    """Apply the mandatory base prompt for the selected Ripple experience."""
    if template_id == UGC_CLONE_V1.id:
        return strict_generation_prompt(direction, has_voice=has_voice)
    if template_id != UGC_PRODUCT_CLONE_V1.id:
        raise ValueError(f"unknown prompt template: {template_id}")
    return _apply_base(
        UGC_PRODUCT_CLONE_V1.body,
        direction,
        defaults=(UGC_PRODUCT_CLONE_V1.body,),
    )
