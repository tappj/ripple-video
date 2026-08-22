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
_CLONE_OPENING = (
    "Recreate Video 1 as one continuous take at 1.0x speed: the same person, in the same "
    "place, moving and speaking exactly as they do in Video 1, with the face of Image 1."
)

_CLONE_IDENTITY = (
    "IDENTITY. Image 1 provides the face only - facial features, head shape, hairline, and "
    "skin tone - identical in every frame. One single face carries the whole clip, from the "
    "first frame to the last."
)

_CLONE_FRAME = (
    "FRAME. Everything else comes from Video 1 and only from Video 1: the same body and "
    "posture, the same clothing in the same cut and color, the same accessories, eyewear, "
    "jewelry, and hands, the same held objects, the same background with the same objects in "
    "the same positions, the same lighting and color grade. What Video 1 shows stays exactly "
    "as it is; what Video 1 does not show never appears."
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

_CLONE_EXCLUSIONS = "No on-screen text, no subtitles, no second person."


UGC_CLONE_V1 = PromptTemplate(
    id="ugc_clone_v1",
    label="UGC Clone",
    version=7,
    body="\n\n".join(
        (
            _CLONE_OPENING,
            _CLONE_IDENTITY,
            _CLONE_FRAME,
            _CLONE_MOTION,
            (
                "SPEECH. Video 1 is the only source of the words. Speak Video 1's dialogue "
                "verbatim, in its language, word for word and in the same order, starting and "
                "ending on Video 1's timings, with the same pace, pauses, emphasis, and "
                "breaths. Lips match every syllable."
            ),
            (
                "VOICE. Audio 1 provides vocal identity only - timbre, pitch, and accent. "
                "Audio 1 is a voice sample, not a script; its words never appear in the "
                "output. One consistent voice, tone, and recording quality carries the whole "
                "clip."
            ),
            "SOUND. Reproduce Video 1's background ambience at its original level.",
            (
                "ENDING. The dialogue finishes before the final frame, then hold the closing "
                "moment naturally - no filler words, repeated phrases, or invented gestures to "
                "fill time."
            ),
            f"{_CLONE_EXCLUSIONS[:-1]}, no background music.",
        )
    ),
    editable_by_user=True,
)

# Ripple remuxes the original audio whenever no target voice is supplied, so this
# variant drops every audio instruction and keeps only the articulation that has
# to stay locked to the source dialogue.
UGC_CLONE_NO_VOICE_V1 = PromptTemplate(
    id="ugc_clone_v1_no_voice",
    label="UGC Clone (source audio)",
    version=7,
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
    editable_by_user=False,
)

UGC_PRODUCT_CLONE_V1 = PromptTemplate(
    id="ugc_product_clone_v1",
    label="UGC Product Clone",
    version=2,
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
    return f"{base}\n\nADDITIONAL DIRECTION. {cleaned}"


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
