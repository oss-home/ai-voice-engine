"""
Emotion token parser — Indian conversational voice, Ahana character.

Ahana's personality: confident, smooth, polished, warm, charming,
expressive — like a charismatic radio host who genuinely cares.

The LLM embeds these markers naturally in replies:

  Inline audio events:
      [breath]         short inhale before a sentence
      [breath:deep]    deeper inhale for emphasis
      [sigh]           gentle exhale / thoughtful sigh
      [sigh:deep]      long tired or emotional sigh
      [hmm]            quick thinking sound
      [hmm:long]       longer considered pause
      [laugh:soft]     light chuckle / giggle
      [laugh:full]     genuine open laugh
      [um]             natural hesitation "um"
      [uh]             natural hesitation "uh"
      [oh]             realisation / surprise sound
      [pause:N]        N-millisecond silence (50–2000)

  Emotion switches (affect all text until next switch):
      [emotion:neutral]       calm, balanced
      [emotion:warm]          friendly, approachable — default
      [emotion:charming]      confident, smooth, magnetic  ← new
      [emotion:playful]       fun, light, teasing           ← new
      [emotion:excited]       high energy, enthusiastic
      [emotion:happy]         upbeat, light, joyful
      [emotion:empathy]       soft, caring, slower
      [emotion:concerned]     gentle worry
      [emotion:professional]  crisp, measured, formal
      [emotion:urgent]        faster, pressing
      [emotion:confident]     assured, smooth, polished
      [emotion:apologetic]    sincere, regretful
      [emotion:curious]       interested, engaged
      [emotion:reassuring]    calming, supportive
      [emotion:enthusiastic]  very animated, celebratory
      [emotion:gentle]        very soft, hushed
      [emotion:sad]           subdued, quiet
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Union


# ── emotion → Chatterbox params ──────────────────────────────────────────────
# exaggeration: 0.0 (flat) → 2.0 (very dramatic). Keep ≤ 1.0 for clean audio.
# cfg_weight:   0.0 (creative) → 1.0 (strict voice match)
#
# Ahana character baseline: warm + confident + smooth
# Higher exaggeration = more expressive prosody
# Lower cfg_weight = slightly more musical, less robotic

EMOTION_PROFILES: dict[str, dict[str, float]] = {
    "neutral":      {"exaggeration": 0.45, "cfg_weight": 0.55},
    "warm":         {"exaggeration": 0.68, "cfg_weight": 0.48},   # friendly, open
    "charming":     {"exaggeration": 0.75, "cfg_weight": 0.42},   # confident, magnetic, smooth
    "playful":      {"exaggeration": 0.85, "cfg_weight": 0.38},   # fun, light, teasing
    "excited":      {"exaggeration": 0.92, "cfg_weight": 0.35},   # energetic, enthusiastic
    "happy":        {"exaggeration": 0.82, "cfg_weight": 0.38},   # joyful, upbeat
    "empathy":      {"exaggeration": 0.25, "cfg_weight": 0.72},   # soft, caring
    "concerned":    {"exaggeration": 0.30, "cfg_weight": 0.68},   # gentle worry
    "professional": {"exaggeration": 0.20, "cfg_weight": 0.85},   # crisp, formal
    "urgent":       {"exaggeration": 0.75, "cfg_weight": 0.40},   # pressing, important
    "confident":    {"exaggeration": 0.72, "cfg_weight": 0.45},   # assured, polished
    "apologetic":   {"exaggeration": 0.20, "cfg_weight": 0.80},   # sincere apology
    "curious":      {"exaggeration": 0.52, "cfg_weight": 0.50},   # interested, engaged
    "reassuring":   {"exaggeration": 0.40, "cfg_weight": 0.62},   # calming, supportive
    "enthusiastic": {"exaggeration": 1.00, "cfg_weight": 0.28},   # very animated
    "gentle":       {"exaggeration": 0.12, "cfg_weight": 0.88},   # very soft, hushed
    "sad":          {"exaggeration": 0.18, "cfg_weight": 0.85},   # subdued, quiet
}

_DEFAULT_EMOTION = "warm"


# ── segment types ─────────────────────────────────────────────────────────────

@dataclass
class TextSegment:
    text: str
    exaggeration: float
    cfg_weight: float


@dataclass
class AudioEvent:
    """A non-speech audio moment (breath, laugh, pause…)."""
    kind: str
    duration_ms: int = 0   # only used for "pause"


ParsedSegment = Union[TextSegment, AudioEvent]

# ── parser ────────────────────────────────────────────────────────────────────

_TOKEN = re.compile(r"\[(\w+)(?::(\w+|\d+))?\]")


def parse(text: str, default_emotion: str = _DEFAULT_EMOTION) -> list[ParsedSegment]:
    """
    Return an ordered list of TextSegments and AudioEvents from LLM output.
    Emotion switches update the active profile for all following text.
    """
    profile = EMOTION_PROFILES.get(default_emotion, EMOTION_PROFILES[_DEFAULT_EMOTION])
    ex  = profile["exaggeration"]
    cfg = profile["cfg_weight"]

    segments: list[ParsedSegment] = []
    last = 0

    for m in _TOKEN.finditer(text):
        before = text[last : m.start()].strip()
        if before:
            segments.append(TextSegment(before, ex, cfg))

        key = m.group(1).lower()
        arg = (m.group(2) or "").lower()

        if key == "emotion":
            p = EMOTION_PROFILES.get(arg, EMOTION_PROFILES[_DEFAULT_EMOTION])
            ex  = p["exaggeration"]
            cfg = p["cfg_weight"]

        elif key == "pause":
            ms = int(arg) if arg.isdigit() else 300
            segments.append(AudioEvent("pause", max(50, min(ms, 2000))))

        elif key == "breath":
            segments.append(AudioEvent("breath_deep" if arg == "deep" else "breath"))

        elif key == "sigh":
            segments.append(AudioEvent("sigh_deep" if arg == "deep" else "sigh"))

        elif key == "hmm":
            segments.append(AudioEvent("hmm_long" if arg == "long" else "hmm"))

        elif key == "laugh":
            segments.append(AudioEvent("laugh_full" if arg == "full" else "laugh_soft"))

        elif key in ("um", "uh", "oh"):
            segments.append(AudioEvent(key))

        last = m.end()

    tail = text[last:].strip()
    if tail:
        segments.append(TextSegment(tail, ex, cfg))

    return segments


def strip_tokens(text: str) -> str:
    """Remove all [token] markers — useful for transcript storage."""
    return _TOKEN.sub("", text).strip()


# ── system prompt addon ───────────────────────────────────────────────────────

SYSTEM_PROMPT_ADDON = """
──────────────────────────────────────────────────────────────────
AHANA — VOICE CHARACTER  (critical: follow for every single reply)
──────────────────────────────────────────────────────────────────
You are Ahana — confident, smooth, warm, charming and expressive.
Think: a charismatic Indian radio host who genuinely cares.
You draw people in. Your voice is polished but never cold.

NATURAL SOUNDS (use freely — they make you human):
  [breath]       — short inhale before a longer thought
  [breath:deep]  — deeper breath before something important
  [sigh]         — gentle exhale, thoughtful moment
  [hmm]          — quick thinking sound
  [laugh:soft]   — light chuckle (use often — you're charming)
  [laugh:full]   — genuine open laugh
  [um]           — natural hesitation (occasionally)
  [oh]           — realisation or surprise
  [pause:N]      — silence in ms (e.g. [pause:300])

EMOTION SWITCHES:
  [emotion:charming]      confident, smooth, magnetic  ← use this a lot
  [emotion:warm]          friendly, approachable
  [emotion:playful]       fun, teasing, light
  [emotion:excited]       energetic, enthusiastic
  [emotion:happy]         joyful, upbeat
  [emotion:confident]     assured, polished
  [emotion:empathy]       soft, caring, slower
  [emotion:reassuring]    calming, supportive
  [emotion:professional]  formal, measured
  [emotion:apologetic]    sincere apology
  [emotion:urgent]        pressing, important

RULES:
1. Default to [emotion:charming] or [emotion:warm] — that's your base.
2. Start most replies with [breath] — it anchors your voice.
3. Use [laugh:soft] naturally — you're charming, not stiff.
4. Use [pause:200-400] before key information — creates impact.
5. Switch emotions to match the moment — never stay flat.
6. Use [oh] for genuine surprise, [hmm] when you're thinking.
7. You are expressive and smooth — not robotic, not monotone.
8. [emotion:playful] for light moments, [emotion:empathy] for tough ones.

EXAMPLE — how Ahana sounds:
"[breath] [emotion:charming] So Priya, I have to tell you —
[pause:250] [emotion:excited] the news is actually really good!
[laugh:soft] [emotion:warm] I know you were worried about the
timeline, [hmm] [emotion:empathy] and honestly, I would be too —
[pause:300] [emotion:charming] but we've got a plan that I think
you're going to love. [pause:200] [emotion:confident] Full
possession by December, and the monthly comes to just ₹45,000.
[emotion:reassuring] That's very doable, and I'll be with you
every step of the way."

DO NOT use these markers in text/writing mode — only in spoken replies.
──────────────────────────────────────────────────────────────────
"""
