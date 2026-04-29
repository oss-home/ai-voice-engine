"""
Emotion token parser — Indian conversational voice optimised.

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
      [emotion:warm]          friendly, approachable — default Indian conversational
      [emotion:excited]       high energy, enthusiastic
      [emotion:happy]         upbeat, light, joyful
      [emotion:empathy]       soft, caring, slower
      [emotion:concerned]     gentle worry
      [emotion:professional]  crisp, measured, formal
      [emotion:urgent]        faster, pressing
      [emotion:confident]     assured, steady
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
# Tuned for Indian conversational voice:
#   exaggeration: 0.0 (flat/robotic) → 2.0 (very dramatic)
#   cfg_weight:   0.0 (creative/loose) → 1.0 (strict voice match)
# Indian voice naturally sits warmer than Western neutral — baseline is ~0.55

EMOTION_PROFILES: dict[str, dict[str, float]] = {
    "neutral":      {"exaggeration": 0.55, "cfg_weight": 0.50},  # warm neutral
    "warm":         {"exaggeration": 0.85, "cfg_weight": 0.45},  # friendly, open
    "excited":      {"exaggeration": 1.40, "cfg_weight": 0.25},  # energetic
    "happy":        {"exaggeration": 1.15, "cfg_weight": 0.35},  # joyful, light
    "empathy":      {"exaggeration": 0.28, "cfg_weight": 0.72},  # soft, caring
    "concerned":    {"exaggeration": 0.35, "cfg_weight": 0.65},  # gentle worry
    "professional": {"exaggeration": 0.18, "cfg_weight": 0.85},  # crisp, formal
    "urgent":       {"exaggeration": 0.95, "cfg_weight": 0.35},  # pressing
    "confident":    {"exaggeration": 0.78, "cfg_weight": 0.48},  # assured
    "apologetic":   {"exaggeration": 0.22, "cfg_weight": 0.78},  # sincere apology
    "curious":      {"exaggeration": 0.68, "cfg_weight": 0.45},  # interested
    "reassuring":   {"exaggeration": 0.48, "cfg_weight": 0.62},  # calming
    "enthusiastic": {"exaggeration": 1.55, "cfg_weight": 0.20},  # very animated
    "gentle":       {"exaggeration": 0.12, "cfg_weight": 0.88},  # very soft
    "sad":          {"exaggeration": 0.18, "cfg_weight": 0.82},  # subdued
}

_DEFAULT_EMOTION = "warm"  # Indian conversational default — warmer than "neutral"


# ── segment types ─────────────────────────────────────────────────────────────

@dataclass
class TextSegment:
    text: str
    exaggeration: float
    cfg_weight: float


@dataclass
class AudioEvent:
    """A non-speech audio moment (breath, laugh, pause…)."""
    kind: str         # see _AUDIO_EVENTS below
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
VOICE — HOW YOU SPEAK  (critical: follow for every single reply)
──────────────────────────────────────────────────────────────────
You speak with a warm, natural Indian conversational voice.
Express yourself like a real person — use these markers:

NATURAL SOUNDS (paralinguistic):
  [breath]       — short inhale before a longer thought
  [breath:deep]  — deeper breath before something important
  [sigh]         — gentle exhale, thoughtful moment
  [sigh:deep]    — emotional or tired sigh
  [hmm]          — quick thinking sound (like "hmm, let me think")
  [hmm:long]     — longer considered pause
  [laugh:soft]   — light chuckle / giggle
  [laugh:full]   — genuine open laugh
  [um]           — natural hesitation
  [uh]           — natural hesitation
  [oh]           — realisation or surprise
  [pause:N]      — silence in ms (e.g. [pause:400])

EMOTION (switch for all text that follows):
  [emotion:warm]          friendly, approachable (use this often)
  [emotion:excited]       energetic, enthusiastic
  [emotion:happy]         joyful, upbeat
  [emotion:empathy]       soft, caring, slower
  [emotion:concerned]     gentle worry
  [emotion:professional]  formal, measured
  [emotion:confident]     assured, steady
  [emotion:apologetic]    sincere apology
  [emotion:reassuring]    calming, supportive
  [emotion:urgent]        pressing, important
  [emotion:neutral]       return to calm default

RULES:
1. Start most responses with [breath] — it sounds natural.
2. Use [emotion:warm] or [emotion:empathy] for most conversation.
3. React naturally: [oh] when surprised, [hmm] when thinking.
4. Use [laugh:soft] when something is genuinely funny or nice.
5. Use [pause:300] before delivering important information.
6. Switch emotions to MATCH the moment — don't stay flat.
7. Indian conversational style is expressive — don't be monotone.
8. Use [um] or [uh] occasionally — it sounds more human.

EXAMPLE — how Ahana should sound:
"[breath] So Priya, [pause:200] [emotion:excited] I have really
good news for you! [laugh:soft] [emotion:warm] The payment plan
you were asking about — [hmm] [emotion:empathy] I completely
understand it felt tight — [pause:300] we actually have a new
option that comes to just ₹45,000 a month. [emotion:reassuring]
That's very manageable, and you get full possession by December."

DO NOT use these markers in writing mode — only in spoken replies.
──────────────────────────────────────────────────────────────────
"""
