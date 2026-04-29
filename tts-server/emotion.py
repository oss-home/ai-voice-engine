"""
Emotion token parser.

The LLM is instructed to embed these markers naturally in its replies:

    Inline audio events:
        [breath]        natural inhale before a long sentence
        [breath:deep]   deeper inhale for a dramatic moment
        [laugh:soft]    soft chuckle
        [laugh:full]    genuine full laugh
        [sigh]          exhale / sigh
        [hmm]           thinking sound
        [pause:N]       N-millisecond silence (100–2000)

    Emotion switches (affect all text that follows until next switch):
        [emotion:neutral]       calm, balanced
        [emotion:excited]       high energy, enthusiastic
        [emotion:empathy]       warm, soft, slower
        [emotion:warm]          friendly, approachable
        [emotion:professional]  crisp, measured
        [emotion:happy]         upbeat, light
        [emotion:concerned]     caring worry
        [emotion:urgent]        faster, assertive
        [emotion:sad]           gentle, subdued
        [emotion:confident]     steady, authoritative

System-prompt snippet to paste into voice agents:

    Express yourself naturally using these emotion markers. Examples:
    "[breath] So what I can tell you is, [emotion:excited] this tower
    is our fastest seller right now! [laugh:soft] The numbers really
    speak for themselves. [pause:300] [emotion:empathy] I completely
    understand your concern about the timeline."
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Union

# ── emotion → Chatterbox params ──────────────────────────────────────────────

EMOTION_PROFILES: dict[str, dict[str, float]] = {
    "neutral":      {"exaggeration": 0.50, "cfg_weight": 0.50},
    "excited":      {"exaggeration": 1.40, "cfg_weight": 0.30},
    "empathy":      {"exaggeration": 0.30, "cfg_weight": 0.70},
    "warm":         {"exaggeration": 0.80, "cfg_weight": 0.50},
    "professional": {"exaggeration": 0.20, "cfg_weight": 0.80},
    "happy":        {"exaggeration": 1.10, "cfg_weight": 0.40},
    "concerned":    {"exaggeration": 0.40, "cfg_weight": 0.60},
    "urgent":       {"exaggeration": 0.90, "cfg_weight": 0.40},
    "sad":          {"exaggeration": 0.25, "cfg_weight": 0.75},
    "confident":    {"exaggeration": 0.70, "cfg_weight": 0.55},
}

# ── segment types ─────────────────────────────────────────────────────────────

@dataclass
class TextSegment:
    text: str
    exaggeration: float
    cfg_weight: float


@dataclass
class AudioEvent:
    """A non-speech audio moment (breath, laugh, pause…)."""
    kind: str        # breath | breath_deep | laugh_soft | laugh_full | sigh | hmm | pause
    duration_ms: int = 0   # only used for "pause"


ParsedSegment = Union[TextSegment, AudioEvent]

# ── parser ────────────────────────────────────────────────────────────────────

_TOKEN = re.compile(r"\[(\w+)(?::(\w+|\d+))?\]")


def parse(text: str, default_emotion: str = "neutral") -> list[ParsedSegment]:
    """
    Return an ordered list of TextSegments and AudioEvents derived from
    the raw LLM output.  Emotion switches update the active profile for
    all text that follows.
    """
    profile = EMOTION_PROFILES.get(default_emotion, EMOTION_PROFILES["neutral"])
    ex  = profile["exaggeration"]
    cfg = profile["cfg_weight"]

    segments: list[ParsedSegment] = []
    last = 0

    for m in _TOKEN.finditer(text):
        # flush accumulated text before this token
        before = text[last : m.start()].strip()
        if before:
            segments.append(TextSegment(before, ex, cfg))

        key = m.group(1).lower()
        arg = (m.group(2) or "").lower()

        if key == "emotion":
            p = EMOTION_PROFILES.get(arg, EMOTION_PROFILES["neutral"])
            ex  = p["exaggeration"]
            cfg = p["cfg_weight"]

        elif key == "pause":
            ms = int(arg) if arg.isdigit() else 300
            ms = max(50, min(ms, 2000))
            segments.append(AudioEvent("pause", ms))

        elif key == "breath":
            segments.append(AudioEvent("breath_deep" if arg == "deep" else "breath"))

        elif key == "laugh":
            segments.append(AudioEvent("laugh_full" if arg == "full" else "laugh_soft"))

        elif key in ("sigh", "hmm"):
            segments.append(AudioEvent(key))

        last = m.end()

    tail = text[last:].strip()
    if tail:
        segments.append(TextSegment(tail, ex, cfg))

    return segments


def strip_tokens(text: str) -> str:
    """Remove all [token] markers — useful for transcript storage."""
    return _TOKEN.sub("", text).strip()


SYSTEM_PROMPT_ADDON = """
──────────────────────────────────────────────────────
VOICE EXPRESSION GUIDE  (follow this for every reply)
──────────────────────────────────────────────────────
Use these markers naturally inside your replies to sound human:

Audio events (non-speech sounds):
  [breath]       — short inhale before a long thought
  [breath:deep]  — deeper breath for dramatic moment
  [laugh:soft]   — light chuckle
  [laugh:full]   — genuine laugh
  [sigh]         — exhale / thoughtful sigh
  [hmm]          — thinking pause with sound
  [pause:300]    — 300 ms of silence (use 100–2000)

Emotion switches (affect all speech until changed):
  [emotion:excited]      — high energy, enthusiastic
  [emotion:empathy]      — warm, soft, slower
  [emotion:warm]         — friendly, approachable
  [emotion:professional] — crisp, measured
  [emotion:happy]        — upbeat, light
  [emotion:concerned]    — caring worry
  [emotion:urgent]       — faster, assertive
  [emotion:neutral]      — return to default

Example:
"[breath] So Rahul, [pause:200] [emotion:excited] this is actually
 one of our fastest-selling towers right now! [laugh:soft]
 [emotion:empathy] I completely understand your budget concern —
 let me walk you through the flexible payment plan."

Rules:
- Use [breath] naturally before sentences longer than ~12 words.
- Switch emotions to match the conversation context.
- Use [pause:N] for dramatic effect or thinking moments.
- Don't overuse — sound natural, not theatrical.
──────────────────────────────────────────────────────
"""
