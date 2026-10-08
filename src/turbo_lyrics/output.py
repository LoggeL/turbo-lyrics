"""Word timing post-processing and the WhisperX-compatible output format."""

from __future__ import annotations

import numpy as np

from .text import ends_sentence
from .vad import HOP, SAMPLE_RATE

MIN_WORD_S = 0.05
TRIM_TAIL_S = 0.08  # kept after the last voiced frame of a word


def fill_missing(words: list[dict], lo: float, hi: float) -> None:
    """Give words without timing (digits, symbols, failed alignment) times between their aligned neighbours."""
    n = len(words)
    i = 0
    while i < n:
        if words[i].get("start") is not None:
            i += 1
            continue
        j = i
        while j < n and words[j].get("start") is None:
            j += 1
        left = words[i - 1]["end"] if i > 0 else lo
        right = words[j]["start"] if j < n else hi
        if right - left < MIN_WORD_S * (j - i):
            right = left + MIN_WORD_S * (j - i)
        step = (right - left) / (j - i)
        for k in range(i, j):
            words[k]["start"] = left + step * (k - i)
            words[k]["end"] = left + step * (k - i + 1)
            words[k]["interpolated"] = True
        i = j


def trim_to_voice(words: list[dict], voiced: np.ndarray | None) -> None:
    """Cut each word's end back to its last voiced frame (+80 ms): a CTC span also holds the silence/instrumental
    gap until the next token, which would hide the pauses the karaoke line splitter relies on."""
    if voiced is None or not len(voiced):
        return
    fs = HOP / SAMPLE_RATE
    for w in words:
        if w.get("interpolated"):
            continue
        a, b = int(w["start"] / fs), int(np.ceil(w["end"] / fs))
        seg = voiced[a:b]
        if not seg.size or seg[-1]:
            continue
        hits = np.flatnonzero(seg)
        if hits.size:
            last = (a + hits[-1] + 1) * fs + TRIM_TAIL_S
            w["end"] = max(w["start"] + MIN_WORD_S, min(w["end"], last))


def make_monotonic(words: list[dict], duration: float) -> None:
    """start >= 0, end > start (at least 50 ms), no overlap with the next word, everything within the song."""
    prev_end = 0.0
    for i, w in enumerate(words):
        s = max(float(w["start"]), prev_end, 0.0)
        e = max(float(w["end"]), s + MIN_WORD_S)
        nxt = words[i + 1]["start"] if i + 1 < len(words) and words[i + 1].get("start") is not None else None
        if nxt is not None and e > nxt >= s + MIN_WORD_S:
            e = nxt
        if duration:
            e = min(e, duration)
            s = min(s, max(0.0, e - MIN_WORD_S))
        w["start"], w["end"] = round(s, 3), round(e, 3)
        prev_end = w["end"]


def sentences(words: list[dict]) -> list[list[dict]]:
    """Split a word list after sentence-ending punctuation (WhisperX segments are sentences)."""
    out: list[list[dict]] = []
    cur: list[dict] = []
    for w in words:
        cur.append(w)
        if ends_sentence(w["word"]):
            out.append(cur)
            cur = []
    if cur:
        out.append(cur)
    return out


def segment(words: list[dict]) -> dict:
    return {
        "start": words[0]["start"],
        "end": words[-1]["end"],
        "text": " ".join(w["word"] for w in words),
        "words": [{k: v for k, v in w.items() if k in ("word", "start", "end", "score")} for w in words],
    }
