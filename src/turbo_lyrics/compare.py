"""Compare two word-timed transcripts (e.g. reference alignment vs ASR alignment, or local vs WhisperX)."""

from __future__ import annotations

from difflib import SequenceMatcher

import numpy as np

from .text import fold


def words_of(result: dict | list) -> list[dict]:
    segments = result if isinstance(result, list) else result.get("segments", [])
    return [w for s in segments for w in s.get("words", []) if fold(w.get("word", ""))]


def compare(a: dict | list, b: dict | list) -> dict:
    """Word-level agreement: text similarity and start-time differences of the words both transcripts share."""
    wa, wb = words_of(a), words_of(b)
    ta, tb = [fold(w["word"]) for w in wa], [fold(w["word"]) for w in wb]
    sm = SequenceMatcher(None, ta, tb, autojunk=False)
    deltas = []
    for block in sm.get_matching_blocks():
        for k in range(block.size):
            x, y = wa[block.a + k], wb[block.b + k]
            if x.get("start") is not None and y.get("start") is not None:
                deltas.append(abs(float(x["start"]) - float(y["start"])))
    d = np.array(deltas) if deltas else np.array([np.nan])
    return {
        "words_a": len(wa),
        "words_b": len(wb),
        "text_similarity": round(sm.ratio(), 4),
        "matched": len(deltas),
        "start_delta_median_s": None if not deltas else round(float(np.median(d)), 3),
        "start_delta_p90_s": None if not deltas else round(float(np.percentile(d, 90)), 3),
        "within_100ms": None if not deltas else round(float(np.mean(d <= 0.1)), 4),
        "within_300ms": None if not deltas else round(float(np.mean(d <= 0.3)), 4),
    }
