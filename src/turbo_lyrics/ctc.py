"""CTC forced alignment in numpy.

The trellis follows the WhisperX/torchaudio-tutorial formulation that MelodAI's karaoke timing was tuned on: a
token is entered on the frame where the model emits it and then *held* until the next token starts. Holding a
token scores ``max(p(blank), p(token))`` per frame (WhisperX uses ``p(blank)`` only, which penalises the long,
repeated vowel emissions that sung notes produce). A token span therefore covers its emission plus the following
blank frames, so a word's end is where the next word's first token (or a delimiter) is emitted.

``STAR`` is a wildcard token (score: best class of the frame) used to absorb ad-libs, instrumental passages and
unsung text around reference lines. Every star consumes at least one frame (20 ms), which is negligible.

Memory is O(T*N) bits for the back-pointers (T frames, N tokens): a 5-minute song (15 000 frames) with 3 000
tokens needs 45 MB. Time is O(T*N) vectorised numpy, ~0.1-0.5 s for a whole song.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

STAR = -1
NEG = np.float32(-1e30)


@dataclass
class TokenSpan:
    token: int
    start: int  # first frame (inclusive)
    end: int  # last frame (exclusive)
    score: float  # mean probability of the path over the span


def align(logp: np.ndarray, tokens: list[int] | np.ndarray, blank: int = 0) -> list[TokenSpan] | None:
    """Viterbi-align ``tokens`` to log-probabilities ``logp`` (T, C). Returns one span per token, or None when
    the sequence does not fit (more tokens than frames)."""
    logp = np.asarray(logp, dtype=np.float32)
    tokens = np.asarray(tokens, dtype=np.int64)
    T, _C = logp.shape
    N = len(tokens)
    if N == 0 or N > T:
        return None

    star_row = logp.max(axis=1)
    is_star = tokens == STAR
    safe = np.where(is_star, 0, tokens)
    blank_lp = logp[:, blank]

    # dp[j] = best score with token j being the latest emitted token; "pre" = nothing emitted yet (leading blanks)
    dp = np.full(N, NEG, dtype=np.float32)
    advanced = np.zeros((T, N), dtype=bool)  # True: entered token j at frame t, False: held it
    pre = np.float32(blank_lp[0])
    emit0 = star_row[0] if is_star[0] else logp[0, safe[0]]
    dp[0] = emit0
    advanced[0, 0] = True
    for t in range(1, T):
        row = logp[t]
        emit = np.where(is_star, star_row[t], row[safe])
        hold = dp + np.maximum(emit, blank_lp[t])
        prev = np.empty_like(dp)
        prev[0] = pre
        prev[1:] = dp[:-1]
        enter = prev + emit
        take = enter > hold
        dp = np.where(take, enter, hold)
        advanced[t] = take
        pre = pre + blank_lp[t]

    if dp[-1] <= NEG / 2:
        return None

    # backtrack
    ends = np.empty(N, dtype=np.int64)
    starts = np.empty(N, dtype=np.int64)
    j = N - 1
    ends[j] = T
    t = T - 1
    while j >= 0:
        while t > 0 and not advanced[t, j]:
            t -= 1
        starts[j] = t
        if j > 0:
            ends[j - 1] = t
        j -= 1
        t -= 1
        if t < 0 and j >= 0:  # cannot happen for a valid path; guard against corrupt input
            return None

    spans = []
    for j in range(N):
        s, e = int(starts[j]), int(ends[j])
        seg = logp[s:e]
        if is_star[j]:
            path = seg.max(axis=1)
        else:
            path = np.maximum(seg[:, safe[j]], seg[:, blank])
            path[0] = seg[0, safe[j]]
        spans.append(TokenSpan(int(tokens[j]), s, e, float(np.exp(path).mean())))
    return spans


def greedy_decode(logp: np.ndarray, blank: int = 0) -> list[int]:
    """Best-path CTC decoding (collapse repeats, drop blanks)."""
    best = np.asarray(logp).argmax(axis=1)
    out = []
    prev = -2
    for b in best:
        if b != prev and b != blank:
            out.append(int(b))
        prev = b
    return out
