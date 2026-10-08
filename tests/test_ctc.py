import numpy as np

from turbo_lyrics import ctc


def emissions(path, n_classes=6, frames_per=4, peak=0.97, blank_frames=3):
    """Synthetic log-probs: each token of ``path`` peaks for one frame, then blank for a while."""
    rows = []
    for tok in path:
        for k in range(frames_per):
            p = np.full(n_classes, (1 - peak) / (n_classes - 1))
            p[tok if k == 0 else 0] = peak
            rows.append(p)
        for _ in range(blank_frames):
            p = np.full(n_classes, (1 - peak) / (n_classes - 1))
            p[0] = peak
            rows.append(p)
    return np.log(np.array(rows, dtype=np.float32))


def test_align_finds_token_onsets():
    lp = emissions([1, 2, 3, 2])
    spans = ctc.align(lp, [1, 2, 3, 2], blank=0)
    assert [s.start for s in spans] == [0, 7, 14, 21]
    # each span holds until the next token starts; the last runs to the end
    assert [s.end for s in spans] == [7, 14, 21, len(lp)]
    assert all(s.score > 0.9 for s in spans)


def test_align_repeated_tokens_and_held_notes():
    # a sung, held vowel: the model keeps emitting the token itself instead of blank
    rows = [[0.01, 0.97, 0.01, 0.01]] * 6 + [[0.97, 0.01, 0.01, 0.01]] * 2 + [[0.01, 0.01, 0.97, 0.01]] * 3
    spans = ctc.align(np.log(np.array(rows, dtype=np.float32)), [1, 2], blank=0)
    assert (spans[0].start, spans[0].end) == (0, 8)
    assert spans[1].start == 8
    assert spans[0].score > 0.9


def test_star_absorbs_unknown_audio():
    lp = emissions([4, 5, 1, 2, 3])  # 4, 5 are an ad-lib that is not in the text
    spans = ctc.align(lp, [ctc.STAR, 1, 2, 3, ctc.STAR], blank=0)
    assert [s.start for s in spans[1:4]] == [14, 21, 28]
    assert spans[0].start == 0


def test_too_many_tokens_returns_none():
    lp = emissions([1])
    assert ctc.align(lp[:2], [1, 2, 3], blank=0) is None
    assert ctc.align(lp, [], blank=0) is None


def test_greedy_decode():
    lp = emissions([1, 1, 2])
    assert ctc.greedy_decode(lp, blank=0) == [1, 1, 2]
