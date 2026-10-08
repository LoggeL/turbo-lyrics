from turbo_lyrics import compare, output


def test_fill_missing_interpolates_between_neighbours():
    words = [
        {"word": "a", "start": 1.0, "end": 1.5},
        {"word": "42", "start": None, "end": None},
        {"word": "%", "start": None, "end": None},
        {"word": "b", "start": 2.5, "end": 3.0},
    ]
    output.fill_missing(words, 0.0, 10.0)
    assert words[1]["start"] == 1.5 and words[2]["end"] == 2.5
    assert words[1]["interpolated"]


def test_make_monotonic_contract():
    words = [
        {"word": "a", "start": -0.2, "end": -0.1},
        {"word": "b", "start": 1.0, "end": 1.0},
        {"word": "c", "start": 0.9, "end": 5.0},
        {"word": "d", "start": 4.0, "end": 12.0},
    ]
    output.make_monotonic(words, duration=10.0)
    prev = 0.0
    for w in words:
        assert w["start"] >= prev and w["end"] > w["start"] and w["end"] <= 10.0
        prev = w["end"]


def test_trim_to_voice_cuts_silent_tails():
    import numpy as np

    voiced = np.zeros(500, dtype=bool)
    voiced[50:100] = True  # 1.0 s - 2.0 s
    words = [{"word": "la", "start": 1.0, "end": 6.0}]
    output.trim_to_voice(words, voiced)
    assert abs(words[0]["end"] - 2.08) < 1e-6


def test_sentences_and_segments():
    words = [{"word": w, "start": i, "end": i + 0.5, "score": 0.9} for i, w in enumerate(["Hey", "you.", "Go", "now"])]
    segs = [output.segment(s) for s in output.sentences(words)]
    assert [s["text"] for s in segs] == ["Hey you.", "Go now"]
    assert segs[1]["start"] == 2 and set(segs[0]["words"][0]) == {"word", "start", "end", "score"}


def test_compare():
    a = {"segments": [{"words": [{"word": "Hello", "start": 1.0}, {"word": "world", "start": 2.0}]}]}
    b = {"segments": [{"words": [{"word": "hello,", "start": 1.05}, {"word": "World", "start": 2.2}]}]}
    r = compare(a, b)
    assert r["matched"] == 2 and r["text_similarity"] == 1.0 and r["within_100ms"] == 0.5
