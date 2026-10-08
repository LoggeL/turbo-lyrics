"""End-to-end on real models. Needs --models and TURBO_LYRICS_TEST_AUDIO (a sung vocal stem, any language)."""

import os

import pytest

pytestmark = pytest.mark.models


@pytest.fixture(scope="module")
def setup():
    path = os.environ.get("TURBO_LYRICS_TEST_AUDIO")
    if not path:
        pytest.skip("set TURBO_LYRICS_TEST_AUDIO to a vocal stem")
    from turbo_lyrics import Transcriber

    tr = Transcriber()
    return tr, tr.prepare(path)


def test_transcribe_contract_and_reference_roundtrip(setup):
    from turbo_lyrics import compare

    tr, song = setup
    asr = tr.transcribe(song)
    words = asr.words
    assert words, "no words transcribed"
    prev = 0.0
    for w in words:
        assert w["start"] >= prev and w["end"] > w["start"] and w["end"] <= song.duration
        prev = w["end"]
    lines = [s["text"] for s in asr.data["segments"]]
    ref = tr.align_reference(song, lines, language=asr.data["detected_language"])
    assert ref.data["gate"]["metrics"]["mean_score"] > 0.6
    assert compare(ref.data, asr.data)["start_delta_median_s"] < 0.1
