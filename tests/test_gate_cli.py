import numpy as np

from turbo_lyrics.cli import _read_reference
from turbo_lyrics.transcriber import GateOptions, Transcriber


def line(words, t0, score=0.95):
    return [{"word": w, "start": t0 + i * 0.5, "end": t0 + i * 0.5 + 0.4, "score": score} for i, w in enumerate(words)]


def test_gate_accepts_good_and_rejects_bad_alignments():
    mask = np.zeros(1000, dtype=bool)  # 20 s
    mask[50:250] = True  # 1 s - 5 s sung
    good = [line(["a", "b", "c", "d"], 1.0), line(["e", "f", "g", "h"], 3.0)]
    report = Transcriber._gate(good, mask, [1.0, 3.0], GateOptions())
    assert report["passed"], report
    bad = [line(["a", "b", "c", "d"], 10.0, score=0.1)]  # in silence and unsure; most singing uncovered
    report = Transcriber._gate(bad, mask, None, GateOptions())
    assert not report["passed"]
    assert {"mean_score", "voiced_fraction", "unaligned_voiced_fraction"} <= set(report["reasons"])


def test_gate_checks_lrc_drift():
    mask = np.ones(3000, dtype=bool)
    lines = [line(["x", "y"], t) for t in (0.0, 10.0, 20.0, 30.0, 40.0, 50.0)]
    lrc = [0.2, 10.2, 35.0, 30.2, 2.0, 50.2]  # two lines far off after the offset
    report = Transcriber._gate(lines, mask, lrc, GateOptions(max_unaligned_voiced_fraction=1.0))
    assert report["metrics"]["lrc_offset_s"] == -0.2
    assert report["metrics"]["lrc_within_2s"] < 0.7 and "lrc_within" in report["reasons"]


def test_read_reference_lrc_and_plain(tmp_path):
    lrc = tmp_path / "a.lrc"
    lrc.write_text("[ar:Someone]\n[00:12.50] First line\n[01:02.00]Second\n[01:05.00]\n", encoding="utf-8")
    assert _read_reference(str(lrc)) == (["First line", "Second"], [12.5, 62.0])
    txt = tmp_path / "a.txt"
    txt.write_text("One\n\n[Chorus]\nTwo\n", encoding="utf-8")
    assert _read_reference(str(txt)) == (["One", "Two"], None)
