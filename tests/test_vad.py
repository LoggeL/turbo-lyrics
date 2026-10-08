import numpy as np

from turbo_lyrics.vad import SAMPLE_RATE, VadOptions, find_regions, pack


def tone(seconds, amp=0.3, f=220.0):
    t = np.arange(int(seconds * SAMPLE_RATE)) / SAMPLE_RATE
    return (amp * np.sin(2 * np.pi * f * t)).astype(np.float32)


def silence(seconds):
    return np.zeros(int(seconds * SAMPLE_RATE), dtype=np.float32)


def test_regions_follow_the_voice():
    wav = np.concatenate([silence(2), tone(3), silence(4), tone(2), silence(1)])
    regions = find_regions(wav)
    assert len(regions) == 2
    (a, b), (c, d) = regions
    assert abs(a - 2) < 0.2 and abs(b - 5) < 0.2 and abs(c - 9) < 0.2 and abs(d - 11) < 0.2


def test_short_dips_stay_inside_a_region():
    wav = np.concatenate([tone(2), silence(0.3), tone(2)])
    assert len(find_regions(wav)) == 1


def test_packing_fills_windows_and_maps_back():
    parts = []
    for _ in range(10):
        parts += [tone(6), silence(3)]
    wav = np.concatenate(parts)
    packs = pack(wav, VadOptions())
    assert len(packs) == 3  # 10 x ~6.3 s of voice -> 3 windows instead of 4 contiguous 30 s ones
    assert all(p.duration <= 29.5 + 1e-6 for p in packs)
    p = packs[1]
    second = p.pieces[1]
    t = second.offset + 1.0
    assert abs(p.to_original(t) - (second.start + 1.0)) < 1e-6
    audio = p.audio(wav)
    assert abs(len(audio) / SAMPLE_RATE - p.duration) < 0.01


def test_long_region_is_cut_to_fill_the_window():
    wav = np.concatenate([tone(10), silence(2), tone(70)])
    packs = pack(wav)
    assert len(packs) == 3
    assert packs[0].duration > 25  # the 70 s region tops up the first window
    covered = sum(pc.end - pc.start for p in packs for pc in p.pieces)
    assert covered > 79
