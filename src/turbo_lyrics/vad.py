"""Energy VAD for separated vocal stems and packing of voiced regions into ≤30 s Whisper windows.

A BS-RoFormer vocal stem is nearly silent outside sung passages, so a frame-RMS gate relative to the song's own
loud level finds the voiced regions in milliseconds (a neural VAD trained on speech is slower and less reliable
on singing). Whisper pads every input to a 30 s window, so encoder cost scales with the number of windows, not
with the voiced seconds: the regions are therefore *packed* — concatenated with short silences between them —
into as few windows as possible. ``Pack.to_original`` maps a time inside a packed window back to the song.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

SAMPLE_RATE = 16000
HOP = 320  # 20 ms frames, the same rate as wav2vec2 emissions


@dataclass
class VadOptions:
    threshold_db: float = 32.0  # voiced = within this many dB of the song's loud level (95th percentile)
    floor_dbfs: float = -55.0  # never voiced below this absolute level (stem bleed / noise)
    min_silence_s: float = 0.5  # shorter dips stay inside a region
    min_region_s: float = 0.12  # drop clicks / breaths shorter than this
    pad_s: float = 0.15  # context kept around every region
    max_window_s: float = 29.5  # Whisper window (30 s) minus a margin
    gap_s: float = 0.4  # silence inserted between packed regions (Whisper hears a phrase break)


@dataclass
class Piece:
    start: float  # original time (s)
    end: float
    offset: float  # position inside the packed window (s)


@dataclass
class Pack:
    """One Whisper window: several voiced regions of the song back to back."""

    pieces: list[Piece] = field(default_factory=list)

    @property
    def duration(self) -> float:
        return self.pieces[-1].offset + (self.pieces[-1].end - self.pieces[-1].start) if self.pieces else 0.0

    @property
    def start(self) -> float:
        return self.pieces[0].start

    @property
    def end(self) -> float:
        return self.pieces[-1].end

    def audio(self, wav: np.ndarray, sr: int = SAMPLE_RATE) -> np.ndarray:
        out = np.zeros(round(self.duration * sr) + 1, dtype=np.float32)
        for p in self.pieces:
            a, b = round(p.start * sr), round(p.end * sr)
            o = round(p.offset * sr)
            seg = wav[a:b]
            out[o : o + len(seg)] = seg[: len(out) - o]
        return out

    def to_original(self, t: float) -> float:
        """Packed-window time -> song time (times in an inserted gap snap to the nearest piece edge)."""
        best = None
        for p in self.pieces:
            length = p.end - p.start
            if p.offset <= t <= p.offset + length:
                return p.start + (t - p.offset)
            edge = p.start if t < p.offset else p.end
            dist = p.offset - t if t < p.offset else t - (p.offset + length)
            if best is None or dist < best[0]:
                best = (dist, edge)
        return best[1] if best else t


def frame_db(wav: np.ndarray, hop: int = HOP) -> np.ndarray:
    n = len(wav) // hop
    if n == 0:
        return np.full(1, -120.0, dtype=np.float32)
    frames = wav[: n * hop].reshape(n, hop).astype(np.float32)
    rms = np.sqrt(np.mean(frames * frames, axis=1) + 1e-12)
    return (20.0 * np.log10(rms + 1e-10)).astype(np.float32)


def voiced_mask(db: np.ndarray, opts: VadOptions) -> np.ndarray:
    """Per-frame voiced flags for frame levels ``db``."""
    loud = float(np.percentile(db, 95)) if db.size else -120.0
    threshold = max(loud - opts.threshold_db, opts.floor_dbfs)
    return db > threshold


def find_regions(wav: np.ndarray, opts: VadOptions | None = None, sr: int = SAMPLE_RATE) -> list[tuple[float, float]]:
    """Voiced regions (start, end) in seconds."""
    opts = opts or VadOptions()
    db = frame_db(wav)
    mask = voiced_mask(db, opts)
    fs = HOP / sr
    regions: list[list[float]] = []
    idx = np.flatnonzero(np.diff(np.concatenate([[0], mask.astype(np.int8), [0]])))
    for a, b in zip(idx[0::2], idx[1::2]):
        s, e = a * fs, b * fs
        if regions and s - regions[-1][1] < opts.min_silence_s:
            regions[-1][1] = e
        else:
            regions.append([s, e])
    duration = len(wav) / sr
    out: list[tuple[float, float]] = []
    for s, e in regions:
        if e - s < opts.min_region_s:
            continue
        s, e = max(0.0, s - opts.pad_s), min(duration, e + opts.pad_s)
        if out and s <= out[-1][1]:
            out[-1] = (out[-1][0], e)
        else:
            out.append((s, e))
    return out


MIN_FILL_S = 6.0  # a long region is cut to fill the rest of a window only when at least this much room is left


def _quiet_cut(db: np.ndarray, s: float, room: float, sr: int) -> float:
    """Time of the quietest frame in [s + 0.8 room, s + room] (a breath close to the end of the window)."""
    fs = HOP / sr
    lo = int((s + room * 0.8) / fs)
    hi = max(lo + 1, int((s + room) / fs))
    window = db[lo:hi]
    # the latest frame that is about as quiet as the quietest one (fills the window as far as possible)
    cut = (lo + int(np.flatnonzero(window <= window.min() + 1.5)[-1])) * fs if window.size else s + room
    return min(max(cut, s + 1.0), s + room)


def pack(
    wav: np.ndarray,
    opts: VadOptions | None = None,
    sr: int = SAMPLE_RATE,
    regions: list[tuple[float, float]] | None = None,
) -> list[Pack]:
    """Voiced regions packed greedily (in order) into windows of at most ``opts.max_window_s``.

    Regions that fit are appended whole. A region longer than a window has to be cut anyway, so it is cut at a
    breath to fill the current window first; that saves a window on songs that are sung almost throughout.
    """
    opts = opts or VadOptions()
    if regions is None:
        regions = find_regions(wav, opts, sr)
    db = frame_db(wav)
    packs: list[Pack] = []
    current = Pack()

    def flush():
        nonlocal current
        if current.pieces:
            packs.append(current)
        current = Pack()

    for s, e in regions:
        while e - s > 1e-6:
            offset = current.duration + opts.gap_s if current.pieces else 0.0
            room = opts.max_window_s - offset
            if e - s <= room:
                current.pieces.append(Piece(s, e, offset))
                break
            if (e - s > opts.max_window_s and room >= MIN_FILL_S) or not current.pieces:
                cut = _quiet_cut(db, s, room, sr)
                current.pieces.append(Piece(s, cut, offset))
                s = cut
            flush()
    flush()
    return packs


def voiced_fraction(
    db: np.ndarray,
    start: float,
    end: float,
    opts: VadOptions | None = None,
    sr: int = SAMPLE_RATE,
    _mask: np.ndarray | None = None,
) -> float:
    """Share of voiced frames in [start, end) (1.0 for an empty interval)."""
    opts = opts or VadOptions()
    mask = voiced_mask(db, opts) if _mask is None else _mask
    fs = HOP / sr
    a, b = int(start / fs), max(int(start / fs) + 1, int(np.ceil(end / fs)))
    seg = mask[a:b]
    return float(seg.mean()) if seg.size else 1.0
