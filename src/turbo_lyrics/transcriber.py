"""Transcriber: sung vocals -> WhisperX-compatible word-timed lyrics, and reference-lyrics forced alignment.

Typical use (one vocal stem, with or without known lyrics)::

    tr = Transcriber()                       # load once
    song = tr.prepare("vocals.mp3")          # decode + VAD (milliseconds)
    ref = tr.align_reference(song, lines)    # ~emissions only; check ref.data["gate"]["passed"]
    asr = tr.transcribe(song)                # Whisper + CTC alignment, reuses the same emissions

Both paths share ``song``'s cached wav2vec2 emissions, so running both costs one alignment model pass.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field, replace

import numpy as np

from . import __version__, audio, ctc, output
from .align import Aligner, Emissions
from .asr import AsrOptions, AsrWindow, WhisperASR
from .language import detect_text_language
from .text import is_hallucination, split_words
from .vad import HOP, SAMPLE_RATE, Pack, Piece, VadOptions, find_regions, frame_db, pack, voiced_mask

log = logging.getLogger(__name__)

LOOP_COMPRESSION_RATIO = 2.6  # windows above this are re-decoded with beam search (repetition loops)


@dataclass
class GateOptions:
    """Acceptance thresholds for reference alignment (calibration notes in docs/how-it-works.md)."""

    min_mean_score: float = 0.6
    low_score: float = 0.3
    max_low_score_fraction: float = 0.15  # words scoring below ``low_score``
    min_voiced_fraction: float = 0.7  # share of aligned word time where the vocal stem is actually voiced
    max_unaligned_voiced_fraction: float = 0.4  # sung time that no reference line covers (missing repeats)
    max_lrc_deviation_s: float = 1.5  # median |aligned line start - LRC start| after the global offset
    min_lrc_within: float = 0.7  # share of lines within 2 s of their LRC time (after the offset)


@dataclass
class Prepared:
    """A decoded vocal stem with its VAD; caches alignment emissions per language."""

    wav: np.ndarray
    duration: float
    db: np.ndarray
    mask: np.ndarray
    regions: list[tuple[float, float]]
    packs: list[Pack]
    emissions: dict[str, Emissions] = field(default_factory=dict)
    timings: dict[str, float] = field(default_factory=dict)


@dataclass
class Result:
    data: dict  # WhisperX-compatible: {"segments": [...], "detected_language": ...} plus extras
    timings: dict = field(default_factory=dict)

    @property
    def words(self) -> list[dict]:
        return [w for s in self.data.get("segments", []) for w in s.get("words", [])]


class _Progress:
    def __init__(self, fn, total):
        self.fn, self.total, self.done = fn, max(1, total), 0

    def set(self, done):
        self.done = min(self.total, max(self.done, done))
        if self.fn is not None:
            self.fn(self.done, self.total)


class Transcriber:
    """Load once, reuse. One call at a time (the models keep per-call buffers)."""

    def __init__(
        self,
        model: str = "large-v3-turbo",
        compute_type: str = "int8",
        threads: int = 8,
        align_precision: str = "auto",
        align_models: dict[str, str] | None = None,
        cache_dir: str | None = None,
        local_files_only: bool = False,
        vad: VadOptions | None = None,
        asr: AsrOptions | None = None,
    ):
        self.model_name = model
        self.compute_type = compute_type
        self.threads = threads
        self.cache_dir = cache_dir
        self.local_files_only = local_files_only
        self.vad = vad or VadOptions()
        self.asr_options = asr or AsrOptions()
        self.aligner = Aligner(
            align_models,
            threads=threads,
            precision=align_precision,
            cache_dir=cache_dir,
            local_files_only=local_files_only,
        )
        self._asr: WhisperASR | None = None

    # ------------------------------------------------------------------ models
    @property
    def asr(self) -> WhisperASR:
        if self._asr is None:
            self._asr = WhisperASR(
                self.model_name,
                self.compute_type,
                self.threads,
                download_root=self.cache_dir,
                local_files_only=self.local_files_only,
            )
        return self._asr

    def warm(self, languages=("en", "de")) -> dict:
        """Load the Whisper model and the aligners for ``languages`` now (worker start)."""
        t0 = time.perf_counter()
        _ = self.asr
        for lang in languages:
            if self.aligner.supports(lang):
                self.aligner.model(lang)
        return {"load_s": round(time.perf_counter() - t0, 2)}

    def info(self) -> dict:
        return {
            "engine": "turbo-lyrics",
            "version": __version__,
            "model": self.model_name,
            "compute_type": self.compute_type,
            "threads": self.threads,
            "align_precision": self.aligner.precision,
        }

    # ------------------------------------------------------------------ shared preparation
    def prepare(self, source, sr: int | None = None) -> Prepared:
        if isinstance(source, Prepared):
            return source
        t0 = time.perf_counter()
        wav = audio.load(source, sr)
        db = frame_db(wav)
        mask = voiced_mask(db, self.vad)
        regions = find_regions(wav, self.vad)
        packs = pack(wav, self.vad, regions=regions)
        return Prepared(
            wav,
            len(wav) / SAMPLE_RATE,
            db,
            mask,
            regions,
            packs,
            timings={"prepare_s": round(time.perf_counter() - t0, 3)},
        )

    def _emissions(self, song: Prepared, language: str, progress=None) -> tuple[Emissions, float]:
        if language not in song.emissions:
            t = time.perf_counter()
            model = self.aligner.model(language)
            song.emissions[language] = model.song_emissions(song.wav, song.regions, progress=progress)
            return song.emissions[language], time.perf_counter() - t
        return song.emissions[language], 0.0

    # ------------------------------------------------------------------ ASR path
    def transcribe(
        self,
        source,
        *,
        sr: int | None = None,
        language: str | None = None,
        reference: list[str] | str | None = None,
        progress=None,
        use_vad: bool = True,
        asr: AsrOptions | None = None,
    ) -> Result:
        """Transcribe and word-align a vocal stem (a path, an array or a :class:`Prepared`).

        ``language`` forces the language; otherwise it comes from ``reference`` text (lyrics of the same song), and
        only then from Whisper's detector run on the fullest voiced windows. ``progress(done, total)`` may raise to
        abort the call.
        """
        opts = asr or self.asr_options
        timings: dict[str, float] = {}
        t0 = time.perf_counter()
        song = self.prepare(source, sr)
        packs = song.packs if use_vad else _fixed_windows(song.duration)
        steps = _Progress(progress, 2 + 2 * len(packs))
        steps.set(1)
        if not packs:
            return Result(
                {
                    "segments": [],
                    "detected_language": language,
                    "engine": self.info(),
                    "vad": {"voiced_s": 0.0, "windows": 0, "duration_s": round(song.duration, 3)},
                },
                timings,
            )

        t = time.perf_counter()
        language_source, language_probability = "given", 1.0
        if not language and reference:
            language, language_source = detect_text_language(reference), "reference"
        if not language:
            language, language_probability = self._detect_language(packs, song.wav)
            language_source = "whisper"
        timings["language_s"] = time.perf_counter() - t

        can_align = self.aligner.supports(language)
        opts = replace(opts, word_timestamps=not can_align)
        t = time.perf_counter()
        windows = self.asr.transcribe(packs, song.wav, language, opts, progress=lambda done, _n: steps.set(1 + done))
        windows = self._retry_loops(packs, song.wav, language, opts, windows)
        timings["asr_s"] = time.perf_counter() - t

        model, em = None, None
        if can_align:
            model = self.aligner.model(language)
            em, timings["emissions_s"] = self._emissions(song, language)
        steps.set(2 + len(packs))
        t = time.perf_counter()
        all_words: list[list[dict]] = []
        for k, (p, win) in enumerate(zip(packs, windows)):
            words = self._window_words(p, win, model, em, opts)
            if words:
                output.fill_missing(words, p.start, p.end)
            all_words.append(words)
            steps.set(2 + len(packs) + k + 1)
        timings["align_s"] = time.perf_counter() - t

        flat = [w for ws in all_words for w in ws]
        output.trim_to_voice(flat, song.mask)
        output.make_monotonic(flat, song.duration)
        segments = [output.segment(s) for ws in all_words if ws for s in output.sentences(ws)]
        timings["total_s"] = time.perf_counter() - t0
        timings = {**song.timings, **{k: round(v, 3) for k, v in timings.items()}}
        voiced_s = sum(pc.end - pc.start for p in packs for pc in p.pieces)
        data = {
            "segments": segments,
            "detected_language": language,
            "language_probability": round(language_probability, 4),
            "language_source": language_source,
            "alignment": "ctc" if can_align else "whisper",
            "engine": {**self.info(), "align_model": model.name if model else None},
            "vad": {"voiced_s": round(voiced_s, 2), "windows": len(packs), "duration_s": round(song.duration, 3)},
            "timings": timings,
        }
        return Result(data, timings)

    def _detect_language(self, packs: list[Pack], wav, confident: float = 0.7, max_windows: int = 3):
        """Whisper's detector on the fullest windows: shouts and spoken intros fool a single 30 s look, so an
        unsure answer is cross-checked on up to ``max_windows`` windows (probabilities averaged)."""
        ranked = sorted(packs, key=lambda p: sum(pc.end - pc.start for pc in p.pieces), reverse=True)
        totals: dict[str, float] = {}
        looked = 0
        for p in ranked[:max_windows]:
            _lang, _prob, probs = self.asr.model.detect_language(audio=p.audio(wav)[: 30 * SAMPLE_RATE])
            looked += 1
            for lang, prob in probs:
                totals[lang] = totals.get(lang, 0.0) + float(prob)
            if max(totals.values()) / looked >= confident:
                break
        best = max(totals, key=totals.get)
        return best, totals[best] / looked

    def _retry_loops(self, packs, wav, language, opts, windows):
        """Re-decode windows that look like repetition loops with beam search; keep the less repetitive text."""
        bad = [i for i, w in enumerate(windows) if w.text and w.compression_ratio > LOOP_COMPRESSION_RATIO]
        if not bad or opts.beam_size > 1:
            return windows
        retry = self.asr.transcribe([packs[i] for i in bad], wav, language, replace(opts, beam_size=5, temperature=0.4))
        for i, w in zip(bad, retry):
            if w.text and w.compression_ratio < windows[i].compression_ratio:
                windows[i] = w
        return windows

    def _window_words(self, p: Pack, win: AsrWindow, model, em: Emissions | None, opts: AsrOptions) -> list[dict]:
        if not win.text or is_hallucination(win.text):
            return []
        if win.no_speech_prob > opts.no_speech_threshold and win.avg_logprob < opts.log_prob_threshold:
            return []
        if model is None:  # no aligner for this language: Whisper's cross-attention word times
            return [
                {"word": w, "start": p.to_original(s), "end": p.to_original(e), "score": round(float(pr), 3)}
                for w, s, e, pr in (win.words or [])
                if w
            ]
        words = split_words(win.text)
        frames = np.concatenate(
            [np.arange(max(0, em.frame(pc.start)), min(len(em.logp), em.frame(pc.end))) for pc in p.pieces]
        )
        timings = model.align_words(em, words, frames=frames)
        return [
            {"word": w, "start": None, "end": None}
            if tm is None
            else {"word": w, "start": tm.start, "end": tm.end, "score": round(tm.score, 3)}
            for w, tm in zip(words, timings)
        ]

    # ------------------------------------------------------------------ reference path
    def align_reference(
        self,
        source,
        lines: list[str],
        *,
        sr: int | None = None,
        line_times: list[float | None] | None = None,
        language: str | None = None,
        progress=None,
        gate: GateOptions | None = None,
    ) -> Result:
        """Force-align known lyric ``lines`` to a vocal stem without running ASR.

        Lines are aligned in order with a wildcard between them that absorbs ad-libs, instrumental parts and
        sung passages missing from the text. ``data["gate"]`` says whether the alignment is trustworthy; callers
        should fall back to :meth:`transcribe` when ``gate["passed"]`` is false. ``line_times`` (LRC timestamps)
        only feed the gate, as an independent check.
        """
        gate = gate or GateOptions()
        timings: dict[str, float] = {}
        t0 = time.perf_counter()
        language = language or detect_text_language(lines)
        if not self.aligner.supports(language):
            raise ValueError(f"no alignment model for language {language!r}")
        song = self.prepare(source, sr)
        model = self.aligner.model(language)
        steps = _Progress(progress, 3)
        steps.set(1)
        em, timings["emissions_s"] = self._emissions(song, language)
        steps.set(2)

        t = time.perf_counter()
        line_words = [split_words(line) for line in lines]
        tokens: list[int] = [ctc.STAR]
        ranges: list[list[tuple[int, int] | None]] = []
        for words in line_words:
            toks, rng = model.tokens_for(words)
            base = len(tokens)
            ranges.append([None if r is None else (r[0] + base, r[1] + base) for r in rng])
            tokens.extend(toks)
            tokens.append(ctc.STAR)
        spans = ctc.align(em.logp, tokens, model.blank)
        timings["viterbi_s"] = time.perf_counter() - t
        if spans is None:
            raise ValueError("reference text is too long for the audio")

        lines_out: list[list[dict]] = []
        for words, rng in zip(line_words, ranges):
            ws = []
            for w, r in zip(words, rng):
                if r is None:
                    ws.append({"word": w, "start": None, "end": None})
                    continue
                sp = spans[r[0] : r[1]]
                ws.append(
                    {
                        "word": w,
                        "start": em.time(sp[0].start),
                        "end": em.time(sp[-1].end),
                        "score": round(float(np.mean([x.score for x in sp])), 3),
                    }
                )
            lines_out.append(ws)
        flat = [w for ws in lines_out for w in ws]
        if flat:
            output.fill_missing(flat, 0.0, song.duration)
        output.trim_to_voice(flat, song.mask)
        output.make_monotonic(flat, song.duration)
        steps.set(3)

        report = self._gate(lines_out, song.mask, line_times, gate)
        segments = [output.segment(ws) for ws in lines_out if ws]
        line_starts, k = [], 0
        for ws in lines_out:
            if ws:
                line_starts.append(k)
            k += len(ws)
        timings["total_s"] = time.perf_counter() - t0
        timings = {**song.timings, **{k2: round(v, 3) for k2, v in timings.items()}}
        data = {
            "segments": segments,
            "detected_language": language,
            "source": "reference",
            "line_starts": line_starts,
            "alignment": "ctc-reference",
            "engine": {**self.info(), "align_model": model.name},
            "gate": report,
            "timings": timings,
        }
        return Result(data, timings)

    @staticmethod
    def _gate(lines_out, mask, line_times, gate: GateOptions) -> dict:
        fs = HOP / SAMPLE_RATE
        words = [w for ws in lines_out for w in ws if not w.get("interpolated")]
        scores = np.array([w.get("score", 0.0) for w in words]) if words else np.zeros(1)
        voiced_time = word_time = 0.0
        covered = np.zeros(len(mask), dtype=bool)
        for w in words:
            a, b = int(w["start"] / fs), int(np.ceil(w["end"] / fs))
            seg = mask[a:b]
            voiced_time += float(seg.sum())
            word_time += float(seg.size)
        for ws in lines_out:  # a line covers its breath gaps too
            if ws:
                covered[int(ws[0]["start"] / fs) : int(np.ceil(ws[-1]["end"] / fs))] = True
        total_voiced = float(mask.sum()) or 1.0
        metrics = {
            "words": len(words),
            "mean_score": round(float(scores.mean()), 4),
            "low_score_fraction": round(float((scores < gate.low_score).mean()), 4),
            "voiced_fraction": round(voiced_time / word_time, 4) if word_time else 0.0,
            "unaligned_voiced_fraction": round(float((mask & ~covered).sum()) / total_voiced, 4),
        }
        checks = [
            ("mean_score", metrics["mean_score"] < gate.min_mean_score),
            ("low_score_fraction", metrics["low_score_fraction"] > gate.max_low_score_fraction),
            ("voiced_fraction", metrics["voiced_fraction"] < gate.min_voiced_fraction),
            ("unaligned_voiced_fraction", metrics["unaligned_voiced_fraction"] > gate.max_unaligned_voiced_fraction),
            ("no_words", not words),
        ]
        if line_times:
            pairs = [(ws[0]["start"], lt) for ws, lt in zip(lines_out, line_times) if ws and lt is not None]
            if len(pairs) >= 3:
                diffs = np.array([a - b for a, b in pairs])
                offset = float(np.median(diffs))
                dev = np.abs(diffs - offset)
                metrics["lrc_offset_s"] = round(offset, 3)
                metrics["lrc_deviation_median_s"] = round(float(np.median(dev)), 3)
                metrics["lrc_within_2s"] = round(float((dev <= 2.0).mean()), 4)
                checks.append(("lrc_deviation", metrics["lrc_deviation_median_s"] > gate.max_lrc_deviation_s))
                checks.append(("lrc_within", metrics["lrc_within_2s"] < gate.min_lrc_within))
        reasons = [name for name, failed in checks if failed]
        return {"passed": not reasons, "reasons": reasons, "metrics": metrics, "thresholds": dict(gate.__dict__)}


def _fixed_windows(duration: float, length: float = 29.5) -> list[Pack]:
    packs = []
    s = 0.0
    while s < duration:
        e = min(duration, s + length)
        packs.append(Pack([Piece(s, e, 0.0)]))
        s = e
    return packs
