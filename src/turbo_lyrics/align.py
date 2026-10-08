"""wav2vec2 CTC emissions and word alignment.

Only permissively licensed (Apache-2.0) checkpoints are in the default registry; WhisperX's defaults for
de/fr/es/it (VoxPopuli) are CC-BY-NC and deliberately not used. Any other CTC checkpoint can be passed per
language via ``Aligner(models={"xx": "org/model"})``.
"""

from __future__ import annotations

import logging
import threading
import time
from dataclasses import dataclass

import numpy as np

from . import ctc
from .text import CharMapper

log = logging.getLogger(__name__)

SAMPLE_RATE = 16000
WINDOW_S = 30.0  # emissions are computed on windows of this length ...
CONTEXT_S = 1.0  # ... with this much context on each side (dropped after the forward pass)

ALIGN_MODELS = {
    "en": "facebook/wav2vec2-base-960h",
    "de": "jonatasgrosman/wav2vec2-large-xlsr-53-german",
    "fr": "jonatasgrosman/wav2vec2-large-xlsr-53-french",
    "es": "jonatasgrosman/wav2vec2-large-xlsr-53-spanish",
    "it": "jonatasgrosman/wav2vec2-large-xlsr-53-italian",
    "pt": "jonatasgrosman/wav2vec2-large-xlsr-53-portuguese",
    "nl": "jonatasgrosman/wav2vec2-large-xlsr-53-dutch",
    "pl": "jonatasgrosman/wav2vec2-large-xlsr-53-polish",
    "ru": "jonatasgrosman/wav2vec2-large-xlsr-53-russian",
    "fi": "jonatasgrosman/wav2vec2-large-xlsr-53-finnish",
    "hu": "jonatasgrosman/wav2vec2-large-xlsr-53-hungarian",
    "el": "jonatasgrosman/wav2vec2-large-xlsr-53-greek",
    "fa": "jonatasgrosman/wav2vec2-large-xlsr-53-persian",
    "ar": "jonatasgrosman/wav2vec2-large-xlsr-53-arabic",
    "ja": "jonatasgrosman/wav2vec2-large-xlsr-53-japanese",
    "zh": "jonatasgrosman/wav2vec2-large-xlsr-53-chinese-zh-cn",
}


def cpu_has_bf16() -> bool:
    """True on CPUs with native bf16 dot products (AVX512_BF16 / AMX: Zen 4+, Sapphire Rapids+)."""
    try:
        with open("/proc/cpuinfo") as handle:
            flags = next((line for line in handle if line.startswith("flags")), "")
    except OSError:
        return False
    return " avx512_bf16" in flags or " amx_bf16" in flags


def resolve_precision(precision: str) -> str:
    if precision == "auto":
        return "bf16" if cpu_has_bf16() else "fp32"
    if precision not in ("fp32", "bf16"):
        raise ValueError(f"precision must be auto, fp32 or bf16, not {precision!r}")
    return precision


@dataclass
class Emissions:
    logp: np.ndarray  # (T, C) log-probabilities
    frame_s: float  # seconds per frame
    offset: float = 0.0  # time of frame 0

    def time(self, frame: float) -> float:
        return self.offset + frame * self.frame_s

    def frame(self, t: float) -> int:
        return round((t - self.offset) / self.frame_s)

    def slice(self, start: float, end: float) -> Emissions:
        a = max(0, self.frame(start))
        b = min(len(self.logp), max(a + 1, self.frame(end)))
        return Emissions(self.logp[a:b], self.frame_s, self.time(a))


@dataclass
class WordTiming:
    start: float
    end: float
    score: float


class AlignModel:
    """One wav2vec2 CTC checkpoint (loaded once, reused)."""

    def __init__(
        self,
        name: str,
        threads: int | None = None,
        precision: str = "auto",
        cache_dir: str | None = None,
        local_files_only: bool = False,
    ):
        import torch
        from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor

        t0 = time.perf_counter()
        if threads:
            torch.set_num_threads(int(threads))
        self.name = name
        self.precision = resolve_precision(precision)
        self.processor = Wav2Vec2Processor.from_pretrained(name, cache_dir=cache_dir, local_files_only=local_files_only)
        self.model = Wav2Vec2ForCTC.from_pretrained(name, cache_dir=cache_dir, local_files_only=local_files_only)
        self.model.eval()
        vocab = self.processor.tokenizer.get_vocab()
        self.blank = int(self.processor.tokenizer.pad_token_id or 0)
        self.mapper = CharMapper(vocab, self.blank)
        self.uses_mask = bool(getattr(self.processor.feature_extractor, "return_attention_mask", False))
        self.load_s = time.perf_counter() - t0
        self._lock = threading.Lock()

    def _forward(self, audio: np.ndarray) -> np.ndarray:
        import torch

        inputs = self.processor(audio, sampling_rate=SAMPLE_RATE, return_tensors="pt")
        with torch.inference_mode():
            if self.precision == "bf16":
                with torch.autocast("cpu", dtype=torch.bfloat16):
                    logits = self.model(inputs.input_values).logits
            else:
                logits = self.model(inputs.input_values).logits
            return torch.log_softmax(logits.float(), dim=-1)[0].numpy()

    def emissions(self, audio: np.ndarray, offset: float = 0.0) -> Emissions:
        """Emissions for ``audio`` (16 kHz mono). Long inputs run in 30 s windows with 1 s of context."""
        audio = np.asarray(audio, dtype=np.float32)
        n = len(audio)
        if n < 400:
            audio = np.pad(audio, (0, 400 - n))
            n = 400
        with self._lock:
            if n <= int((WINDOW_S + 2 * CONTEXT_S) * SAMPLE_RATE):
                lp = self._forward(audio)
                return Emissions(lp, n / SAMPLE_RATE / len(lp), offset)
            frame_s = 0.02
            total = int(np.ceil(n / SAMPLE_RATE / frame_s))
            out = None
            win, ctx = int(WINDOW_S * SAMPLE_RATE), int(CONTEXT_S * SAMPLE_RATE)
            for a in range(0, n, win):
                b = min(n, a + win)
                lo, hi = max(0, a - ctx), min(n, b + ctx)
                lp = self._forward(audio[lo:hi])
                fs = (hi - lo) / SAMPLE_RATE / len(lp)
                if out is None:
                    out = np.full((total, lp.shape[1]), -1e4, dtype=np.float32)
                    out[:, self.blank] = 0.0
                f0, f1 = round(a / SAMPLE_RATE / frame_s), min(total, round(b / SAMPLE_RATE / frame_s))
                times = (np.arange(f0, f1) * frame_s - lo / SAMPLE_RATE) / fs
                idx = np.clip(np.round(times).astype(int), 0, len(lp) - 1)
                out[f0:f1] = lp[idx]
            return Emissions(out, frame_s, offset)

    def song_emissions(
        self,
        audio: np.ndarray,
        regions: list[tuple[float, float]],
        merge_gap_s: float = 2.0,
        context_s: float = 0.5,
        progress=None,
    ) -> Emissions:
        """Emissions on the song's timeline (20 ms frames), computed only inside the voiced ``regions``; every
        other frame is pure blank. One computation serves both the ASR path and the reference alignment."""
        duration = len(audio) / SAMPLE_RATE
        frame_s = 0.02
        total = int(np.ceil(duration / frame_s)) + 1
        merged: list[list[float]] = []
        for s, e in regions:
            s, e = max(0.0, s - context_s), min(duration, e + context_s)
            if merged and s - merged[-1][1] < merge_gap_s:
                merged[-1][1] = max(merged[-1][1], e)
            else:
                merged.append([s, e])
        out = None
        for k, (s, e) in enumerate(merged):
            a, b = int(s * SAMPLE_RATE), int(e * SAMPLE_RATE)
            em = self.emissions(audio[a:b], offset=a / SAMPLE_RATE)
            if out is None:
                out = np.full((total, em.logp.shape[1]), -1e4, dtype=np.float32)
                out[:, self.blank] = 0.0
            f0, f1 = int(np.ceil(s / frame_s)), min(total, int(e / frame_s))
            idx = np.clip(
                np.round((np.arange(f0, f1) * frame_s - em.offset) / em.frame_s).astype(int), 0, len(em.logp) - 1
            )
            out[f0:f1] = em.logp[idx]
            if progress is not None:
                progress(k + 1, len(merged))
        if out is None:
            vocab = len(self.processor.tokenizer.get_vocab())
            out = np.full((total, vocab), -1e4, dtype=np.float32)
            out[:, self.blank] = 0.0
        return Emissions(out, frame_s, 0.0)

    # ------------------------------------------------------------------ alignment
    def tokens_for(self, words: list[str]) -> tuple[list[int], list[tuple[int, int] | None]]:
        """Token sequence for ``words`` (delimiter between words) and each word's token index range."""
        tokens: list[int] = []
        ranges: list[tuple[int, int] | None] = []
        delim = self.mapper.delimiter_id
        for w in words:
            ids = self.mapper.word_ids(w)
            if not ids:
                ranges.append(None)
                continue
            if tokens and delim is not None:
                tokens.append(delim)
            ranges.append((len(tokens), len(tokens) + len(ids)))
            tokens.extend(ids)
        return tokens, ranges

    def align_words(self, em: Emissions, words: list[str], frames: np.ndarray | None = None) -> list[WordTiming | None]:
        """Align ``words`` (in order) to ``em``. Words without alignable characters get None.

        ``frames`` selects (and concatenates) the frames of ``em`` to align against, e.g. the voiced pieces of one
        Whisper window; times are reported on ``em``'s timeline either way.
        """
        tokens, ranges = self.tokens_for(words)
        if not tokens:
            return [None] * len(words)
        logp = em.logp if frames is None else em.logp[frames]
        spans = ctc.align(logp, tokens, self.blank)
        if spans is None:
            return [None] * len(words)

        def start_time(f):
            return em.time(f if frames is None else frames[f])

        def end_time(f_excl):
            return em.time(f_excl) if frames is None else em.time(frames[f_excl - 1] + 1)

        out: list[WordTiming | None] = []
        for r in ranges:
            if r is None:
                out.append(None)
                continue
            a, b = r
            ws = spans[a:b]
            out.append(WordTiming(start_time(ws[0].start), end_time(ws[-1].end), float(np.mean([s.score for s in ws]))))
        return out


class Aligner:
    """Lazily loads one alignment model per language."""

    def __init__(
        self,
        models: dict[str, str] | None = None,
        threads: int | None = None,
        precision: str = "auto",
        cache_dir: str | None = None,
        local_files_only: bool = False,
    ):
        self.registry = {**ALIGN_MODELS, **(models or {})}
        self.threads = threads
        self.precision = resolve_precision(precision)
        self.cache_dir = cache_dir
        self.local_files_only = local_files_only
        self._models: dict[str, AlignModel] = {}
        self._lock = threading.Lock()

    def supports(self, language: str | None) -> bool:
        return bool(language) and language in self.registry

    def model(self, language: str) -> AlignModel:
        with self._lock:
            if language not in self._models:
                name = self.registry[language]
                log.info("loading alignment model %s for %s", name, language)
                self._models[language] = AlignModel(
                    name, self.threads, self.precision, self.cache_dir, self.local_files_only
                )
            return self._models[language]
