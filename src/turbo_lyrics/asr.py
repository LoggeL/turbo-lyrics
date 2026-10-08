"""Whisper through faster-whisper/CTranslate2 on the CPU, fed with packed voiced windows."""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass

import numpy as np

from .vad import SAMPLE_RATE, Pack

log = logging.getLogger(__name__)


@dataclass
class AsrOptions:
    beam_size: int = 1  # greedy: ~2-3x faster decoder, fewer repetition loops on sung vocals
    batch_size: int = 8  # windows per encoder call (a 4-minute song needs ~6-8)
    temperature: float = 0.0
    no_speech_threshold: float = 0.6  # Whisper's silence rule: drop windows above this no-speech probability ...
    log_prob_threshold: float = -1.0  # ... when their average log-probability is also below this
    initial_prompt: str | None = None
    word_timestamps: bool = False  # Whisper's own (cross-attention) word times; used when no aligner exists


@dataclass
class AsrWindow:
    text: str
    avg_logprob: float
    no_speech_prob: float
    compression_ratio: float
    words: list[tuple[str, float, float, float]] | None = None  # (word, start, end, prob) in window time


class WhisperASR:
    def __init__(
        self,
        model: str = "large-v3-turbo",
        compute_type: str = "int8",
        threads: int = 8,
        download_root: str | None = None,
        local_files_only: bool = False,
    ):
        from faster_whisper import BatchedInferencePipeline, WhisperModel

        t0 = time.perf_counter()
        self.model_name = model
        self.compute_type = compute_type
        self.model = WhisperModel(
            model,
            device="cpu",
            compute_type=compute_type,
            cpu_threads=int(threads),
            num_workers=1,
            download_root=download_root,
            local_files_only=local_files_only,
        )
        self.pipeline = BatchedInferencePipeline(self.model)
        self.load_s = time.perf_counter() - t0

    def transcribe(
        self, packs: list[Pack], wav: np.ndarray, language: str, opts: AsrOptions, progress=None
    ) -> list[AsrWindow]:
        """One AsrWindow per pack (same order). ``progress(done, total)`` is called after every batch."""
        if not packs:
            return []
        gap = int(SAMPLE_RATE * 1.0)
        audios = [p.audio(wav) for p in packs]
        # Lay the packed windows out back to back (1 s apart) and let faster-whisper slice them again.
        offsets, pos = [], 0
        for a in audios:
            offsets.append(pos)
            pos += len(a) + gap
        joined = np.zeros(pos, dtype=np.float32)
        for a, o in zip(audios, offsets):
            joined[o : o + len(a)] = a
        clips = [{"start": o / SAMPLE_RATE, "end": (o + len(a)) / SAMPLE_RATE} for a, o in zip(audios, offsets)]

        results: list[AsrWindow | None] = [None] * len(packs)
        segments, _info = self.pipeline.transcribe(
            joined,
            language=language,
            task="transcribe",
            beam_size=opts.beam_size,
            batch_size=opts.batch_size,
            temperature=opts.temperature,
            without_timestamps=True,
            word_timestamps=opts.word_timestamps,
            clip_timestamps=clips,
            vad_filter=False,
            condition_on_previous_text=False,
            initial_prompt=opts.initial_prompt,
            no_speech_threshold=opts.no_speech_threshold,
            log_prob_threshold=opts.log_prob_threshold,
        )
        done = 0
        batch = max(1, opts.batch_size)
        starts = np.array([c["start"] for c in clips])
        for seg in segments:
            # a window can come back as several segments: attribute each to the window it starts in
            k = max(0, int(np.searchsorted(starts, seg.start + 0.01, side="right")) - 1)
            words = None
            if seg.words:
                base = clips[k]["start"]
                words = [(w.word.strip(), w.start - base, w.end - base, w.probability) for w in seg.words]
            prev = results[k]
            if prev is None:
                results[k] = AsrWindow(
                    seg.text.strip(), seg.avg_logprob, seg.no_speech_prob, seg.compression_ratio, words
                )
            else:
                prev.text = (prev.text + " " + seg.text.strip()).strip()
                if words:
                    prev.words = (prev.words or []) + words
            finished = min(len(packs), (k // batch + 1) * batch)  # segments arrive after their batch decoded
            if finished > done:
                done = finished
                if progress is not None:
                    progress(done, len(packs))
        if progress is not None and done < len(packs):  # trailing windows without any text
            progress(len(packs), len(packs))
        return [r if r is not None else AsrWindow("", 0.0, 1.0, 0.0) for r in results]
