"""Audio input: any ffmpeg-readable file or an array, as 16 kHz mono float32."""

from __future__ import annotations

import subprocess

import numpy as np

SAMPLE_RATE = 16000


def decode(path: str, sample_rate: int = SAMPLE_RATE) -> np.ndarray:
    proc = subprocess.run(
        [
            "ffmpeg",
            "-v",
            "error",
            "-nostdin",
            "-i",
            str(path),
            "-vn",
            "-f",
            "f32le",
            "-acodec",
            "pcm_f32le",
            "-ac",
            "1",
            "-ar",
            str(sample_rate),
            "pipe:1",
        ],
        capture_output=True,
        timeout=600,
        check=False,
    )
    if proc.returncode != 0:
        raise RuntimeError(f"ffmpeg could not decode {path}: {proc.stderr.decode(errors='replace').strip()[-300:]}")
    data = np.frombuffer(proc.stdout, dtype=np.float32).copy()
    if data.size == 0:
        raise RuntimeError(f"{path} contains no audio")
    return data


def to_mono_16k(audio: np.ndarray, sr: int) -> np.ndarray:
    """(n,), (channels, n) or (n, channels) float audio at ``sr`` -> 16 kHz mono float32."""
    a = np.asarray(audio, dtype=np.float32)
    if a.ndim == 2:
        a = a.mean(axis=0) if a.shape[0] <= 8 and a.shape[0] < a.shape[1] else a.mean(axis=1)
    if sr != SAMPLE_RATE:
        import soxr

        a = soxr.resample(a, sr, SAMPLE_RATE, quality="HQ")
    return np.ascontiguousarray(a, dtype=np.float32)


def load(audio, sr: int | None = None) -> np.ndarray:
    """A path, a 16 kHz mono array, or any array with its sample rate ``sr``."""
    if isinstance(audio, (str, bytes)) or hasattr(audio, "__fspath__"):
        return decode(str(audio))
    return to_mono_16k(audio, sr or SAMPLE_RATE)
