# Changelog

## 0.1.0

- Packed energy VAD for vocal stems, faster-whisper large-v3-turbo int8 (batched, greedy) with loop re-decoding.
- numpy CTC Viterbi (WhisperX trellis, sung-vowel hold, wildcard token), wav2vec2 emissions shared per song.
- Reference-lyrics forced alignment with an acceptance gate (scores, voicing, coverage, LRC drift).
- WhisperX-compatible output with a strict timing contract; `compare()` for word-level agreement.
- CLI (`--reference`, `--mode`, `--download`, `--info`) and `benchmarks/bench.py`.
