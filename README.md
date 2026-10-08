# turbo-lyrics

**Word-timed lyrics from a sung vocal stem on a CPU, with WhisperX-compatible output.**

turbo-lyrics is built for karaoke pipelines that already have an isolated vocal stem (for example from
[turbo-roformer](https://github.com/LoggeL/turbo-roformer)). It replaces a cloud WhisperX call with two local paths:

- **Reference path.** When the lyrics are known (lrclib, LRC files), the text is force-aligned straight to the
  vocals, with no speech recognition. A gate checks the result and rejects bad alignments.
- **ASR path.** Otherwise the voiced parts are packed into as few 30 s Whisper windows as possible, transcribed
  with faster-whisper (large-v3-turbo, int8, greedy decoding, batched), and every word is CTC-aligned with
  wav2vec2.

Both paths share one wav2vec2 pass per song. Measurements for a 4:12 live song (German musical, sung almost
throughout) on an Apple M5 Pro CPU:

| path | wall | RTF | what runs |
|---|---:|---:|---|
| reference (known lyrics, gate passed) | **13.8 s** | **0.055** | wav2vec2 emissions 10.6 s + Viterbi 0.08 s |
| ASR (language from lyrics or forced) | 42 s | 0.17 | Whisper 25.6 s (9 windows) + emissions 13 s + alignment 0.04 s |
| ASR (language auto-detected) | 46 s | 0.18 | + 3.8 s language detection on the fullest window |

On the same stem, word starts from the reference path and the ASR path agree to a median of **0 ms** (99.7 % of
words within 100 ms). Two runs of the ASR path agree to a median of 12 ms. Numbers for an AMD Zen 4 home server
are measured with [`benchmarks/bench.py`](benchmarks/bench.py); see [docs/benchmarks.md](docs/benchmarks.md).

## Quickstart

```bash
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install "turbo-lyrics @ git+https://github.com/LoggeL/turbo-lyrics@v0.1.0"

turbo-lyrics vocals.mp3 -o lyrics.json                      # ASR path
turbo-lyrics vocals.mp3 --reference song.lrc -o lyrics.json  # reference path, ASR fallback if the gate fails
turbo-lyrics --download en,de                                # prefetch models (Docker build step)
turbo-lyrics --info                                          # CPU features, CTranslate2/torch report
```

```python
from turbo_lyrics import Transcriber

tr = Transcriber()  # large-v3-turbo int8, 8 threads; load once, reuse
song = tr.prepare("vocals.mp3")  # decode + VAD, a few hundred ms
ref = tr.align_reference(song, lines, line_times=lrc_times)
result = ref if ref.data["gate"]["passed"] else tr.transcribe(song, reference=lines)
result.data  # {"segments": [{"start", "end", "text", "words": [{"word", "start", "end", "score"}]}], ...}
```

`progress(done, total)` callbacks are accepted by `transcribe` and `align_reference`. They may raise to abort a
call, which is how a job queue can preempt work.

## Output

The output is the WhisperX JSON shape: `segments[].words[]` with `word`, `start`, `end` and `score`, where `score`
is the mean CTC character probability on WhisperX's scale. Segments are sentences. On top of WhisperX:

- **Word timing guarantees:**
  - every word has `start` ≥ 0, `end` > `start` (at least 50 ms) and `end` ≤ the audio duration;
  - words never overlap;
  - digits and symbols get interpolated times instead of none.
- **Word ends follow the voice.** Ends are trimmed to the last voiced frame, so a word does not extend across an
  instrumental gap.
- **Extra fields:** `detected_language`, `language_source` (given / reference / whisper), `alignment`, `engine`,
  `vad` and `timings`.
- **Reference path additions:** `source: "reference"`, `line_starts` (word index of each lyric line) and `gate`
  (passed, reasons, metrics).

There is no speaker diarization. Speech diarizers are unreliable on duets, and running one on a CPU costs more
than the transcription itself.

## How it is fast

The details are in [docs/how-it-works.md](docs/how-it-works.md). In short:

1. **Packed windows.** Whisper pads every input to 30 s, so encoder cost follows the number of windows, not the
   voiced seconds. An energy VAD on the stem finds the voiced regions, and they are concatenated with 0.4 s gaps;
   long regions are cut at the latest breath that still fits the current window. A sung-through 4:12 song needs
   9 windows instead of 11.
2. **Whisper settings.** large-v3-turbo (4 decoder layers instead of 32), int8 weights, greedy decoding, one
   batched encoder call per 8 windows, no timestamp tokens (the CTC aligner provides the times). Windows that
   look like repetition loops are re-decoded with beam search.
3. **Language from the lyrics.** When reference lyrics exist, the language comes from a stopword count over the
   text. Otherwise Whisper's detector runs on the fullest window, not on the intro, and on up to 3 windows when
   it is unsure. A spoken or shouted intro made single-window detection pick English for a German song.
4. **One wav2vec2 pass per song, voiced regions only.** The ASR path and the reference path read the same cached
   emissions; on Zen 4 / Sapphire Rapids the pass runs in bf16 autocast (`precision="auto"`).
5. **numpy Viterbi.** It handles a whole song with a wildcard token between lyric lines in about 0.1 s.

## Alignment models

Only permissively licensed (Apache-2.0) CTC checkpoints are registered: `facebook/wav2vec2-base-960h` for English
and `jonatasgrosman/wav2vec2-large-xlsr-53-*` for de, fr, es, it, pt, nl, pl, ru, fi, hu, el, fa, ar, ja and zh.
WhisperX's VoxPopuli defaults for de/fr/es/it are CC-BY-NC and are not used. Pass `align_models={"xx": "org/model"}`
for anything else. Languages without an aligner fall back to Whisper's cross-attention word times.

## Development

```bash
pip install -e ".[dev]"
pytest -q              # unit tests, no models needed
pytest -q --models     # also the tests that download the real models
ruff check . && ruff format --check .
```

MIT licensed. Model weights keep their own licenses.
