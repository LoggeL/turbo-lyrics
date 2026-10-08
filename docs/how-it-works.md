# How turbo-lyrics works

The target is a home server CPU (AMD Ryzen 7 8845HS: 8 Zen 4 cores, AVX-512, AVX512_BF16, no GPU). The job is to
turn a separated vocal stem into karaoke word timings. That is WhisperX's job, done more cheaply. The design
follows from where WhisperX spends its time on a CPU.

| WhisperX step on a CPU (4-minute song) | cost | what turbo-lyrics does instead |
|---|---|---|
| model load per call (cog/Replicate style) | 10-20 s | load once in a long-lived worker |
| large-v3 decoder (32 layers), beam 5 | dominant after the encoder | large-v3-turbo (4 layers), greedy |
| encoder on every 30 s window, VAD cut at short pauses | ~one window per 30 s of song | voiced regions packed into full windows |
| language detection on the first 30 s (often an intro) | wrong-language reruns | language from the lyrics, or detection on the fullest windows |
| pyannote diarization | more than the ASR itself | none (speaker labels are cosmetic for karaoke) |
| wav2vec2 alignment per segment | small | one pass per song, shared with the reference path |

## 1. Energy VAD and window packing (`vad.py`)

A BS-RoFormer vocal stem is close to silent where nobody sings. Frame RMS in 20 ms frames, compared to the song's
own loud level (the 95th percentile), separates voice from bleed reliably:

- **Threshold:** a frame is voiced when it is within 32 dB of that level and above −55 dBFS.
- **Region shaping:** dips under 0.5 s stay inside a region, regions shorter than 120 ms are dropped, and each
  region keeps 150 ms of padding.

Whisper pads every input to a 30 s window, so cost follows the *number of windows*. `pack()` therefore
concatenates voiced regions, with 0.4 s of silence between them, into windows of up to 29.5 s.

- **Long regions:** a region that is longer than a window has to be cut anyway. It is cut at the latest frame
  within 1.5 dB of the quietest one in the last 20 % of the remaining room (a breath), which fills the current
  window instead of starting a new one.
- **Time mapping:** `Pack.to_original()` maps window time back to song time.

On a sung-through 4:12 live song this gives 9 windows. Cutting at silences without packing gave 11, and fixed
30 s windows would give 9 windows that are each a third empty.

## 2. Whisper (`asr.py`)

faster-whisper's `BatchedInferencePipeline` receives the packed windows as `clip_timestamps`:

- large-v3-turbo with int8 weights through CTranslate2;
- greedy decoding (`beam_size=1`) and `without_timestamps=True`, so there are no timestamp tokens to decode
  because the CTC aligner provides the times;
- `condition_on_previous_text=False`.

Windows are filtered after decoding:

- Whisper's silence rule drops a window when no-speech probability > 0.6 and average log-probability < −1.
- Known hallucinations ("Untertitel der Amara.org-Community", "Thanks for watching", …) are dropped.
- A window with compression ratio > 2.6 is re-decoded once with beam 5 at temperature 0.4, and the less repetitive
  text is kept.

## 3. CTC alignment (`align.py`, `ctc.py`)

Emissions are computed once per song and language, inside the voiced regions only. Regions closer than 2 s are
merged, each gets 0.5 s of context, and inputs longer than 30 s run in windows with 1 s of context. All other
frames are pure blank. On CPUs with AVX512_BF16 the forward pass runs in bf16 autocast (oneDNN).

`ctc.align` is a vectorised numpy Viterbi over the WhisperX trellis:

- **Entering and holding tokens:** a token is entered on the frame that emits it and held until the next token
  starts. Holding scores `max(p(blank), p(token))`, because sung vowels keep emitting the token itself, which
  WhisperX's blank-only hold penalises.
- **Scores:** a character's score is the mean path probability over its span, and a word's score is the mean over
  its characters. That is WhisperX's scale, so existing confidence thresholds keep their meaning.
- **ASR path:** the frames of each window's pieces are concatenated, and the window's words are aligned against
  them.

After alignment the timings are post-processed (`output.py`):

- Words without alignable characters (digits, symbols) are interpolated between their neighbours.
- Ends are trimmed to the last voiced frame plus 80 ms. A CTC span holds until the next token, which can be after
  an instrumental break.
- The contract is enforced: start ≥ 0, end > start (≥ 50 ms), no overlaps, end ≤ duration.

## 4. Reference alignment and the gate (`transcriber.py`)

With known lyrics, the token sequence is `★ line₁ ★ line₂ ★ … ★`. `★` is a wildcard that matches any frame
(score: the best class of the frame) and absorbs ad-libs, intros, instrumental parts and repeats the text lacks.
One Viterbi over the whole song (about 15 000 frames × 3 000 tokens) takes about 0.1 s.

Forced alignment always produces *some* answer, so a gate decides whether to trust it:

| metric | correct text | 40 % of lines missing | wrong song | threshold |
|---|---:|---:|---:|---:|
| mean word score | 0.93 | 0.93 | 0.45 | ≥ 0.6 |
| words scoring < 0.3 | 0.7 % | 0.4 % | 31 % | ≤ 15 % |
| word time that is voiced | 99 % | 99 % | 94 % | ≥ 70 % |
| sung time outside every line | 32 % ¹ | 59 % | 46 % | ≤ 40 % |
| LRC line starts within 2 s (after global offset) | 98 % | – | – | ≥ 70 % |

¹ This live recording has shouting and crowd vocals that no lyric covers. On studio recordings the value is
expected to be lower.

The thresholds come from this one song and its controlled variants. They are a starting point: integrate in
shadow mode first (compute both paths and compare with `turbo_lyrics.compare`) and recalibrate on a real library.
