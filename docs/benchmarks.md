# Benchmarks

## Apple M5 Pro (development machine, not the target)

The test file is a 4:12 live song (German musical), BS-RoFormer vocal stem, 238 s voiced. Default settings were
large-v3-turbo int8 with 8 threads, alignment in fp32 on this CPU, and the alignment model
jonatasgrosman/wav2vec2-large-xlsr-53-german.

| run | windows | language | ASR | emissions | other | wall | RTF |
|---|---:|---:|---:|---:|---:|---:|---:|
| ASR, language auto-detected | 9 | 3.8 s | 25.7 s | 13.2 s | 0.3 s | 46.2 s | 0.183 |
| ASR, batch size 16 | 9 | 3.8 s | 28.4 s | 12.6 s | 0.3 s | 48.3 s | 0.192 |
| ASR, before packing fix (cut at silences only) | 11 | given | 29.9 s | 11.6 s | 0.3 s | 41.8 s | – |
| reference path | – | – | – | 10.6 s | 0.4 s | 13.8 s | 0.055 |

## AMD Ryzen 7 8845HS (target)

Run this inside the production container, with nothing else heavy running:

```bash
python benchmarks/bench.py /data/songs/*/vocals.mp3 --runs 1 --json bench.json
python benchmarks/bench.py song.mp3 --sweep compute_type=int8,int8_float32 threads=6,8,16 batch_size=4,8 --runs 3
```

Check `turbo-lyrics --info` first: on Zen 4 it should list `avx512_bf16` (the aligner then uses bf16 autocast)
and `avx512_vnni` (the CTranslate2 int8 path).
