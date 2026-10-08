"""Command line: ``turbo-lyrics vocals.mp3 [--reference lyrics.txt] [-o out.json]``."""

from __future__ import annotations

import argparse
import json
import logging
import os
import re
import sys
import time

from . import __version__


def _read_reference(path: str) -> tuple[list[str], list[float | None] | None]:
    """Plain text (one line per lyric line) or LRC (``[mm:ss.xx] line``)."""
    with open(path, encoding="utf-8") as handle:
        raw = handle.read().splitlines()
    lines, times, synced = [], [], False
    for row in raw:
        m = re.match(r"^\[(\d+):(\d+(?:[.:]\d+)?)\]\s*(.*)$", row.strip())
        if m:
            synced = True
            text = m.group(3).strip()
            if text:
                lines.append(text)
                times.append(int(m.group(1)) * 60 + float(m.group(2).replace(":", ".")))
        elif row.strip() and not re.fullmatch(r"\[.*\]", row.strip()):
            lines.append(row.strip())
            times.append(None)
    return lines, (times if synced else None)


def info() -> dict:
    out = {"turbo_lyrics": __version__, "cpu_count": os.cpu_count()}
    try:
        import ctranslate2

        out["ctranslate2"] = ctranslate2.__version__
        out["ct2_compute_types_cpu"] = sorted(ctranslate2.get_supported_compute_types("cpu"))
    except Exception as e:  # pragma: no cover - environment report only
        out["ctranslate2"] = f"unavailable: {e}"
    try:
        import torch

        out["torch"] = torch.__version__
        out["torch_cpu_capability"] = torch.backends.cpu.get_cpu_capability()
    except Exception as e:  # pragma: no cover
        out["torch"] = f"unavailable: {e}"
    if sys.platform.startswith("linux"):
        try:
            with open("/proc/cpuinfo") as handle:
                flags = next((ln for ln in handle if ln.startswith("flags")), "")
            out["cpu_flags"] = sorted(
                f for f in ("avx2", "avx512f", "avx512_vnni", "avx512_bf16", "amx_bf16") if f" {f}" in flags
            )
        except OSError:
            pass
    return out


def main(argv=None) -> int:
    p = argparse.ArgumentParser(prog="turbo-lyrics", description="Word-timed lyrics from a sung vocal stem (CPU).")
    p.add_argument("audio", nargs="?", help="vocal stem (any ffmpeg-readable file)")
    p.add_argument("-o", "--output", help="write the WhisperX-compatible JSON here (default: stdout summary only)")
    p.add_argument("--reference", help="known lyrics (.txt one line per lyric line, or .lrc) for forced alignment")
    p.add_argument(
        "--mode",
        choices=["auto", "asr", "reference"],
        default="auto",
        help="auto: reference alignment when it passes the gate, else ASR (default)",
    )
    p.add_argument("--language", help="force the language (ISO 639-1)")
    p.add_argument("--model", default="large-v3-turbo")
    p.add_argument("--compute-type", default="int8")
    p.add_argument("--threads", type=int, default=min(8, os.cpu_count() or 8))
    p.add_argument("--beam-size", type=int, default=1)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--align-precision", choices=["auto", "fp32", "bf16"], default="auto")
    p.add_argument("--no-vad", action="store_true", help="fixed 30 s windows instead of packed voiced regions")
    p.add_argument("--info", action="store_true", help="print the environment report and exit")
    p.add_argument(
        "--download",
        metavar="LANGS",
        help="fetch the Whisper model and the aligners for these "
        "comma-separated languages (e.g. en,de) into the cache and exit (Docker build step)",
    )
    p.add_argument("--cache-dir", help="model cache directory (default: the Hugging Face cache)")
    p.add_argument("-v", "--verbose", action="store_true")
    args = p.parse_args(argv)

    if args.info:
        print(json.dumps(info(), indent=2))
        return 0
    if args.download:
        from .transcriber import Transcriber

        tr = Transcriber(args.model, args.compute_type, args.threads, cache_dir=args.cache_dir)
        print(json.dumps(tr.warm([x.strip() for x in args.download.split(",") if x.strip()])))
        return 0
    if not args.audio:
        p.error("audio is required")
    logging.basicConfig(level=logging.INFO if args.verbose else logging.WARNING)

    from .asr import AsrOptions
    from .transcriber import Transcriber

    tr = Transcriber(
        args.model,
        args.compute_type,
        args.threads,
        align_precision=args.align_precision,
        cache_dir=args.cache_dir,
        asr=AsrOptions(beam_size=args.beam_size, batch_size=args.batch_size),
    )
    t0 = time.perf_counter()
    song = tr.prepare(args.audio)
    lines, times = _read_reference(args.reference) if args.reference else ([], None)
    result = None
    if lines and args.mode in ("auto", "reference"):
        ref = tr.align_reference(song, lines, line_times=times, language=args.language)
        gate = ref.data["gate"]
        print(
            f"reference alignment: {'passed' if gate['passed'] else 'rejected'} {gate['reasons']} {gate['metrics']}",
            file=sys.stderr,
        )
        if gate["passed"] or args.mode == "reference":
            result = ref
    if result is None:
        if args.mode == "reference":
            p.error("--mode reference needs --reference")
        result = tr.transcribe(song, language=args.language, reference=lines or None)
    wall = time.perf_counter() - t0
    data = result.data
    n_words = len(result.words)
    print(
        f"{args.audio}: {data.get('detected_language')} | {len(data['segments'])} segments, {n_words} words | "
        f"wall {wall:.1f} s | RTF {wall / song.duration:.3f} | {data.get('timings')}",
        file=sys.stderr,
    )
    if args.output:
        with open(args.output, "w", encoding="utf-8") as handle:
            json.dump(data, handle, ensure_ascii=False, indent=1)
    else:
        for s in data["segments"]:
            print(f"[{s['start']:7.2f} - {s['end']:7.2f}] {s['text']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
