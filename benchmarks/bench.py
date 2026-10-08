"""Speed/quality sweep for one machine (run it on the target box, e.g. inside the production container).

    python benchmarks/bench.py vocals1.mp3 vocals2.mp3 --compare-dir ref/ --json out.json
    python benchmarks/bench.py song.mp3 --sweep compute_type=int8,int8_float32 threads=6,8 batch_size=4,8

Each configuration runs ``--runs`` times per file (median reported). With ``--compare-dir``, ``<stem>.json`` in that
directory (e.g. an existing WhisperX ``lyrics_raw.json``) is compared word by word: text similarity and start
time deltas. Sweeps are one factor at a time around the defaults (a full grid explodes quickly).
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
import time

from turbo_lyrics import AsrOptions, Transcriber, compare

DEFAULTS = {
    "model": "large-v3-turbo",
    "compute_type": "int8",
    "threads": 8,
    "batch_size": 8,
    "beam_size": 1,
    "align_precision": "auto",
}


def configs(sweeps: list[str]) -> list[dict]:
    out = [dict(DEFAULTS)]
    for spec in sweeps:
        key, values = spec.split("=", 1)
        for raw in values.split(","):
            value = int(raw) if raw.isdigit() else raw
            if value != DEFAULTS[key]:
                out.append({**DEFAULTS, key: value})
    return out


def run(cfg: dict, files: list[str], runs: int, compare_dir: str | None, language: str | None) -> dict:
    tr = Transcriber(
        cfg["model"],
        cfg["compute_type"],
        cfg["threads"],
        align_precision=cfg["align_precision"],
        asr=AsrOptions(beam_size=cfg["beam_size"], batch_size=cfg["batch_size"]),
    )
    t0 = time.perf_counter()
    tr.warm((language,) if language else ("en", "de"))
    report = {"config": cfg, "load_s": round(time.perf_counter() - t0, 2), "files": []}
    for path in files:
        walls, result = [], None
        for _ in range(runs):
            t = time.perf_counter()
            result = tr.transcribe(path, language=language)  # fresh prepare: emissions are not reused across runs
            walls.append(time.perf_counter() - t)
        duration = result.data["vad"]["duration_s"]
        entry = {
            "file": os.path.basename(path),
            "duration_s": duration,
            "wall_s": round(statistics.median(walls), 2),
            "rtf": round(statistics.median(walls) / duration, 4),
            "timings": result.timings,
            "windows": result.data["vad"]["windows"],
            "language": result.data["detected_language"],
            "words": len(result.words),
        }
        if compare_dir:
            ref = os.path.join(compare_dir, os.path.splitext(os.path.basename(path))[0] + ".json")
            if os.path.exists(ref):
                with open(ref, encoding="utf-8") as handle:
                    entry["vs_reference"] = compare(result.data, json.load(handle))
        report["files"].append(entry)
        print(json.dumps(entry), flush=True)
    return report


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("files", nargs="+")
    p.add_argument("--sweep", nargs="*", default=[], help="key=v1,v2 (keys: " + ", ".join(DEFAULTS) + ")")
    p.add_argument("--runs", type=int, default=1)
    p.add_argument("--language")
    p.add_argument("--compare-dir")
    p.add_argument("--json")
    args = p.parse_args()
    reports = [run(cfg, args.files, args.runs, args.compare_dir, args.language) for cfg in configs(args.sweep)]
    for r in reports:
        rtfs = [f["rtf"] for f in r["files"]]
        print(f"{r['config']} -> median RTF {statistics.median(rtfs):.4f}")
    if args.json:
        with open(args.json, "w", encoding="utf-8") as handle:
            json.dump(reports, handle, indent=1)


if __name__ == "__main__":
    main()
