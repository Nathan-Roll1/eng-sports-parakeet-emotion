#!/usr/bin/env python3
from __future__ import annotations

import argparse
import io
import json
import random
import re
from collections import Counter
from pathlib import Path
from typing import Any

from train_parakeet_emotion_tokens import EMOTION_TOKENS, primary_emotion, strip_all_emotion_tokens


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-parquet", default="data/emotion_mixture_train")
    parser.add_argument("--output-dir", default="data/emotion_mixture_concat_iu")
    parser.add_argument("--seed", type=int, default=641)
    parser.add_argument("--sampling-rate", type=int, default=16000)
    parser.add_argument("--min-iu-duration-seconds", type=float, default=0.35)
    parser.add_argument("--max-iu-duration-seconds", type=float, default=20.0)
    parser.add_argument("--min-ius", type=int, default=2)
    parser.add_argument("--max-ius", type=int, default=6)
    parser.add_argument("--target-duration-seconds", type=float, default=28.0)
    parser.add_argument("--max-duration-seconds", type=float, default=55.0)
    parser.add_argument("--silence-ms", type=float, default=80.0)
    parser.add_argument("--rows-per-shard", type=int, default=2500)
    parser.add_argument("--max-input-rows", type=int, default=0)
    return parser.parse_args()


def parquet_files_from_path(path: Path) -> list[Path]:
    if path.is_dir():
        files = sorted(path.glob("*.parquet"))
        if not files:
            files = sorted((path / "data").glob("*.parquet"))
    elif any(char in str(path) for char in "*?[]"):
        import glob

        files = [Path(item) for item in sorted(glob.glob(str(path)))]
    else:
        files = [path]
    if not files:
        raise FileNotFoundError(f"no parquet files found at {path}")
    return files


def audio_payload(audio: dict[str, Any]) -> bytes | None:
    value = audio.get("bytes")
    if value:
        return bytes(value)
    path = audio.get("path")
    if path and Path(path).exists():
        return Path(path).read_bytes()
    return None


def duration_seconds(audio: dict[str, Any]) -> float:
    import soundfile as sf

    payload = audio_payload(audio)
    if not payload:
        return 0.0
    info = sf.info(io.BytesIO(payload))
    return float(info.frames) / float(info.samplerate)


def row_duration_seconds(row: dict[str, Any]) -> float:
    for key in ["duration_seconds", "duration", "Duration"]:
        value = row.get(key)
        if value is not None:
            try:
                return float(value)
            except (TypeError, ValueError):
                pass
    audio = row.get("audio")
    return duration_seconds(audio) if isinstance(audio, dict) else 0.0


def clean_body(text: str) -> str:
    body = strip_all_emotion_tokens(text)
    body = "".join(ch for ch in body if ch.isprintable() or ch.isspace())
    return re.sub(r"\s+", " ", body).strip()


def source_name(audio: dict[str, Any]) -> str:
    path = str(audio.get("path") or "")
    return path.split("/", 1)[0] if "/" in path else "unknown"


def load_iu_rows(args: argparse.Namespace) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    import pyarrow.parquet as pq

    files = parquet_files_from_path(Path(args.input_parquet).expanduser())
    rows: list[dict[str, Any]] = []
    skipped: Counter[str] = Counter()
    label_counts: Counter[str] = Counter()
    source_counts: Counter[str] = Counter()
    input_rows = 0

    for file_path in files:
        parquet_file = pq.ParquetFile(file_path)
        for batch in parquet_file.iter_batches(batch_size=512):
            for row in batch.to_pylist():
                input_rows += 1
                if args.max_input_rows > 0 and len(rows) >= args.max_input_rows:
                    break
                audio = row.get("audio")
                if not isinstance(audio, dict):
                    skipped["missing_audio"] += 1
                    continue
                if not audio_payload(audio):
                    skipped["missing_audio_payload"] += 1
                    continue
                text = str(row.get("text") or "").strip()
                token = primary_emotion(text)
                if token not in EMOTION_TOKENS:
                    skipped["missing_emotion_token"] += 1
                    continue
                body = clean_body(text)
                if not body:
                    skipped["empty_body"] += 1
                    continue
                duration = row_duration_seconds(row)
                if not args.min_iu_duration_seconds <= duration <= args.max_iu_duration_seconds:
                    skipped["iu_duration"] += 1
                    continue
                source = source_name(audio)
                rows.append({
                    "audio": audio,
                    "body": body,
                    "token": token,
                    "duration_seconds": duration,
                    "source": source,
                    "source_path": str(audio.get("path") or ""),
                })
                label_counts[token] += 1
                source_counts[source] += 1
            if args.max_input_rows > 0 and len(rows) >= args.max_input_rows:
                break
        if args.max_input_rows > 0 and len(rows) >= args.max_input_rows:
            break

    summary = {
        "input_parquet": str(Path(args.input_parquet).expanduser()),
        "input_parquet_files": [str(path) for path in files],
        "input_rows_seen": input_rows,
        "valid_iu_rows": len(rows),
        "skipped": dict(sorted(skipped.items())),
        "input_label_counts": dict(sorted(label_counts.items())),
        "input_source_counts": dict(sorted(source_counts.items())),
    }
    return rows, summary


def decode_audio(audio: dict[str, Any], target_sampling_rate: int):
    import numpy as np
    import soundfile as sf

    payload = audio_payload(audio)
    if not payload:
        raise ValueError("audio row has no usable bytes or path")
    samples, sampling_rate = sf.read(io.BytesIO(payload), dtype="float32", always_2d=False)
    if samples.ndim > 1:
        samples = samples.mean(axis=1)
    if int(sampling_rate) != int(target_sampling_rate):
        import librosa

        samples = librosa.resample(samples, orig_sr=int(sampling_rate), target_sr=int(target_sampling_rate))
    samples = np.asarray(samples, dtype=np.float32)
    samples = np.nan_to_num(samples, nan=0.0, posinf=0.0, neginf=0.0)
    return np.clip(samples, -1.0, 1.0)


def make_concat_row(group: list[dict[str, Any]], args: argparse.Namespace, output_index: int) -> dict[str, Any]:
    import numpy as np
    import soundfile as sf

    silence_samples = int(round(args.sampling_rate * args.silence_ms / 1000.0))
    silence = np.zeros(silence_samples, dtype=np.float32)
    parts = []
    text_parts: list[str] = []
    source_paths: list[str] = []
    source_counts: Counter[str] = Counter()
    emotion_counts: Counter[str] = Counter()

    for idx, item in enumerate(group):
        if idx:
            parts.append(silence)
        samples = decode_audio(item["audio"], args.sampling_rate)
        parts.append(samples)
        text_parts.extend([item["body"], item["token"]])
        source_paths.append(item["source_path"])
        source_counts[item["source"]] += 1
        emotion_counts[item["token"]] += 1

    combined = np.concatenate(parts) if parts else np.zeros(0, dtype=np.float32)
    duration = float(len(combined)) / float(args.sampling_rate)
    buffer = io.BytesIO()
    sf.write(buffer, combined, args.sampling_rate, format="WAV", subtype="PCM_16")
    return {
        "audio": {"bytes": buffer.getvalue(), "path": f"concat_iu/{output_index:07d}.wav"},
        "text": " ".join(text_parts),
        "duration_seconds": duration,
        "iu_count": len(group),
        "source_paths": source_paths,
        "source_counts_json": json.dumps(dict(sorted(source_counts.items())), sort_keys=True),
        "emotion_counts_json": json.dumps(dict(sorted(emotion_counts.items())), sort_keys=True),
    }


def make_groups(rows: list[dict[str, Any]], args: argparse.Namespace) -> list[list[dict[str, Any]]]:
    rng = random.Random(args.seed)
    shuffled = list(rows)
    rng.shuffle(shuffled)
    groups: list[list[dict[str, Any]]] = []
    idx = 0
    while idx < len(shuffled):
        target_iu_count = rng.randint(args.min_ius, args.max_ius)
        group: list[dict[str, Any]] = []
        duration = 0.0
        while idx < len(shuffled) and len(group) < target_iu_count:
            item = shuffled[idx]
            added = float(item["duration_seconds"])
            if group:
                added += args.silence_ms / 1000.0
            if group and len(group) >= args.min_ius and duration + added > args.max_duration_seconds:
                break
            group.append(item)
            duration += added
            idx += 1
            if len(group) >= args.min_ius and duration >= args.target_duration_seconds:
                break
        if len(group) >= args.min_ius:
            groups.append(group)
        else:
            idx += max(1, len(group))
    return groups


def parquet_schema():
    import pyarrow as pa

    return pa.schema([
        pa.field("audio", pa.struct([
            pa.field("bytes", pa.binary()),
            pa.field("path", pa.string()),
        ])),
        pa.field("text", pa.string()),
        pa.field("duration_seconds", pa.float64()),
        pa.field("iu_count", pa.int32()),
        pa.field("source_paths", pa.list_(pa.string())),
        pa.field("source_counts_json", pa.string()),
        pa.field("emotion_counts_json", pa.string()),
    ])


def write_shard(rows: list[dict[str, Any]], data_dir: Path, shard_idx: int) -> Path:
    import pyarrow as pa
    import pyarrow.parquet as pq

    path = data_dir / f"train-{shard_idx:05d}.parquet"
    table = pa.Table.from_pylist(rows, schema=parquet_schema())
    pq.write_table(table, path)
    return path


def main() -> None:
    args = parse_args()
    if args.min_ius < 2:
        raise ValueError("--min-ius must be at least 2 for concat-IU training")
    if args.max_ius < args.min_ius:
        raise ValueError("--max-ius must be >= --min-ius")
    if args.max_duration_seconds <= 0:
        raise ValueError("--max-duration-seconds must be positive")

    output_dir = Path(args.output_dir).expanduser()
    data_dir = output_dir / "data"
    data_dir.mkdir(parents=True, exist_ok=True)
    for old_file in data_dir.glob("*.parquet"):
        old_file.unlink()
    rows, input_summary = load_iu_rows(args)
    groups = make_groups(rows, args)

    shard_rows: list[dict[str, Any]] = []
    parquet_paths: list[Path] = []
    output_label_counts: Counter[str] = Counter()
    output_source_counts: Counter[str] = Counter()
    iu_counts: Counter[int] = Counter()
    durations: list[float] = []
    decode_skipped = 0

    for output_index, group in enumerate(groups):
        try:
            row = make_concat_row(group, args, output_index)
        except Exception as exc:
            decode_skipped += 1
            print(f"skipping group {output_index}: {type(exc).__name__}: {exc}", flush=True)
            continue
        shard_rows.append(row)
        durations.append(float(row["duration_seconds"]))
        iu_counts[int(row["iu_count"])] += 1
        output_label_counts.update(token for item in group for token in [item["token"]])
        output_source_counts.update(item["source"] for item in group)
        if len(shard_rows) >= args.rows_per_shard:
            parquet_paths.append(write_shard(shard_rows, data_dir, len(parquet_paths)))
            shard_rows = []
    if shard_rows:
        parquet_paths.append(write_shard(shard_rows, data_dir, len(parquet_paths)))

    duration_summary = {}
    if durations:
        sorted_durations = sorted(durations)
        duration_summary = {
            "min": sorted_durations[0],
            "p50": sorted_durations[len(sorted_durations) // 2],
            "p95": sorted_durations[int((len(sorted_durations) - 1) * 0.95)],
            "max": sorted_durations[-1],
            "total_hours": sum(durations) / 3600.0,
        }
    summary = {
        **input_summary,
        "output_dir": str(output_dir),
        "output_data_dir": str(data_dir),
        "output_parquet_files": [str(path) for path in parquet_paths],
        "seed": args.seed,
        "sampling_rate": args.sampling_rate,
        "target_format": "utterance_text <|emotion|> utterance_text <|emotion|> ...",
        "min_ius": args.min_ius,
        "max_ius": args.max_ius,
        "target_duration_seconds": args.target_duration_seconds,
        "max_duration_seconds": args.max_duration_seconds,
        "silence_ms": args.silence_ms,
        "groups_built": len(groups),
        "rows_written": sum(iu_counts.values()),
        "decode_skipped_groups": decode_skipped,
        "output_iu_count_distribution": {str(k): v for k, v in sorted(iu_counts.items())},
        "output_label_counts": dict(sorted(output_label_counts.items())),
        "output_source_counts": dict(sorted(output_source_counts.items())),
        "duration_seconds": duration_summary,
    }
    (output_dir / "dataset_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    (output_dir / "README.md").write_text(
        "# Parakeet Concat-IU Emotion Boundary Dataset\n\n"
        "Each row concatenates multiple IU audio chunks. The target text places an emotion token after each IU, "
        "so the token is both the emotion label for that IU and the boundary marker before the next IU.\n\n"
        f"Rows written: {summary['rows_written']}\n"
    )
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
