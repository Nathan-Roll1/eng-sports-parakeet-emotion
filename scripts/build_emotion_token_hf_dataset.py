#!/usr/bin/env python3
"""Build a two-column audio/text dataset with emotion tokens."""

from __future__ import annotations

import argparse
import json
import pathlib
from collections import Counter
from typing import Any

import pyarrow as pa
import pyarrow.parquet as pq


DEFAULT_INPUT = "runs/eng_sports_processing/hf_iu_parakeet_labeled_dataset/data/train-00000.parquet"
DEFAULT_OUT = "runs/eng_sports_processing/hf_iu_emotion_token_dataset"
EMOTIONS = ("neutral", "joy", "sadness", "anger", "fear", "disgust", "surprise")


ARROW_SCHEMA = pa.schema(
    [
        pa.field("audio", pa.struct([pa.field("bytes", pa.binary()), pa.field("path", pa.string())])),
        pa.field("text", pa.string()),
    ]
)

FEATURES_META = {
    "audio": {"sampling_rate": 16000, "num_channels": 1, "_type": "Audio"},
    "text": {"dtype": "string", "_type": "Value"},
}


def read_rows(path: pathlib.Path) -> list[dict[str, Any]]:
    return pq.read_table(path).to_pylist()


def token_for(emotion: str) -> str:
    return f"<|{emotion}|>"


def build_rows(rows: list[dict[str, Any]], max_duration: float) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    selected: list[dict[str, Any]] = []
    drop_reasons: Counter[str] = Counter()
    source_counts: Counter[str] = Counter()
    emotion_counts: Counter[str] = Counter()

    for row in rows:
        emotion = str(row.get("emotion") or "").strip()
        text = str(row.get("text") or "").strip()
        duration = float(row.get("duration_seconds") or 0.0)
        audio = row.get("audio")

        if not emotion:
            drop_reasons["unlabeled"] += 1
            continue
        if emotion == "low_quality":
            drop_reasons["low_quality"] += 1
            continue
        if emotion not in EMOTIONS:
            drop_reasons[f"unknown_emotion:{emotion}"] += 1
            continue
        if duration >= max_duration:
            drop_reasons["duration_gte_limit"] += 1
            continue
        if not text:
            drop_reasons["empty_text"] += 1
            continue
        if not isinstance(audio, dict) or not audio.get("bytes"):
            drop_reasons["missing_audio"] += 1
            continue

        selected.append({"audio": audio, "text": f"{token_for(emotion)} {text}"})
        source_counts[str(row.get("source_id") or "")] += 1
        emotion_counts[emotion] += 1

    summary = {
        "input_rows": len(rows),
        "row_count": len(selected),
        "max_duration_seconds_exclusive": max_duration,
        "columns": ["audio", "text"],
        "special_tokens": [token_for(emotion) for emotion in EMOTIONS],
        "emotion_counts": dict(sorted(emotion_counts.items())),
        "source_counts": dict(sorted(source_counts.items())),
        "drop_reasons": dict(sorted(drop_reasons.items())),
        "audio_hours": round(
            sum(float(row.get("duration_seconds") or 0.0) for row in rows
                if str(row.get("emotion") or "").strip() in EMOTIONS
                and float(row.get("duration_seconds") or 0.0) < max_duration
                and str(row.get("text") or "").strip()
                and isinstance(row.get("audio"), dict)
                and row.get("audio", {}).get("bytes"))
            / 3600.0,
            6,
        ),
    }
    return selected, summary


def write_dataset(rows: list[dict[str, Any]], summary: dict[str, Any], out_dir: pathlib.Path) -> None:
    data_dir = out_dir / "data"
    data_dir.mkdir(parents=True, exist_ok=True)
    table = pa.Table.from_pylist(rows, schema=ARROW_SCHEMA)
    table = table.replace_schema_metadata(
        {b"huggingface": json.dumps({"info": {"features": FEATURES_META}}, sort_keys=True).encode("utf-8")}
    )
    pq.write_table(table, data_dir / "train-00000.parquet", compression="zstd", use_dictionary=True)
    (out_dir / "dataset_summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    (out_dir / "README.md").write_text(readme(summary))


def readme(summary: dict[str, Any]) -> str:
    emotion_lines = "\n".join(f"- `{emotion}`: {count}" for emotion, count in summary["emotion_counts"].items())
    source_lines = "\n".join(f"- `{source}`: {count}" for source, count in summary["source_counts"].items())
    drop_lines = "\n".join(f"- `{reason}`: {count}" for reason, count in summary["drop_reasons"].items())
    tokens = ", ".join(f"`{token}`" for token in summary["special_tokens"])
    return f"""---
license: other
language:
- en
task_categories:
- automatic-speech-recognition
tags:
- audio
- radio
- sports
- parakeet
- emotion
- emotion-token
- intonation-units
size_categories:
- 1K<n<10K
---

# English Sports Radio Emotion-Token IUs

Two-column dataset of English sports-radio intonation units.

Each row has:

- `audio`: embedded 16 kHz mono FLAC audio for one IU.
- `text`: emotion special token followed by the Parakeet transcript.

The dataset excludes `low_quality`, unlabeled rows, empty transcripts, and clips with duration greater than or equal to {summary["max_duration_seconds_exclusive"]} seconds.

## Contents

- Rows: {summary["row_count"]}
- Audio hours: {summary["audio_hours"]:.3f}
- Columns: `audio`, `text`
- Emotion tokens: {tokens}

## Emotion Counts

{emotion_lines}

## Source Counts

{source_lines}

## Dropped Rows

{drop_lines}
"""


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=pathlib.Path, default=pathlib.Path(DEFAULT_INPUT))
    parser.add_argument("--out-dir", type=pathlib.Path, default=pathlib.Path(DEFAULT_OUT))
    parser.add_argument("--max-duration", type=float, default=20.0)
    args = parser.parse_args()

    rows, summary = build_rows(read_rows(args.input), args.max_duration)
    write_dataset(rows, summary, args.out_dir)
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
