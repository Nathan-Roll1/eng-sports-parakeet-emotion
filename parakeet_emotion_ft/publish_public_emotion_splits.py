#!/usr/bin/env python3
from __future__ import annotations

import argparse
import io
import json
from pathlib import Path
from typing import Any

import soundfile as sf
from datasets import Audio, Dataset, DatasetDict
from huggingface_hub import HfApi
import pyarrow.parquet as pq


SOURCE_REPO = "NathanRoll/eng-sports-radio-psst-iu"
TARGET_REPO = "NathanRoll/eng-sports-radio-psst-iu-emotion-splits"
RICH_SOURCE_PARQUET = "runs/eng_sports_processing/hf_iu_parakeet_labeled_dataset/data/train-00000.parquet"
EMOTIONS = ["joy", "surprise", "sadness", "anger", "disgust", "fear"]
ACCENT_BY_COUNTRY = {
    "AU": "australian",
    "GB": "british",
    "CA": "canadian",
    "US": "american",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-repo", default=SOURCE_REPO)
    parser.add_argument("--target-repo", default=TARGET_REPO)
    parser.add_argument("--rich-source-parquet", default=RICH_SOURCE_PARQUET)
    parser.add_argument("--work-dir", default="artifacts/public_emotion_splits")
    parser.add_argument("--shuffle-seed", type=int, default=397)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def leading_token(text: str) -> str:
    first = (text or "").strip().split(maxsplit=1)[0]
    return first if first.startswith("<|") and first.endswith("|>") else ""


def audio_duration_seconds(audio: dict[str, Any]) -> float:
    if audio.get("bytes") is not None:
        info = sf.info(io.BytesIO(audio["bytes"]))
    elif audio.get("path"):
        info = sf.info(audio["path"])
    else:
        return 0.0
    return float(info.frames) / float(info.samplerate)


def accent_for_country(country_code: str) -> str:
    return ACCENT_BY_COUNTRY.get((country_code or "").upper(), "unknown")


def build_public_rows(rich_source_parquet: str) -> list[dict[str, Any]]:
    rows = pq.read_table(rich_source_parquet).to_pylist()
    out: list[dict[str, Any]] = []
    for row in rows:
        emotion = row.get("emotion")
        text = (row.get("text") or "").strip()
        if not row.get("is_labeled"):
            continue
        if emotion in (None, "", "low_quality", "neutral"):
            continue
        if emotion not in EMOTIONS:
            continue
        if not text:
            continue
        if float(row.get("duration_seconds") or 0.0) >= 20.0:
            continue
        out.append({
            "audio": row["audio"],
            "text": f"<|{emotion}|> {text}",
            "accent": accent_for_country(row.get("country_code") or ""),
            "_emotion": emotion,
            "_source_id": row.get("source_id") or "unknown",
            "_country_code": row.get("country_code") or "unknown",
        })
    return out


def write_card(path: Path, source_repo: str, target_repo: str, summary: dict[str, Any]) -> None:
    split_lines = "\n".join(
        f"- `{name}`: {stats['rows']} rows, {stats['hours']:.3f} audio hours"
        for name, stats in summary["splits"].items()
    )
    accent_lines = "\n".join(
        f"- `{name}`: {count} rows"
        for name, count in summary["accent_counts"].items()
    )
    text = f"""---
license: other
language:
- en
task_categories:
- automatic-speech-recognition
tags:
- audio
- radio
- sports
- emotion
- emotion-token
- intonation-units
- parakeet
- non-neutral
pretty_name: English Sports Radio Non-Neutral Emotion IU Splits
size_categories:
- n<1K
configs:
- config_name: default
  data_files:
{chr(10).join(f'  - split: {name}{chr(10)}    path: data/{name}-*' for name in summary['splits'])}
---

# English Sports Radio Non-Neutral Emotion IU Splits

Public non-neutral subset of [{source_repo}](https://huggingface.co/datasets/{source_repo}).

Each row is one intonation unit with exactly three columns:

- `audio`: embedded 16 kHz mono audio for the IU
- `text`: a leading emotion special token followed by the Parakeet transcript
- `accent`: broadcast-location proxy accent label

Neutral examples were removed. The remaining rows are split by emotion:

{split_lines}

Total rows: {summary['total_rows']}

Total audio hours: {summary['total_hours']:.3f}

Emotion tokens present:
{', '.join(f'`<|{emotion}|>`' for emotion in EMOTIONS)}

Accent labels are inferred from the source station's broadcast country, not
speaker-level accent verification:

- `AU` -> `australian`
- `GB` -> `british`
- `CA` -> `canadian`
- `US` -> `american`

Accent distribution:

{accent_lines}

Each emotion split is deterministically shuffled with seed `{summary['shuffle_seed']}`.

Source revision: `{summary['source_revision']}`
"""
    path.write_text(text)


def main() -> None:
    args = parse_args()
    work_dir = Path(args.work_dir)
    work_dir.mkdir(parents=True, exist_ok=True)

    api = HfApi()
    source_info = api.dataset_info(args.source_repo, token=True)
    public_rows = build_public_rows(args.rich_source_parquet)
    splits = {}
    split_summary: dict[str, dict[str, float | int]] = {}
    for emotion in EMOTIONS:
        rows = [
            {key: row[key] for key in ("audio", "text", "accent")}
            for row in public_rows
            if row["_emotion"] == emotion
        ]
        split = Dataset.from_list(rows)
        split = split.cast_column("audio", Audio(sampling_rate=16000, decode=False))
        split = split.shuffle(seed=args.shuffle_seed)
        splits[emotion] = split
        hours = sum(audio_duration_seconds(row["audio"]) for row in split) / 3600.0
        accent_counts = {
            accent: split["accent"].count(accent)
            for accent in sorted(set(split["accent"]))
        }
        split_summary[emotion] = {"rows": len(split), "hours": hours, "accent_counts": accent_counts}

    rich_rows = pq.read_table(
        args.rich_source_parquet,
        columns=["emotion", "is_labeled", "text", "duration_seconds"],
    ).to_pylist()
    neutral_rows = sum(
        1
        for row in rich_rows
        if row.get("is_labeled")
        and row.get("emotion") == "neutral"
        and (row.get("text") or "").strip()
        and float(row.get("duration_seconds") or 0.0) < 20.0
    )
    total_rows = sum(int(stats["rows"]) for stats in split_summary.values())
    total_hours = sum(float(stats["hours"]) for stats in split_summary.values())
    accent_counts_total: dict[str, int] = {}
    source_counts: dict[str, int] = {}
    country_counts: dict[str, int] = {}
    for row in public_rows:
        accent_counts_total[row["accent"]] = accent_counts_total.get(row["accent"], 0) + 1
        source_counts[row["_source_id"]] = source_counts.get(row["_source_id"], 0) + 1
        country_counts[row["_country_code"]] = country_counts.get(row["_country_code"], 0) + 1
    summary = {
        "source_repo": args.source_repo,
        "target_repo": args.target_repo,
        "source_revision": source_info.sha,
        "rich_source_parquet": args.rich_source_parquet,
        "shuffle_seed": args.shuffle_seed,
        "removed_neutral_rows": neutral_rows,
        "total_rows": total_rows,
        "total_hours": total_hours,
        "accent_counts": dict(sorted(accent_counts_total.items())),
        "country_counts": dict(sorted(country_counts.items())),
        "source_counts": dict(sorted(source_counts.items())),
        "splits": split_summary,
        "columns": ["audio", "text", "accent"],
        "accent_mapping": ACCENT_BY_COUNTRY,
    }

    summary_path = work_dir / "dataset_summary.json"
    readme_path = work_dir / "README.md"
    summary_path.write_text(json.dumps(summary, indent=2))
    write_card(readme_path, args.source_repo, args.target_repo, summary)
    print(json.dumps(summary, indent=2))

    if args.dry_run:
        return

    api.create_repo(args.target_repo, repo_type="dataset", private=False, exist_ok=True)
    DatasetDict(splits).push_to_hub(
        args.target_repo,
        private=False,
        token=True,
        embed_external_files=True,
        commit_message="Create public non-neutral emotion split dataset",
    )
    api.upload_file(
        path_or_fileobj=str(readme_path),
        path_in_repo="README.md",
        repo_id=args.target_repo,
        repo_type="dataset",
        commit_message="Add dataset card",
    )
    api.upload_file(
        path_or_fileobj=str(summary_path),
        path_in_repo="dataset_summary.json",
        repo_id=args.target_repo,
        repo_type="dataset",
        commit_message="Add dataset summary",
    )
    api.update_repo_settings(args.target_repo, repo_type="dataset", private=False)
    print(f"Published https://huggingface.co/datasets/{args.target_repo}")


if __name__ == "__main__":
    main()
