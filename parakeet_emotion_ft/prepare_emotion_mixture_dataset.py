#!/usr/bin/env python3
from __future__ import annotations

import argparse
import io
import json
import random
import re
import unicodedata
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable


EMOTION_TOKENS = {
    "neutral": "<|neutral|>",
    "low_quality": "<|low_quality|>",
    "joy": "<|joy|>",
    "surprise": "<|surprise|>",
    "sadness": "<|sadness|>",
    "anger": "<|anger|>",
    "disgust": "<|disgust|>",
    "fear": "<|fear|>",
}
TOKEN_PATTERN = re.compile(r"^\s*<\|(?:neutral|low_quality|joy|surprise|sadness|anger|disgust|fear)\|>\s*")

CREMA_SENTENCES = {
    "IEO": "It's eleven o'clock.",
    "TIE": "That is exactly what happened.",
    "IOM": "I'm on my way to the meeting.",
    "IWW": "I wonder what this is about.",
    "TAI": "The airplane is almost full.",
    "MTI": "Maybe tomorrow it will be cold.",
    "IWL": "I would like a new alarm clock.",
    "ITH": "I think I have a doctor's appointment.",
    "DFA": "Don't forget a jacket.",
    "ITS": "I think I've seen this before.",
    "TSI": "The surface is slick.",
    "WSI": "We'll stop in a couple of minutes.",
}

CREMA_EMOTIONS = {
    "ANG": "anger",
    "DIS": "disgust",
    "FEA": "fear",
    "HAP": "joy",
    "NEU": "neutral",
    "SAD": "sadness",
}

RAVDESS_STATEMENTS = {
    "01": "Kids are talking by the door.",
    "02": "Dogs are sitting by the door.",
    "kids are talking by the door": "Kids are talking by the door.",
    "dogs are sitting by the door": "Dogs are sitting by the door.",
}

EMOTION_MAP = {
    "neutral": "neutral",
    "low_quality": "low_quality",
    "bad_quality": "low_quality",
    "poor_quality": "low_quality",
    "happy": "joy",
    "happiness": "joy",
    "joy": "joy",
    "excited": "joy",
    "pleasant_surprise": "surprise",
    "surprise": "surprise",
    "surprised": "surprise",
    "sad": "sadness",
    "sadness": "sadness",
    "angry": "anger",
    "anger": "anger",
    "frustrated": "anger",
    "fearful": "fear",
    "fear": "fear",
    "disgust": "disgust",
    "disgusted": "disgust",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--sports-parquet",
        default="runs/eng_sports_processing/hf_iu_parakeet_labeled_dataset/data/train-00000.parquet",
    )
    parser.add_argument("--output-dir", default="runs/parakeet_emotion_mixture/full_mixture")
    parser.add_argument("--seed", type=int, default=917)
    parser.add_argument("--min-duration-seconds", type=float, default=0.35)
    parser.add_argument("--max-duration-seconds", type=float, default=20.0)
    parser.add_argument("--skip-sources", default="")
    parser.add_argument("--max-rows-per-external-source", type=int, default=0)
    parser.add_argument("--rows-per-shard", type=int, default=5000)
    parser.add_argument("--token", default=True, action=argparse.BooleanOptionalAction)
    return parser.parse_args()


def normalize_label(label: Any) -> str | None:
    if label is None:
        return None
    key = str(label).strip().lower().replace("-", "_").replace(" ", "_")
    return EMOTION_MAP.get(key)


def emotion_token(label: Any) -> str | None:
    normalized = normalize_label(label)
    return EMOTION_TOKENS.get(normalized or "")


def strip_leading_token(text: str) -> str:
    return TOKEN_PATTERN.sub("", text or "").strip()


def clean_text(text: Any) -> str:
    text = str(text or "")
    text = (
        text.replace("\x91", "'")
        .replace("\x92", "'")
        .replace("\x93", '"')
        .replace("\x94", '"')
        .replace("\x96", "-")
        .replace("\x97", "-")
    )
    text = unicodedata.normalize("NFKC", text)
    text = "".join(ch for ch in text if ch.isprintable() or ch.isspace())
    return re.sub(r"\s+", " ", text).strip()


def audio_bytes(audio: dict[str, Any]) -> bytes | None:
    value = audio.get("bytes")
    if value:
        return bytes(value)
    path = audio.get("path")
    if path and Path(path).exists():
        return Path(path).read_bytes()
    return None


def duration_seconds_from_audio(audio: dict[str, Any]) -> float:
    import soundfile as sf

    value = audio.get("array")
    rate = int(audio.get("sampling_rate") or 16000)
    if value is not None:
        return float(len(value)) / float(rate)
    payload = audio_bytes(audio)
    if payload:
        info = sf.info(io.BytesIO(payload))
        return float(info.frames) / float(info.samplerate)
    return 0.0


def row_duration_seconds(row: dict[str, Any]) -> float:
    for key in ["duration_seconds", "duration", "Duration"]:
        value = row.get(key)
        if value is not None:
            try:
                return float(value)
            except (TypeError, ValueError):
                pass
    return duration_seconds_from_audio(row["audio"])


def make_audio(audio: dict[str, Any], source: str, index: int) -> dict[str, Any] | None:
    payload = audio_bytes(audio)
    if not payload:
        return None
    raw_path = str(audio.get("path") or "")
    suffix = Path(raw_path).suffix or ".wav"
    return {"bytes": payload, "path": f"{source}/{index:07d}{suffix}"}


def accept_duration(duration: float, args: argparse.Namespace) -> bool:
    return args.min_duration_seconds <= duration < args.max_duration_seconds


def add_row(
    rows: list[dict[str, Any]],
    stats: dict[str, Any],
    *,
    source: str,
    audio: dict[str, Any],
    text: str,
    label: Any,
    duration: float,
    source_index: int,
    args: argparse.Namespace,
) -> None:
    token = emotion_token(label)
    if token is None:
        stats["skipped"][source]["unknown_label"] += 1
        return
    cleaned = strip_leading_token(clean_text(text))
    if not cleaned:
        stats["skipped"][source]["empty_text"] += 1
        return
    if not accept_duration(duration, args):
        stats["skipped"][source]["duration"] += 1
        return
    embedded = make_audio(audio, source, source_index)
    if embedded is None:
        stats["skipped"][source]["missing_audio_bytes"] += 1
        return
    rows.append({"audio": embedded, "text": f"{token} {cleaned}"})
    normalized = normalize_label(label) or "unknown"
    stats["included_by_source"][source] += 1
    stats["included_by_label"][normalized] += 1
    stats["included_by_source_label"][source][normalized] += 1


def load_hf_dataset(repo: str, *, token: bool):
    from datasets import Audio, load_dataset

    ds = load_dataset(repo, token=token)
    out = {}
    for split, split_ds in ds.items():
        if "audio" in split_ds.column_names:
            split_ds = split_ds.cast_column("audio", Audio(decode=False))
        out[split] = split_ds
    return out


def iter_splits(dataset_dict: dict[str, Any]) -> Iterable[tuple[str, dict[str, Any]]]:
    for split, dataset in dataset_dict.items():
        for row in dataset:
            yield split, row


def add_sports(rows: list[dict[str, Any]], stats: dict[str, Any], args: argparse.Namespace) -> None:
    import pyarrow.parquet as pq

    table = pq.read_table(Path(args.sports_parquet).expanduser())
    for idx, row in enumerate(table.to_pylist()):
        audio = row.get("audio")
        if not isinstance(audio, dict):
            stats["skipped"]["sports_radio"]["missing_audio"] += 1
            continue
        add_row(
            rows,
            stats,
            source="sports_radio",
            audio=audio,
            text=row.get("text") or "",
            label=row.get("emotion"),
            duration=row_duration_seconds(row),
            source_index=idx,
            args=args,
        )


def add_iemocap(rows: list[dict[str, Any]], stats: dict[str, Any], args: argparse.Namespace) -> None:
    ds = load_hf_dataset("mteb/iemocap", token=args.token)
    for idx, (_split, row) in enumerate(iter_splits(ds)):
        add_row(
            rows,
            stats,
            source="iemocap",
            audio=row["audio"],
            text=row.get("transcription") or "",
            label=row.get("major_emotion"),
            duration=row_duration_seconds(row),
            source_index=idx,
            args=args,
        )


def add_cremad(rows: list[dict[str, Any]], stats: dict[str, Any], args: argparse.Namespace) -> None:
    ds = load_hf_dataset("confit/cremad-parquet", token=args.token)
    for idx, (_split, row) in enumerate(iter_splits(ds)):
        file_name = Path(str(row.get("file") or "")).name
        pieces = file_name.split("_")
        sentence_code = pieces[1] if len(pieces) > 1 else ""
        text = CREMA_SENTENCES.get(sentence_code, "")
        label = row.get("emotion")
        if not label and len(pieces) > 2:
            label = CREMA_EMOTIONS.get(pieces[2])
        add_row(
            rows,
            stats,
            source="cremad",
            audio=row["audio"],
            text=text,
            label=label,
            duration=row_duration_seconds(row),
            source_index=idx,
            args=args,
        )


def tess_text(row: dict[str, Any]) -> str:
    path = Path(str(row.get("WavPath") or row.get("path") or ""))
    stem = path.stem
    parts = stem.split("_")
    if len(parts) >= 3:
        word = parts[1].replace("-", " ")
        return f"Say the word {word}."
    return ""


def add_tess(rows: list[dict[str, Any]], stats: dict[str, Any], args: argparse.Namespace) -> None:
    ds = load_hf_dataset("TwinkStart/TESS", token=args.token)
    for idx, (_split, row) in enumerate(iter_splits(ds)):
        add_row(
            rows,
            stats,
            source="tess",
            audio=row["audio"],
            text=tess_text(row),
            label=row.get("label"),
            duration=row_duration_seconds(row),
            source_index=idx,
            args=args,
        )


def add_meld(rows: list[dict[str, Any]], stats: dict[str, Any], args: argparse.Namespace) -> None:
    ds = load_hf_dataset("TwinkStart/MELD", token=args.token)
    for idx, (_split, row) in enumerate(iter_splits(ds)):
        add_row(
            rows,
            stats,
            source="meld",
            audio=row["audio"],
            text=row.get("Utterance") or "",
            label=row.get("Emotion"),
            duration=row_duration_seconds(row),
            source_index=idx,
            args=args,
        )


def ravdess_text(row: dict[str, Any]) -> str:
    statement = clean_text(row.get("statement"))
    mapped = RAVDESS_STATEMENTS.get(statement) or RAVDESS_STATEMENTS.get(statement.lower())
    return mapped or statement


def add_ravdess(rows: list[dict[str, Any]], stats: dict[str, Any], args: argparse.Namespace) -> None:
    ds = load_hf_dataset("xbgoose/ravdess", token=args.token)
    for idx, (_split, row) in enumerate(iter_splits(ds)):
        channel = clean_text(row.get("vocal_channel")).lower()
        if channel and channel != "speech":
            stats["skipped"]["ravdess"]["non_speech"] += 1
            continue
        add_row(
            rows,
            stats,
            source="ravdess",
            audio=row["audio"],
            text=ravdess_text(row),
            label=row.get("emotion"),
            duration=row_duration_seconds(row),
            source_index=idx,
            args=args,
        )


SOURCE_LOADERS = {
    "sports_radio": add_sports,
    "iemocap": add_iemocap,
    "cremad": add_cremad,
    "tess": add_tess,
    "meld": add_meld,
    "ravdess": add_ravdess,
}


def cap_external_sources(rows: list[dict[str, Any]], max_rows: int, seed: int) -> list[dict[str, Any]]:
    if max_rows <= 0:
        return rows
    rng = random.Random(seed + 31)
    by_source: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        source = str((row["audio"].get("path") or "").split("/", 1)[0])
        by_source[source].append(row)
    capped: list[dict[str, Any]] = []
    for source, source_rows in by_source.items():
        if source == "sports_radio" or len(source_rows) <= max_rows:
            capped.extend(source_rows)
            continue
        rng.shuffle(source_rows)
        capped.extend(source_rows[:max_rows])
    return capped


def write_dataset(rows: list[dict[str, Any]], output_dir: Path, rows_per_shard: int) -> list[Path]:
    import pyarrow as pa
    import pyarrow.parquet as pq

    data_dir = output_dir / "data"
    data_dir.mkdir(parents=True, exist_ok=True)
    schema = pa.schema([
        pa.field("audio", pa.struct([
            pa.field("bytes", pa.binary()),
            pa.field("path", pa.string()),
        ])),
        pa.field("text", pa.string()),
    ])
    shard_size = rows_per_shard if rows_per_shard > 0 else len(rows)
    shard_count = max(1, (len(rows) + shard_size - 1) // shard_size)
    parquet_paths: list[Path] = []
    for shard_idx in range(shard_count):
        start = shard_idx * shard_size
        stop = min(len(rows), start + shard_size)
        shard_rows = rows[start:stop]
        parquet_path = data_dir / f"train-{shard_idx:05d}-of-{shard_count:05d}.parquet"
        table = pa.Table.from_pylist(shard_rows, schema=schema)
        pq.write_table(table, parquet_path)
        parquet_paths.append(parquet_path)
    return parquet_paths


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, Any]] = []
    stats: dict[str, Any] = {
        "included_by_source": Counter(),
        "included_by_label": Counter(),
        "included_by_source_label": defaultdict(Counter),
        "skipped": defaultdict(Counter),
        "source_errors": {},
    }
    skip_sources = {x.strip() for x in args.skip_sources.split(",") if x.strip()}

    for source, loader in SOURCE_LOADERS.items():
        if source in skip_sources:
            stats["source_errors"][source] = "skipped_by_request"
            continue
        before = len(rows)
        print(f"loading {source}...", flush=True)
        try:
            loader(rows, stats, args)
        except Exception as exc:
            stats["source_errors"][source] = f"{type(exc).__name__}: {exc}"
            print(f"failed {source}: {type(exc).__name__}: {exc}", flush=True)
        print(f"loaded {source}: +{len(rows) - before} rows", flush=True)

    before_cap = len(rows)
    rows = cap_external_sources(rows, args.max_rows_per_external_source, args.seed)
    rng = random.Random(args.seed)
    rng.shuffle(rows)

    parquet_paths = write_dataset(rows, output_dir, args.rows_per_shard)
    summary = {
        "seed": args.seed,
        "sports_parquet": str(Path(args.sports_parquet).expanduser()),
        "output_data_dir": str(output_dir / "data"),
        "output_parquet_files": [str(path) for path in parquet_paths],
        "min_duration_seconds": args.min_duration_seconds,
        "max_duration_seconds": args.max_duration_seconds,
        "rows_per_shard": args.rows_per_shard,
        "rows_before_cap": before_cap,
        "rows_written": len(rows),
        "max_rows_per_external_source": args.max_rows_per_external_source,
        "included_by_source": dict(sorted(stats["included_by_source"].items())),
        "included_by_label": dict(sorted(stats["included_by_label"].items())),
        "included_by_source_label": {
            source: dict(sorted(counts.items()))
            for source, counts in sorted(stats["included_by_source_label"].items())
        },
        "skipped": {
            source: dict(sorted(counts.items()))
            for source, counts in sorted(stats["skipped"].items())
        },
        "source_errors": dict(sorted(stats["source_errors"].items())),
    }
    (output_dir / "dataset_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    (output_dir / "README.md").write_text(
        "# Parakeet Emotion Mixture Dataset\n\n"
        "Two-column local training mixture for Parakeet emotion-token ASR fine-tuning.\n"
        "Each row is one audio clip and one text target beginning with an emotion token.\n\n"
        f"Rows written: {len(rows)}\n\n"
        "Sources: sports radio IU labels, IEMOCAP, CREMA-D, TESS, MELD, and RAVDESS.\n"
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
