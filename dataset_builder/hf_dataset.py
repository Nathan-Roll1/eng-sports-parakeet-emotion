from __future__ import annotations

import csv
import json
from collections import Counter
from datetime import datetime
from pathlib import Path

from dataset_builder.config import RecordingConfig, RecordingResult, StationInfo

_METADATA_FIELDS = [
    "audio",
    "station_name",
    "station_uuid",
    "country",
    "country_code",
    "language",
    "language_raw",
    "tags",
    "votes",
    "codec_original",
    "bitrate_original",
    "homepage",
    "duration_seconds",
    "file_size_bytes",
    "recording_date",
    "sample_rate",
    "channels",
    "silence_ratio",
    "peak_amplitude",
]


def _audio_relative_path(filepath: str) -> str:
    p = Path(filepath)
    if not p.is_absolute():
        return p.as_posix()
    try:
        return p.resolve().relative_to(Path.cwd().resolve()).as_posix()
    except ValueError:
        return p.as_posix()


def _recording_date_iso(completed_at: str, started_at: str) -> str:
    for s in (completed_at, started_at):
        if not s:
            continue
        try:
            return (
                datetime.fromisoformat(s.replace("Z", "+00:00")).date().isoformat()
            )
        except ValueError:
            if len(s) >= 10:
                return s[:10]
    return ""


def build_metadata_table(results: list[RecordingResult]) -> list[dict]:
    cfg = RecordingConfig()
    rows: list[dict] = []
    for r in results:
        if not r.success or not r.filepath:
            continue
        s: StationInfo = r.station
        rows.append(
            {
                "audio": _audio_relative_path(r.filepath),
                "station_name": s.name,
                "station_uuid": s.stationuuid,
                "country": s.country,
                "country_code": s.countrycode,
                "language": s.language_canonical,
                "language_raw": s.language,
                "tags": s.tags,
                "votes": int(s.votes),
                "codec_original": s.codec,
                "bitrate_original": int(s.bitrate),
                "homepage": s.homepage,
                "duration_seconds": float(r.duration_actual),
                "file_size_bytes": int(r.file_size_bytes),
                "recording_date": _recording_date_iso(r.completed_at, r.started_at),
                "sample_rate": int(cfg.sample_rate),
                "channels": int(cfg.channels),
                "silence_ratio": float(r.silence_ratio),
                "peak_amplitude": float(r.peak_amplitude),
            }
        )
    return rows


def _yaml_frontmatter(
    languages: list[str],
) -> str:
    lines = ["---"]
    lines.append("language:")
    for lang in languages:
        lines.append(f"  - {json.dumps(lang, ensure_ascii=False)}")
    lines.append("task_categories:")
    lines.append("  - automatic-speech-recognition")
    lines.append("tags:")
    for tag in ("audio", "radio", "news", "multilingual"):
        lines.append(f"  - {tag}")
    lines.append("license: cc-by-4.0")
    lines.append("---")
    return "\n".join(lines) + "\n"


def _readme_body(
    languages: list[str],
    countries: list[str],
    total_hours: float,
    num_rows: int,
) -> str:
    lang_bullets = "\n".join(f"- {x}" for x in languages) or "- (none)"
    country_bullets = "\n".join(f"- {x}" for x in countries) or "- (none)"
    return f"""# Multilingual news radio recordings

This dataset contains short-form recordings of internet radio streams labeled as news or talk, captured for speech and audio research.

## Languages covered

{lang_bullets}

## Countries covered

{country_bullets}

## Audio volume

- **Total hours of audio (successful recordings):** {total_hours:.2f}
- **Number of audio rows:** {num_rows}

## Recording methodology

Stations are selected from a public radio station directory API. Each stream is recorded for a fixed target duration using `ffmpeg`, with audio normalized to a consistent sample rate and channel layout for downstream use. Failed or empty captures are excluded from the released metadata table. Stream content reflects live broadcast schedules and may include music, ads, or non-news segments depending on the station.

## License

Dataset metadata and this dataset card are licensed under [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/). Individual broadcast streams may be subject to separate terms from their rights holders; users are responsible for complying with applicable law and station policies when using the audio.
"""


def create_dataset(
    results: list[RecordingResult], output_dir: str = "hf_dataset"
) -> str:
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    rows = build_metadata_table(results)
    csv_path = out / "metadata.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=_METADATA_FIELDS)
        writer.writeheader()
        writer.writerows(rows)

    langs_sorted = sorted({r["language"] for r in rows if r.get("language")})
    countries_sorted = sorted({r["country"] for r in rows if r.get("country")})
    total_seconds = sum(r["duration_seconds"] for r in rows)
    total_hours = total_seconds / 3600.0

    dataset_info = {
        "description": "Multilingual news-oriented radio stream recordings with station metadata.",
        "format": "csv+audio",
        "metadata_file": "metadata.csv",
        "audio_column": "audio",
        "num_rows": len(rows),
        "splits": {"train": {"num_rows": len(rows), "metadata_file": "metadata.csv"}},
        "features": {
            "audio": {"dtype": "audio", "decode": False},
            "station_name": "string",
            "station_uuid": "string",
            "country": "string",
            "country_code": "string",
            "language": "string",
            "language_raw": "string",
            "tags": "string",
            "votes": "int64",
            "codec_original": "string",
            "bitrate_original": "int64",
            "homepage": "string",
            "duration_seconds": "float64",
            "file_size_bytes": "int64",
            "recording_date": "string",
            "sample_rate": "int64",
            "channels": "int64",
            "silence_ratio": "float64",
            "peak_amplitude": "float64",
        },
        "languages_covered": langs_sorted,
        "countries_covered": countries_sorted,
        "total_hours_audio": total_hours,
    }
    (out / "dataset_dict.json").write_text(
        json.dumps(dataset_info, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    readme = _yaml_frontmatter(langs_sorted) + "\n" + _readme_body(
        langs_sorted, countries_sorted, total_hours, len(rows)
    )
    (out / "README.md").write_text(readme, encoding="utf-8")

    return str(out.resolve())


def create_hf_dataset_object(metadata_csv: str, audio_dir: str):
    try:
        from datasets import Audio, Dataset, Features, Value
    except ImportError:
        print(
            "The Hugging Face `datasets` package is required. "
            "Install it with: pip install datasets"
        )
        raise

    meta_path = Path(metadata_csv)
    base = Path(audio_dir)
    with meta_path.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        fieldnames = reader.fieldnames or []
        col_data: dict[str, list] = {name: [] for name in fieldnames}
        for row in reader:
            for name in fieldnames:
                col_data[name].append(row.get(name, ""))

    if "audio" not in col_data:
        raise ValueError("metadata CSV must contain an 'audio' column")

    audio_resolved = [str(base / Path(p)) for p in col_data["audio"]]
    col_data["audio"] = audio_resolved

    def _int_col(name: str) -> None:
        if name in col_data:
            col_data[name] = [int(v) if v != "" else 0 for v in col_data[name]]

    def _float_col(name: str) -> None:
        if name in col_data:
            col_data[name] = [float(v) if v != "" else 0.0 for v in col_data[name]]

    _int_col("votes")
    _int_col("bitrate_original")
    _int_col("file_size_bytes")
    _int_col("sample_rate")
    _int_col("channels")
    _float_col("duration_seconds")
    _float_col("silence_ratio")
    _float_col("peak_amplitude")

    cfg = RecordingConfig()
    features = Features(
        {
            "audio": Value("string"),
            "station_name": Value("string"),
            "station_uuid": Value("string"),
            "country": Value("string"),
            "country_code": Value("string"),
            "language": Value("string"),
            "language_raw": Value("string"),
            "tags": Value("string"),
            "votes": Value("int64"),
            "codec_original": Value("string"),
            "bitrate_original": Value("int64"),
            "homepage": Value("string"),
            "duration_seconds": Value("float64"),
            "file_size_bytes": Value("int64"),
            "recording_date": Value("string"),
            "sample_rate": Value("int64"),
            "channels": Value("int64"),
            "silence_ratio": Value("float64"),
            "peak_amplitude": Value("float64"),
        }
    )

    ds = Dataset.from_dict(col_data, features=features)
    ds = ds.cast_column(
        "audio", Audio(sampling_rate=cfg.sample_rate, decode=False)
    )
    return ds


def export_stats(results: list[RecordingResult]) -> dict:
    total = len(results)
    successful = [r for r in results if r.success]
    failed = total - len(successful)
    ok = [r for r in successful if r.filepath]
    total_seconds = sum(r.duration_actual for r in ok)
    total_hours = total_seconds / 3600.0
    lang_counts = Counter(r.station.language_canonical for r in ok)
    country_counts = Counter(r.station.country for r in ok)
    n_ok = len(ok)
    avg_dur = total_seconds / n_ok if n_ok else 0.0
    total_bytes = sum(r.file_size_bytes for r in ok)
    avg_size = total_bytes / n_ok if n_ok else 0.0
    return {
        "total_recordings": total,
        "successful": len(successful),
        "failed": failed,
        "total_hours_audio": total_hours,
        "total_hours": total_hours,
        "languages": dict(sorted(lang_counts.items(), key=lambda x: (-x[1], x[0]))),
        "countries": dict(
            sorted(country_counts.items(), key=lambda x: (-x[1], x[0]))
        ),
        "average_duration_seconds": avg_dur,
        "average_file_size_bytes": avg_size,
        "storage_used_bytes": total_bytes,
    }
