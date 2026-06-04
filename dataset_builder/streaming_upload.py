"""
Streaming dataset builder: records stations and uploads individual
parquet shards to HuggingFace Hub as recordings complete.

Each shard is uploaded once and never re-uploaded.  Local audio files
are deleted after each successful shard upload to free disk and memory.

Usage:
    python -m dataset_builder.streaming_upload \
        --repo NathanRoll/my-dataset \
        --token "$HF_TOKEN" \
        --duration 600 \
        --shard-every 5
"""

from __future__ import annotations

import gc
import os
import subprocess
import tempfile
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import fields as dc_fields
from pathlib import Path

from dataset_builder.config import (
    FFMPEG_BIN,
    RecordingConfig,
    RecordingResult,
    StationInfo,
    normalize_language,
)
from dataset_builder.recorder import record_station

MUSIC_TAGS = frozenset({
    "music", "pop", "rock", "hits", "top 40", "dance", "entertainment",
    "oldies", "classic hits", "adult contemporary", "variety", "jazz",
    "latin music", "local music", "folk music", "classical",
    "classical music", "adult", "contemporany", "urban",
})
NEWS_TAGS = frozenset({
    "news", "talk", "information", "news talk", "talk show",
    "political talk", "local news", "business news", "public radio",
    "all news", "talk news", "nachrichten", "nyheter", "noticias",
    "actualités", "haber", "haberler",
})
_NEWS_NAME_KW = ("news", "noticias", "info", "nachrichten", "nyheter", "haber", "actualit")

_LANG_COLLAPSE = {
    "128 Brazilian Portuguese": "Portuguese", "Brazilian Portuguese": "Portuguese",
    "Portugues Brasil": "Portuguese", "Português  Brasil": "Portuguese",
    "Português (Br)": "Portuguese", "Português (Brasil)": "Portuguese",
    "Portuguese Brazil": "Portuguese",
    "Castellano. Español": "Spanish", "Español Argentina": "Spanish",
    "Español Chile": "Spanish", "Español Mexico": "Spanish",
    "Español Internacional": "Spanish", "Esapa": "Spanish", "Espanish": "Spanish",
    "English Uk": "English", "English/": "English", "Englsih": "English",
    "Arabic.": "Arabic", "Franch": "French", "Romania": "Romanian",
    "Nederland": "Dutch", "Hun": "Hungarian", "Finland": "Finnish",
    "Kurdish.": "Kurdish", "Язык: Русский": "Russian", "Swiss German": "German",
    "Filipino,Ilocano": "Filipino", "Filipino,Kapampangan": "Filipino",
    "Filipino,Kampampangan": "Filipino", "Bicolano,Filipino": "Filipino",
    "Capiznon,Filipino": "Filipino", "Various Filipino Languages": "Filipino",
    "Tagalog": "Filipino", "Bearnese,Gascon,Occitan": "Occitan",
    "Kikuyu Swahili": "Swahili", "Oromo Amharic Somali": "Amharic",
    "Portugues Brasil,Portuguese Brazil": "Portuguese",
    "Brazilian Portuguese,Portugues Do Brasil": "Portuguese",
    "Brazilian Portuguese,Português  Brasil": "Portuguese",
    "Brazilian Portuguese,Português  Brasil,Português (Brasil)": "Portuguese",
    "Castellano. Español,Español Internacional": "Spanish",
    "Cebuano,Filipino": "Filipino",
}


def _news_score(s: dict) -> int:
    tags = {t.strip().lower() for t in (s.get("tags", "") or "").split(",") if t.strip()}
    name_lower = s.get("name", "").lower()
    return (
        len(tags & NEWS_TAGS)
        + sum(2 for w in _NEWS_NAME_KW if w in name_lower)
        - len(tags & MUSIC_TAGS) * 3
    )


def discover_news_stations(min_score: int = 1) -> list[StationInfo]:
    from pyradios import RadioBrowser

    rb = RadioBrowser()
    results = rb.search(tag="news", limit=100000, hidebroken=True)
    si_fields = {f.name for f in dc_fields(StationInfo)}

    best: dict[str, tuple[dict, int, int]] = {}
    for s in results:
        lang_raw = (s.get("language") or "").strip()
        if not lang_raw:
            continue
        canonical = _LANG_COLLAPSE.get(normalize_language(lang_raw), normalize_language(lang_raw))
        url = (s.get("url_resolved") or "").strip()
        if not url or not canonical:
            continue
        score = _news_score(s)
        if score < min_score:
            continue
        votes = s.get("votes", 0)
        prev = best.get(canonical)
        if not prev or score > prev[1] or (score == prev[1] and votes > prev[2]):
            best[canonical] = (s, score, votes)

    out = []
    for lang in sorted(best):
        s, _, _ = best[lang]
        kwargs = {k: v for k, v in s.items() if k in si_fields}
        kwargs["language_canonical"] = lang
        out.append(StationInfo(**kwargs))
    return out


def _to_mp3(flac_path: str, mp3_path: str) -> bool:
    return (
        subprocess.run(
            [FFMPEG_BIN, "-y", "-i", flac_path, "-ar", "16000", "-ac", "1", "-b:a", "64k", mp3_path],
            capture_output=True,
        ).returncode == 0
        and Path(mp3_path).is_file()
    )


def _row_from_result(result: RecordingResult, mp3_path: str) -> dict:
    s = result.station
    return {
        "audio": mp3_path,
        "station_name": s.name.strip(),
        "station_uuid": s.stationuuid,
        "country": s.country,
        "country_code": s.countrycode,
        "language": s.language_canonical,
        "language_raw": s.language,
        "tags": s.tags,
        "votes": s.votes,
        "codec_original": s.codec,
        "bitrate_original": s.bitrate,
        "homepage": s.homepage,
        "duration_seconds": round(result.duration_actual, 1),
        "file_size_bytes": Path(mp3_path).stat().st_size,
        "recording_date": result.completed_at[:10],
        "sample_rate": 16000,
        "channels": 1,
    }


def _metadata_only(row: dict) -> dict:
    """Strip the audio path — keep only lightweight metadata for final stats."""
    return {k: v for k, v in row.items() if k != "audio"}


def _build_and_upload_shard(
    rows: list[dict],
    repo_id: str,
    token: str,
    shard_idx: int,
) -> bool:
    """
    Build a parquet file from `rows` (which contain "audio" pointing to
    local MP3 files), upload it as a single file to the Hub, then return
    True on success.

    Audio bytes are read from disk and embedded directly into the parquet
    so local files can safely be deleted afterward.
    """
    import pyarrow as pa
    import pyarrow.parquet as pq_writer
    from huggingface_hub import HfApi

    audio_structs = []
    meta_rows = {k: [] for k in rows[0] if k != "audio"}

    for row in rows:
        mp3_path = row["audio"]
        with open(mp3_path, "rb") as f:
            audio_bytes = f.read()
        audio_structs.append({
            "bytes": audio_bytes,
            "path": os.path.basename(mp3_path),
        })
        for k in meta_rows:
            meta_rows[k].append(row[k])

    audio_type = pa.struct([("bytes", pa.binary()), ("path", pa.string())])
    columns = {"audio": pa.array(audio_structs, type=audio_type)}
    for k, vals in meta_rows.items():
        columns[k] = pa.array(vals)

    table = pa.table(columns)

    # Attach HF feature metadata so the datasets library recognizes the Audio column
    import json
    hf_meta = {
        "info": {
            "features": {
                "audio": {"_type": "Audio", "sampling_rate": 16000},
                "station_name": {"dtype": "string", "_type": "Value"},
                "station_uuid": {"dtype": "string", "_type": "Value"},
                "country": {"dtype": "string", "_type": "Value"},
                "country_code": {"dtype": "string", "_type": "Value"},
                "language": {"dtype": "string", "_type": "Value"},
                "language_raw": {"dtype": "string", "_type": "Value"},
                "tags": {"dtype": "string", "_type": "Value"},
                "votes": {"dtype": "int64", "_type": "Value"},
                "codec_original": {"dtype": "string", "_type": "Value"},
                "bitrate_original": {"dtype": "int64", "_type": "Value"},
                "homepage": {"dtype": "string", "_type": "Value"},
                "duration_seconds": {"dtype": "float64", "_type": "Value"},
                "file_size_bytes": {"dtype": "int64", "_type": "Value"},
                "recording_date": {"dtype": "string", "_type": "Value"},
                "sample_rate": {"dtype": "int64", "_type": "Value"},
                "channels": {"dtype": "int64", "_type": "Value"},
            }
        }
    }
    existing = table.schema.metadata or {}
    table = table.replace_schema_metadata({
        **existing,
        b"huggingface": json.dumps(hf_meta).encode(),
    })

    with tempfile.TemporaryDirectory() as tmp:
        pq_path = os.path.join(tmp, "shard.parquet")
        pq_writer.write_table(table, pq_path)

        remote_name = f"data/train-{shard_idx:05d}.parquet"
        HfApi(token=token).upload_file(
            path_or_fileobj=pq_path,
            path_in_repo=remote_name,
            repo_id=repo_id,
            repo_type="dataset",
            commit_message=f"Add shard {shard_idx} ({len(rows)} recordings)",
        )

    del table, columns, audio_structs
    gc.collect()
    return True


def _cleanup_local_files(rows: list[dict], flac_dir: str):
    """Delete local MP3 and corresponding FLAC files after a shard is uploaded."""
    for row in rows:
        mp3 = row.get("audio", "")
        if mp3 and os.path.isfile(mp3):
            os.remove(mp3)
        uuid_prefix = row.get("station_uuid", "")
        if uuid_prefix and flac_dir:
            for flac in Path(flac_dir).rglob(f"*{uuid_prefix[:8]}*"):
                try:
                    flac.unlink()
                except OSError:
                    pass


def _upload_readme(repo_id: str, token: str, all_meta: list[dict], shard_count: int):
    from huggingface_hub import HfApi

    langs = sorted(set(r["language"] for r in all_meta))
    countries = sorted(set(r["country_code"] for r in all_meta if r["country_code"]))
    total_min = sum(r["duration_seconds"] for r in all_meta) / 60.0
    total_mb = sum(r["file_size_bytes"] for r in all_meta) / (1024 * 1024)

    readme = f"""---
language:
- multilingual
task_categories:
- automatic-speech-recognition
tags:
- audio
- radio
- news
- multilingual
- speech
license: cc-by-4.0
size_categories:
- n<1K
---

# Global News Radio Dataset

Multilingual **news** radio recordings from {len(langs)} languages across {len(countries)} countries.

| | |
|---|---|
| **Recordings** | {len(all_meta)} |
| **Total audio** | {total_min:.0f} min ({total_min / 60:.1f} h) |
| **Format** | MP3 16kHz mono 64kbps |
| **Parquet shards** | {shard_count} |
| **Languages** | {len(langs)} |
| **Countries** | {len(countries)} |
| **Size** | {total_mb:.0f} MB |

## Languages

{', '.join(langs)}

## Usage

```python
from datasets import load_dataset
ds = load_dataset("{repo_id}")
sample = ds["train"][0]
print(sample["station_name"], sample["language"])
```

## Source

[Radio Browser API](https://www.radio-browser.info/) via pyradios.
Built with streaming append-only shard uploads — each parquet shard
is uploaded once and never re-uploaded.

## License

Metadata: CC-BY-4.0. Audio from public radio broadcasts.
"""
    HfApi(token=token).upload_file(
        path_or_fileobj=readme.encode(),
        path_in_repo="README.md",
        repo_id=repo_id,
        repo_type="dataset",
    )


def stream_record_and_upload(
    repo_id: str,
    token: str,
    duration_seconds: int = 600,
    max_concurrent: int = 50,
    shard_every: int = 5,
    work_dir: str = "/tmp/radio_stream_dataset",
    private: bool = False,
):
    """
    Record one station per language and upload to HuggingFace Hub as
    append-only parquet shards.

    Each shard is uploaded exactly once.  After upload, local audio
    files are deleted and the in-memory audio data is freed.  Only
    lightweight metadata is kept in memory for the final README.

    Flow:
      1. All stations start recording in parallel
      2. As each finishes → convert to MP3 → buffer the row
      3. Every `shard_every` completions → build parquet from buffer →
         upload as data/train-NNNNN.parquet → delete local files → clear buffer
      4. At end → flush remaining buffer → update README with totals
    """
    from huggingface_hub import HfApi

    api = HfApi(token=token)
    api.create_repo(repo_id=repo_id, repo_type="dataset", private=private, exist_ok=True)

    print("Discovering verified news stations...")
    stations = discover_news_stations()
    total = len(stations)
    print(f"Found {total} stations\n")

    flac_dir = f"{work_dir}/flac"
    mp3_dir = f"{work_dir}/mp3"
    os.makedirs(flac_dir, exist_ok=True)
    os.makedirs(mp3_dir, exist_ok=True)

    cfg = RecordingConfig(
        duration_seconds=duration_seconds,
        output_dir=flac_dir,
        ffmpeg_timeout=duration_seconds + 120,
        max_concurrent=max_concurrent,
        sample_rate=16000,
        channels=1,
    )

    buffer: list[dict] = []
    all_meta: list[dict] = []  # lightweight — no audio paths
    lock = threading.Lock()
    shard_idx = 0
    t0 = time.time()

    def flush_buffer(force: bool = False):
        """Build a parquet shard from the buffer and upload it."""
        nonlocal shard_idx

        with lock:
            if not buffer or (not force and len(buffer) < shard_every):
                return
            batch = list(buffer)
            buffer.clear()
            current_shard = shard_idx
            shard_idx += 1

        elapsed = time.time() - t0
        n_total = len(all_meta) + len(batch)
        print(f"\n  >>> Shard {current_shard}: building parquet from {len(batch)} recordings ({n_total} total, {elapsed:.0f}s)...")

        try:
            _build_and_upload_shard(batch, repo_id, token, current_shard)
        except Exception as e:
            print(f"  >>> Shard {current_shard} FAILED: {e}")
            print(f"  >>> Keeping local files for retry. Re-buffering rows.\n")
            with lock:
                buffer.extend(batch)
                shard_idx -= 1
            return

        with lock:
            all_meta.extend(_metadata_only(r) for r in batch)

        _cleanup_local_files(batch, flac_dir)
        del batch
        gc.collect()

        print(f"  >>> Shard {current_shard} uploaded and local files cleaned. Hub has {n_total} recordings.\n")

    def handle_result(station: StationInfo, result: RecordingResult, idx: int):
        if not result.success or not result.filepath:
            err = (result.error or "")[:60]
            print(f"  [{idx}/{total}] FAIL {station.language_canonical:<18} {station.name[:38]:<38} {err}")
            return

        lang_safe = station.language_canonical.replace(" ", "_").replace("/", "-") or "unknown"
        cc = station.countrycode or "XX"
        mp3_path = f"{mp3_dir}/{cc}_{lang_safe}_{station.stationuuid[:8]}.mp3"

        if not _to_mp3(result.filepath, mp3_path):
            print(f"  [{idx}/{total}] FAIL {station.language_canonical:<18} {station.name[:38]:<38} mp3 convert failed")
            return

        row = _row_from_result(result, mp3_path)
        mb = row["file_size_bytes"] / (1024 * 1024)
        print(f"  [{idx}/{total}] OK   {station.language_canonical:<18} {station.name[:38]:<38} {result.duration_actual:.0f}s {mb:.1f}MB")

        with lock:
            buffer.append(row)

        flush_buffer()

    completed = 0

    with ThreadPoolExecutor(max_workers=max_concurrent) as pool:
        futures = {pool.submit(record_station, s, cfg): s for s in stations}
        try:
            for fut in as_completed(futures):
                station = futures[fut]
                result = fut.result()
                completed += 1
                handle_result(station, result, completed)
        except KeyboardInterrupt:
            print("\nInterrupted!")

    flush_buffer(force=True)

    with lock:
        final_meta = list(all_meta)

    if not final_meta:
        print("No successful recordings.")
        return

    elapsed = time.time() - t0
    total_min = sum(r["duration_seconds"] for r in final_meta) / 60.0
    total_mb = sum(r["file_size_bytes"] for r in final_meta) / (1024 * 1024)
    langs = sorted(set(r["language"] for r in final_meta))

    _upload_readme(repo_id, token, final_meta, shard_idx)

    print(f"\n{'=' * 60}")
    print(f"Complete in {elapsed:.0f}s")
    print(f"  {len(final_meta)} recordings in {shard_idx} shards")
    print(f"  {len(langs)} languages, {total_min:.0f} min audio, {total_mb:.0f} MB")
    print(f"  https://huggingface.co/datasets/{repo_id}")


if __name__ == "__main__":
    import argparse
    import sys

    if __package__ is None:
        sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

    p = argparse.ArgumentParser(description="Stream-record news radio to HuggingFace")
    p.add_argument("--repo", required=True, help="HF dataset repo (e.g. user/dataset-name)")
    p.add_argument("--token", required=True, help="HF write token")
    p.add_argument("--duration", type=int, default=600, help="Seconds per station (default 600)")
    p.add_argument("--concurrent", type=int, default=50, help="Max parallel recordings")
    p.add_argument("--shard-every", type=int, default=5, help="Upload a shard every N completions")
    p.add_argument("--work-dir", default="/tmp/radio_stream_dataset")
    p.add_argument("--private", action="store_true")
    args = p.parse_args()

    stream_record_and_upload(
        repo_id=args.repo,
        token=args.token,
        duration_seconds=args.duration,
        max_concurrent=args.concurrent,
        shard_every=args.shard_every,
        work_dir=args.work_dir,
        private=args.private,
    )
