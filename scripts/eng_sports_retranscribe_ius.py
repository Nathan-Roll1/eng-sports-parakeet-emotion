#!/usr/bin/env python3
"""Re-decode sampled sports-radio IUs with Parakeet and merge manual labels."""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import io
import json
import math
import pathlib
import re
import time
import urllib.request
from collections import Counter
from typing import Any

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import soundfile as sf
import torch
from transformers import AutoModelForTDT, AutoProcessor


DEFAULT_INPUT = "runs/eng_sports_processing/hf_iu_sampled_dataset/data/train-00000.parquet"
DEFAULT_OUT = "runs/eng_sports_processing/hf_iu_parakeet_labeled_dataset"
DEFAULT_CACHE = "runs/eng_sports_processing/parakeet_iu_retranscripts.jsonl"
DEFAULT_EXPORT_URL = "https://newsroom-api-production.up.railway.app/iu-annotator/api/export.csv"
DEFAULT_MODEL = "nvidia/parakeet-tdt-0.6b-v3"


FEATURES_META: dict[str, dict[str, Any]] = {
    "audio": {"sampling_rate": 16000, "num_channels": 1, "_type": "Audio"},
    "text": {"dtype": "string", "_type": "Value"},
    "parakeet_clip_text": {"dtype": "string", "_type": "Value"},
    "original_text": {"dtype": "string", "_type": "Value"},
    "iu_id": {"dtype": "string", "_type": "Value"},
    "segment_id": {"dtype": "string", "_type": "Value"},
    "source_id": {"dtype": "string", "_type": "Value"},
    "source_name": {"dtype": "string", "_type": "Value"},
    "country_code": {"dtype": "string", "_type": "Value"},
    "recording_date": {"dtype": "string", "_type": "Value"},
    "started_at": {"dtype": "string", "_type": "Value"},
    "raw_gcs_uri": {"dtype": "string", "_type": "Value"},
    "rights_status": {"dtype": "string", "_type": "Value"},
    "parakeet_model": {"dtype": "string", "_type": "Value"},
    "psst_model": {"dtype": "string", "_type": "Value"},
    "segment_duration_seconds": {"dtype": "float64", "_type": "Value"},
    "iu_index": {"dtype": "int32", "_type": "Value"},
    "start_seconds": {"dtype": "float64", "_type": "Value"},
    "end_seconds": {"dtype": "float64", "_type": "Value"},
    "duration_seconds": {"dtype": "float64", "_type": "Value"},
    "start_word_index": {"dtype": "int32", "_type": "Value"},
    "end_word_index_exclusive": {"dtype": "int32", "_type": "Value"},
    "boundary_after_word_index": {"dtype": "int32", "_type": "Value"},
    "alignment_score": {"dtype": "float64", "_type": "Value"},
    "alignment_method": {"dtype": "string", "_type": "Value"},
    "iu_word_count": {"dtype": "int32", "_type": "Value"},
    "speech_rate_wps": {"dtype": "float64", "_type": "Value"},
    "mean_energy_db": {"dtype": "float64", "_type": "Value"},
    "pause_before_ms": {"dtype": "float64", "_type": "Value"},
    "pause_after_ms": {"dtype": "float64", "_type": "Value"},
    "psst_boundary_strength": {"dtype": "string", "_type": "Value"},
    "words_json": {"dtype": "string", "_type": "Value"},
    "manifest_path": {"dtype": "string", "_type": "Value"},
    "transcript_source": {"dtype": "string", "_type": "Value"},
    "parakeet_retranscribed_model": {"dtype": "string", "_type": "Value"},
    "parakeet_retranscribed_at": {"dtype": "string", "_type": "Value"},
    "parakeet_retranscribe_device": {"dtype": "string", "_type": "Value"},
    "parakeet_retranscribe_seconds": {"dtype": "float64", "_type": "Value"},
    "emotion": {"dtype": "string", "_type": "Value"},
    "is_labeled": {"dtype": "bool", "_type": "Value"},
    "annotation_annotator": {"dtype": "string", "_type": "Value"},
    "annotation_notes": {"dtype": "string", "_type": "Value"},
    "annotation_created_at": {"dtype": "string", "_type": "Value"},
    "annotation_updated_at": {"dtype": "string", "_type": "Value"},
}

ARROW_SCHEMA = pa.schema(
    [
        pa.field("audio", pa.struct([pa.field("bytes", pa.binary()), pa.field("path", pa.string())])),
        pa.field("text", pa.string()),
        pa.field("parakeet_clip_text", pa.string()),
        pa.field("original_text", pa.string()),
        pa.field("iu_id", pa.string()),
        pa.field("segment_id", pa.string()),
        pa.field("source_id", pa.string()),
        pa.field("source_name", pa.string()),
        pa.field("country_code", pa.string()),
        pa.field("recording_date", pa.string()),
        pa.field("started_at", pa.string()),
        pa.field("raw_gcs_uri", pa.string()),
        pa.field("rights_status", pa.string()),
        pa.field("parakeet_model", pa.string()),
        pa.field("psst_model", pa.string()),
        pa.field("segment_duration_seconds", pa.float64()),
        pa.field("iu_index", pa.int32()),
        pa.field("start_seconds", pa.float64()),
        pa.field("end_seconds", pa.float64()),
        pa.field("duration_seconds", pa.float64()),
        pa.field("start_word_index", pa.int32()),
        pa.field("end_word_index_exclusive", pa.int32()),
        pa.field("boundary_after_word_index", pa.int32()),
        pa.field("alignment_score", pa.float64()),
        pa.field("alignment_method", pa.string()),
        pa.field("iu_word_count", pa.int32()),
        pa.field("speech_rate_wps", pa.float64()),
        pa.field("mean_energy_db", pa.float64()),
        pa.field("pause_before_ms", pa.float64()),
        pa.field("pause_after_ms", pa.float64()),
        pa.field("psst_boundary_strength", pa.string()),
        pa.field("words_json", pa.string()),
        pa.field("manifest_path", pa.string()),
        pa.field("transcript_source", pa.string()),
        pa.field("parakeet_retranscribed_model", pa.string()),
        pa.field("parakeet_retranscribed_at", pa.string()),
        pa.field("parakeet_retranscribe_device", pa.string()),
        pa.field("parakeet_retranscribe_seconds", pa.float64()),
        pa.field("emotion", pa.string()),
        pa.field("is_labeled", pa.bool_()),
        pa.field("annotation_annotator", pa.string()),
        pa.field("annotation_notes", pa.string()),
        pa.field("annotation_created_at", pa.string()),
        pa.field("annotation_updated_at", pa.string()),
    ]
)


def utc_now() -> str:
    return dt.datetime.now(dt.UTC).isoformat(timespec="seconds")


def clean_text(text: Any) -> str:
    if isinstance(text, list):
        text = " ".join(str(part) for part in text)
    text = "" if text is None else str(text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def text_from_words_json(value: Any) -> str:
    if not value:
        return ""
    try:
        words = json.loads(value)
    except (TypeError, ValueError):
        return ""
    text = " ".join(str(word.get("word", "")) for word in words if isinstance(word, dict) and word.get("word"))
    text = clean_text(text)
    text = re.sub(r"\s+([,.;:?!])", r"\1", text)
    return text


def parse_iso(value: str | None) -> dt.datetime:
    if not value:
        return dt.datetime.min.replace(tzinfo=dt.UTC)
    try:
        normalized = value.replace("Z", "+00:00")
        parsed = dt.datetime.fromisoformat(normalized)
        return parsed if parsed.tzinfo else parsed.replace(tzinfo=dt.UTC)
    except ValueError:
        return dt.datetime.min.replace(tzinfo=dt.UTC)


def load_rows(path: pathlib.Path, max_rows: int | None) -> list[dict[str, Any]]:
    table = pq.read_table(path)
    if max_rows is not None:
        table = table.slice(0, max_rows)
    return table.to_pylist()


def load_cache(path: pathlib.Path) -> dict[str, dict[str, Any]]:
    cache: dict[str, dict[str, Any]] = {}
    if not path.exists():
        return cache
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            item = json.loads(line)
            iu_id = item.get("iu_id")
            if iu_id:
                cache[iu_id] = item
    return cache


def append_cache(path: pathlib.Path, items: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        for item in items:
            handle.write(json.dumps(item, ensure_ascii=False, sort_keys=True) + "\n")
        handle.flush()


def audio_array(audio: dict[str, Any]) -> np.ndarray:
    payload = audio.get("bytes")
    if payload is None:
        raise ValueError("audio bytes missing")
    samples, sample_rate = sf.read(io.BytesIO(payload), dtype="float32", always_2d=False)
    if sample_rate != 16000:
        raise ValueError(f"expected 16000 Hz audio, found {sample_rate}")
    if samples.ndim == 2:
        samples = samples.mean(axis=1)
    return np.asarray(samples, dtype=np.float32)


def pick_device(requested: str) -> str:
    if requested != "auto":
        return requested
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def transcribe_batch(
    batch: list[dict[str, Any]],
    processor: Any,
    model: Any,
    device: str,
    args: argparse.Namespace,
) -> list[dict[str, Any]]:
    start = time.time()
    arrays = [audio_array(row["audio"]) for row in batch]
    inputs = processor(arrays, sampling_rate=processor.feature_extractor.sampling_rate, padding=True)
    inputs.to(model.device, dtype=model.dtype)
    max_duration = max((len(array) / processor.feature_extractor.sampling_rate for array in arrays), default=0.0)
    max_new_tokens = min(
        args.max_new_tokens_ceiling,
        max(args.min_new_tokens, int(math.ceil(max_duration * args.tokens_per_second)) + args.token_margin),
    )
    with torch.inference_mode():
        output = model.generate(**inputs, return_dict_in_generate=True, max_new_tokens=max_new_tokens)
    decoded = processor.decode(output.sequences, skip_special_tokens=True)
    if isinstance(decoded, str):
        decoded = [decoded]
    elapsed = time.time() - start
    per_item = elapsed / max(len(batch), 1)
    return [
        {
            "iu_id": row["iu_id"],
            "text": clean_text(text),
            "old_text": clean_text(row.get("text")),
            "duration_seconds": float(row.get("duration_seconds") or 0.0),
            "source_id": row.get("source_id"),
            "decoded_at": utc_now(),
            "elapsed_seconds": round(per_item, 4),
        }
        for row, text in zip(batch, decoded, strict=True)
    ]


def transcribe_with_fallback(
    batch: list[dict[str, Any]],
    processor: Any,
    model: Any,
    device: str,
    args: argparse.Namespace,
) -> list[dict[str, Any]]:
    try:
        return transcribe_batch(batch, processor, model, device, args)
    except RuntimeError as exc:
        if len(batch) == 1:
            raise
        if device == "mps":
            torch.mps.empty_cache()
        print(f"batch of {len(batch)} failed ({exc}); splitting", flush=True)
        midpoint = len(batch) // 2
        return transcribe_with_fallback(batch[:midpoint], processor, model, device, args) + transcribe_with_fallback(
            batch[midpoint:], processor, model, device, args
        )


def run_transcription(args: argparse.Namespace, rows: list[dict[str, Any]], cache: dict[str, dict[str, Any]]) -> None:
    pending = [row for row in rows if row["iu_id"] not in cache]
    if not pending:
        print(f"transcription cache complete: {len(cache)} rows", flush=True)
        return

    device = pick_device(args.device)
    print(f"loading {args.model_id} on {device}; pending={len(pending)} cached={len(cache)}", flush=True)
    processor = AutoProcessor.from_pretrained(args.model_id)
    model = AutoModelForTDT.from_pretrained(args.model_id, dtype="auto")
    model.to(device)
    model.eval()

    pending.sort(key=lambda row: float(row.get("duration_seconds") or 0.0))
    started = time.time()
    done = 0
    audio_seconds = 0.0
    for offset in range(0, len(pending), args.batch_size):
        batch = pending[offset : offset + args.batch_size]
        decoded = transcribe_with_fallback(batch, processor, model, device, args)
        for item in decoded:
            item["model_id"] = args.model_id
            item["device"] = device
        append_cache(args.cache, decoded)
        for item in decoded:
            cache[item["iu_id"]] = item
            audio_seconds += item["duration_seconds"]
        done += len(decoded)
        if done == len(decoded) or done % args.progress_every <= len(decoded) or done == len(pending):
            wall = max(time.time() - started, 1e-6)
            print(
                f"transcribed {done}/{len(pending)} pending "
                f"({len(cache)}/{len(rows)} total cache), "
                f"audio={audio_seconds/3600:.3f}h, wall={wall/60:.1f}m",
                flush=True,
            )
        if device == "mps" and done % max(args.batch_size * 10, 1) == 0:
            torch.mps.empty_cache()


def fetch_annotations(url: str, include_codex: bool) -> dict[str, dict[str, str]]:
    with urllib.request.urlopen(url, timeout=60) as response:
        text = response.read().decode("utf-8")
    rows = list(csv.DictReader(io.StringIO(text)))
    labels: dict[str, dict[str, str]] = {}
    for row in rows:
        annotator = row.get("annotator", "")
        if annotator.startswith("codex-") and not include_codex:
            continue
        iu_id = row.get("iu_id", "")
        if not iu_id:
            continue
        previous = labels.get(iu_id)
        if previous is None or parse_iso(row.get("updated_at")) >= parse_iso(previous.get("updated_at")):
            labels[iu_id] = row
    return labels


def with_labels(
    rows: list[dict[str, Any]],
    cache: dict[str, dict[str, Any]],
    labels: dict[str, dict[str, str]],
    args: argparse.Namespace,
) -> list[dict[str, Any]]:
    stamp = utc_now()
    out: list[dict[str, Any]] = []
    missing = [row["iu_id"] for row in rows if row["iu_id"] not in cache]
    if missing:
        raise RuntimeError(f"{len(missing)} rows are missing retranscripts; first={missing[0]}")

    for row in rows:
        item = dict(row)
        transcript = cache[row["iu_id"]]
        label = labels.get(row["iu_id"])
        clip_text = clean_text(transcript.get("text"))
        final_text = clip_text
        transcript_source = "parakeet_tdt_clip_redecode"
        if not final_text:
            final_text = text_from_words_json(row.get("words_json"))
            transcript_source = "parakeet_tdt_word_offsets_fallback"
        if not final_text:
            transcript_source = "parakeet_tdt_blank_decode"
        item["original_text"] = clean_text(row.get("text"))
        item["parakeet_clip_text"] = clip_text
        item["text"] = final_text
        item["transcript_source"] = transcript_source
        item["parakeet_retranscribed_model"] = transcript.get("model_id") or args.model_id
        item["parakeet_retranscribed_at"] = transcript.get("decoded_at") or stamp
        item["parakeet_retranscribe_device"] = transcript.get("device") or pick_device(args.device)
        item["parakeet_retranscribe_seconds"] = float(transcript.get("elapsed_seconds") or math.nan)
        item["emotion"] = label.get("emotion") if label else None
        item["is_labeled"] = bool(label)
        item["annotation_annotator"] = label.get("annotator") if label else None
        item["annotation_notes"] = label.get("notes") if label else None
        item["annotation_created_at"] = label.get("created_at") if label else None
        item["annotation_updated_at"] = label.get("updated_at") if label else None
        out.append(item)
    return out


def write_dataset(rows: list[dict[str, Any]], out_dir: pathlib.Path, summary: dict[str, Any]) -> None:
    data_dir = out_dir / "data"
    data_dir.mkdir(parents=True, exist_ok=True)
    parquet_path = data_dir / "train-00000.parquet"
    table = pa.Table.from_pylist(rows, schema=ARROW_SCHEMA)
    table = table.replace_schema_metadata(
        {b"huggingface": json.dumps({"info": {"features": FEATURES_META}}, sort_keys=True).encode("utf-8")}
    )
    pq.write_table(table, parquet_path, compression="zstd", use_dictionary=True)
    (out_dir / "dataset_summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    (out_dir / "README.md").write_text(readme(summary), encoding="utf-8")


def readme(summary: dict[str, Any]) -> str:
    station_lines = "\n".join(f"- {key}: {value}" for key, value in summary["station_counts"].items())
    emotion_lines = "\n".join(f"- {key}: {value}" for key, value in summary["emotion_counts"].items())
    annotator_lines = "\n".join(f"- {key}: {value}" for key, value in summary["annotator_counts"].items())
    transcript_source_lines = "\n".join(f"- {key}: {value}" for key, value in summary["transcript_source_counts"].items())
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
- psst
- prosody
- intonation-units
- emotion
- annotated
size_categories:
- 1K<n<10K
---

# English Sports Radio PSST IU Sample

This dataset contains one row per PSST intonation unit from English sports-radio broadcasts.
The `text` field has been replaced with direct clip-level `nvidia/parakeet-tdt-0.6b-v3`
transcription, and manual emotion annotations from the Railway annotator have been joined by `iu_id`.

## Contents

- IU rows: {summary["row_count"]}
- Audio hours: {summary["audio_hours"]:.3f}
- Labeled rows: {summary["labeled_rows"]}
- Unlabeled rows: {summary["unlabeled_rows"]}
- Parakeet model: `{summary["parakeet_model"]}`
- Generated at: {summary["generated_at"]}

## Transcript Source Counts

{transcript_source_lines}

## Emotion Counts

{emotion_lines or "- none"}

## Annotator Counts

{annotator_lines or "- none"}

## Station Counts

{station_lines}

## Columns

- `audio`: embedded 16 kHz mono FLAC IU clip.
- `text`: replacement Parakeet transcript. This is direct clip-level Parakeet output when available, with a Parakeet word-offset fallback for clips that decode blank in isolation. Clips with no Parakeet text remain blank rather than keeping the old noisy transcript.
- `parakeet_clip_text`: raw direct clip-level Parakeet output.
- `original_text`: previous IU transcript retained for audit.
- `emotion`: manual emotion label when available.
- `annotation_annotator`, `annotation_created_at`, `annotation_updated_at`, `annotation_notes`: label provenance.
- `words_json`, `psst_boundary_strength`, timing and prosody columns: original PSST/IU metadata.
"""


def summarize(rows: list[dict[str, Any]], labels: dict[str, dict[str, str]], args: argparse.Namespace) -> dict[str, Any]:
    station_counts = Counter(row["source_id"] for row in rows)
    emotion_counts = Counter(row.get("emotion") or "unlabeled" for row in rows)
    annotator_counts = Counter(row.get("annotation_annotator") for row in rows if row.get("annotation_annotator"))
    transcript_source_counts = Counter(row.get("transcript_source") for row in rows)
    labeled_rows = sum(1 for row in rows if row.get("is_labeled"))
    return {
        "row_count": len(rows),
        "audio_hours": round(sum(float(row.get("duration_seconds") or 0.0) for row in rows) / 3600.0, 6),
        "labeled_rows": labeled_rows,
        "unlabeled_rows": len(rows) - labeled_rows,
        "label_export_rows_used": len(labels),
        "station_counts": dict(sorted(station_counts.items())),
        "emotion_counts": dict(sorted(emotion_counts.items())),
        "annotator_counts": dict(sorted(annotator_counts.items())),
        "transcript_source_counts": dict(sorted(transcript_source_counts.items())),
        "parakeet_model": args.model_id,
        "transcript_source": "parakeet_tdt_clip_redecode_with_word_offset_fallback",
        "generated_at": utc_now(),
        "input_parquet": str(args.input),
        "annotation_export_url": args.annotation_export_url,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=pathlib.Path, default=pathlib.Path(DEFAULT_INPUT))
    parser.add_argument("--out-dir", type=pathlib.Path, default=pathlib.Path(DEFAULT_OUT))
    parser.add_argument("--cache", type=pathlib.Path, default=pathlib.Path(DEFAULT_CACHE))
    parser.add_argument("--annotation-export-url", default=DEFAULT_EXPORT_URL)
    parser.add_argument("--model-id", default=DEFAULT_MODEL)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--max-rows", type=int, default=None)
    parser.add_argument("--progress-every", type=int, default=100)
    parser.add_argument("--tokens-per-second", type=float, default=16.0)
    parser.add_argument("--token-margin", type=int, default=64)
    parser.add_argument("--min-new-tokens", type=int, default=96)
    parser.add_argument("--max-new-tokens-ceiling", type=int, default=1024)
    parser.add_argument("--skip-transcribe", action="store_true")
    parser.add_argument("--include-codex-labels", action="store_true")
    args = parser.parse_args()

    rows = load_rows(args.input, args.max_rows)
    cache = load_cache(args.cache)
    if not args.skip_transcribe:
        run_transcription(args, rows, cache)
    labels = fetch_annotations(args.annotation_export_url, args.include_codex_labels)
    merged = with_labels(rows, cache, labels, args)
    summary = summarize(merged, labels, args)
    write_dataset(merged, args.out_dir, summary)
    print(json.dumps(summary, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
