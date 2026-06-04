from __future__ import annotations

import argparse
import concurrent.futures
import io
import json
import math
import os
import re
import shutil
import subprocess
import sys
import tempfile
import time
import warnings
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

if __name__ == "__main__" and __package__ is None:
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from dataset_builder.config import FFMPEG_BIN

FFPROBE_BIN = (
    os.environ.get("FFPROBE_BIN")
    or shutil.which("ffprobe")
    or str(Path(FFMPEG_BIN).with_name("ffprobe"))
)

DEFAULT_PARAKEET_MODEL = "nvidia/parakeet-tdt-0.6b-v3"
DEFAULT_PSST_MODEL = "NathanRoll/psst-medium-en"
DEFAULT_HF_REPO = "NathanRoll/eng-sports-radio-psst-iu"
IU_SEPARATOR_RE = re.compile(r"\s*!{5,}\s*")
WORD_RE = re.compile(r"[A-Za-z0-9]+(?:'[A-Za-z0-9]+)?")
WHISPER_CONTROL_RE = re.compile(r"<\|[^>]+?\|>")

warnings.filterwarnings("ignore", message="Using the model-agnostic default `max_length`.*")
warnings.filterwarnings("ignore", message="Both `max_new_tokens`.*")


@dataclass(frozen=True)
class SegmentManifest:
    segment_id: str
    source_id: str
    source_name: str
    country_code: str
    date: str
    stem: str
    manifest_path: Path
    raw_local_path: Path | None
    raw_gcs_uri: str
    duration_seconds: float
    started_at: str
    completed_at: str
    rights_status: str
    source_sample_rate_hz: int
    target_bit_depth: int


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def log(message: str) -> None:
    print(f"[{utc_now()}] {message}", flush=True)


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    tmp.replace(path)


def run_cmd(cmd: list[str], timeout: int | None = None) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        timeout=timeout,
        check=False,
    )


def parse_iso_date(value: str) -> str:
    if not value:
        return "unknown-date"
    try:
        return datetime.fromisoformat(value.replace("Z", "+00:00")).date().isoformat()
    except ValueError:
        return value[:10] if len(value) >= 10 else "unknown-date"


def parse_manifest(path: Path, cwd: Path) -> SegmentManifest | None:
    try:
        raw = read_json(path)
    except (OSError, json.JSONDecodeError):
        return None

    quality = raw.get("quality") or {}
    if quality.get("valid") is not True:
        return None

    source = raw.get("source") or {}
    capture = raw.get("capture") or {}
    local = raw.get("local") or {}
    gcs = raw.get("gcs") or {}

    source_id = str(source.get("id") or path.parents[1].name)
    started_at = str(capture.get("started_at") or "")
    date = parse_iso_date(started_at) if started_at else path.parent.name
    stem = path.stem
    raw_local = None
    raw_local_s = str(local.get("audio_path") or "")
    if raw_local_s:
        candidate = Path(raw_local_s)
        raw_local = candidate if candidate.is_absolute() else cwd / candidate

    duration = 0.0
    try:
        duration = float(((quality.get("output_probe") or {}).get("format") or {}).get("duration") or 0.0)
    except (TypeError, ValueError):
        duration = 0.0
    if duration <= 0:
        try:
            duration = float(capture.get("duration_requested_seconds") or 0.0)
        except (TypeError, ValueError):
            duration = 0.0

    return SegmentManifest(
        segment_id=f"{source_id}/{date}/{stem}",
        source_id=source_id,
        source_name=str(source.get("name") or source_id),
        country_code=str(source.get("country_code") or ""),
        date=date,
        stem=stem,
        manifest_path=path,
        raw_local_path=raw_local,
        raw_gcs_uri=str(gcs.get("audio_uri") or ""),
        duration_seconds=duration,
        started_at=started_at,
        completed_at=str(capture.get("completed_at") or ""),
        rights_status=str(source.get("rights_status") or ""),
        source_sample_rate_hz=int(quality.get("source_sample_rate_hz") or 0),
        target_bit_depth=int(quality.get("target_bit_depth") or 0),
    )


def interleave_records(records: list[SegmentManifest]) -> list[SegmentManifest]:
    groups: dict[str, list[SegmentManifest]] = defaultdict(list)
    for rec in records:
        groups[rec.source_id].append(rec)
    # Interleave sources so partial shards contain a useful English sports mix.
    for rows in groups.values():
        rows.sort(key=lambda r: (r.started_at, r.stem))
    mixed: list[SegmentManifest] = []
    source_order = sorted(groups)
    while True:
        added = False
        for source_id in source_order:
            rows = groups[source_id]
            if rows:
                mixed.append(rows.pop(0))
                added = True
        if not added:
            return mixed


def load_manifests(root: Path, source_ids: set[str] | None = None) -> list[SegmentManifest]:
    cwd = Path.cwd().resolve()
    records: list[SegmentManifest] = []
    for path in sorted(root.rglob("*.json")):
        rec = parse_manifest(path, cwd)
        if rec is None:
            continue
        if source_ids and rec.source_id not in source_ids:
            continue
        records.append(rec)
    return interleave_records(records)


def record_path(output_dir: Path, rec: SegmentManifest) -> Path:
    return output_dir / "records" / rec.source_id / rec.date / f"{rec.stem}.json"


def failure_path(output_dir: Path, rec: SegmentManifest) -> Path:
    return output_dir / "failures" / rec.source_id / rec.date / f"{rec.stem}.json"


def stripped_audio_path(output_dir: Path, rec: SegmentManifest) -> Path:
    return output_dir / "audio_16k_flac" / rec.source_id / rec.date / f"{rec.stem}_16k.flac"


def raw_cache_path(output_dir: Path, rec: SegmentManifest) -> Path:
    return output_dir / "raw_cache" / rec.source_id / rec.date / f"{rec.stem}.wav"


def probe_audio(path: Path) -> dict[str, Any]:
    cmd = [
        FFPROBE_BIN,
        "-hide_banner",
        "-v",
        "error",
        "-select_streams",
        "a:0",
        "-show_entries",
        "stream=codec_name,sample_rate,channels,bits_per_sample,bits_per_raw_sample:format=duration,size",
        "-of",
        "json",
        str(path),
    ]
    proc = run_cmd(cmd, timeout=60)
    payload: dict[str, Any] = {}
    if proc.stdout.strip():
        try:
            payload = json.loads(proc.stdout)
        except json.JSONDecodeError:
            payload = {"stdout": proc.stdout[:2000]}
    payload["ok"] = proc.returncode == 0
    payload["stderr"] = proc.stderr[-2000:]
    return payload


def valid_stripped_probe(probe: dict[str, Any], sample_rate: int) -> bool:
    if not probe.get("ok"):
        return False
    stream = (probe.get("streams") or [{}])[0]
    try:
        return int(stream.get("sample_rate") or 0) == sample_rate and int(stream.get("channels") or 0) == 1
    except (TypeError, ValueError):
        return False


def fetch_raw_audio(
    rec: SegmentManifest,
    output_dir: Path,
    keep_raw_cache: bool,
    gcs_timeout: int,
) -> tuple[Path, bool]:
    if rec.raw_local_path and rec.raw_local_path.exists() and rec.raw_local_path.stat().st_size > 0:
        return rec.raw_local_path, False
    if not rec.raw_gcs_uri:
        raise FileNotFoundError(f"No local audio or GCS URI for {rec.segment_id}")

    dest = raw_cache_path(output_dir, rec)
    if dest.exists() and dest.stat().st_size > 0:
        return dest, keep_raw_cache

    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(".tmp.wav")
    if tmp.exists():
        tmp.unlink()
    proc = run_cmd(["gcloud", "storage", "cp", rec.raw_gcs_uri, str(tmp)], timeout=gcs_timeout)
    if proc.returncode != 0:
        raise RuntimeError(f"gcloud cp failed for {rec.raw_gcs_uri}: {proc.stderr[-1000:]}")
    tmp.replace(dest)
    return dest, not keep_raw_cache


def ensure_stripped_audio(
    rec: SegmentManifest,
    output_dir: Path,
    sample_rate: int,
    keep_raw_cache: bool,
    gcs_timeout: int,
) -> tuple[Path, dict[str, Any]]:
    out = stripped_audio_path(output_dir, rec)
    if out.exists() and out.stat().st_size > 0:
        probe = probe_audio(out)
        if valid_stripped_probe(probe, sample_rate):
            return out, probe

    raw_audio, delete_raw_after = fetch_raw_audio(rec, output_dir, keep_raw_cache, gcs_timeout)
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_suffix(".tmp.flac")
    if tmp.exists():
        tmp.unlink()

    cmd = [
        FFMPEG_BIN,
        "-hide_banner",
        "-nostdin",
        "-loglevel",
        "error",
        "-y",
        "-i",
        str(raw_audio),
        "-map",
        "0:a:0",
        "-vn",
        "-ac",
        "1",
        "-ar",
        str(sample_rate),
        "-sample_fmt",
        "s16",
        "-c:a",
        "flac",
        "-compression_level",
        "5",
        str(tmp),
    ]
    proc = run_cmd(cmd, timeout=max(180, int(rec.duration_seconds * 4) + 60))
    if proc.returncode != 0:
        raise RuntimeError(f"ffmpeg strip failed for {rec.segment_id}: {proc.stderr[-1200:]}")
    tmp.replace(out)
    if delete_raw_after and raw_audio.exists():
        raw_audio.unlink()

    probe = probe_audio(out)
    if not valid_stripped_probe(probe, sample_rate):
        raise RuntimeError(f"stripped audio failed probe for {rec.segment_id}: {probe}")
    return out, probe


def load_audio(path: Path) -> tuple[Any, int]:
    import numpy as np
    import soundfile as sf

    audio, sr = sf.read(str(path), dtype="float32", always_2d=False)
    if getattr(audio, "ndim", 1) > 1:
        audio = np.mean(audio, axis=1)
    return audio.astype("float32", copy=False), int(sr)


def choose_device(preference: str) -> str:
    if preference != "auto":
        return preference
    import torch

    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def normalize_words(text: str) -> list[str]:
    return [m.group(0).lower() for m in WORD_RE.finditer(text)]


def clean_psst_text(text: str) -> str:
    text = WHISPER_CONTROL_RE.sub(" ", text or "")
    return re.sub(r"\s+", " ", text).strip()


def even_timed_words(text: str, duration_seconds: float) -> list[dict[str, Any]]:
    matches = list(re.finditer(r"\S+", text))
    if not matches:
        return []
    total = max(duration_seconds, 0.01)
    words = []
    for idx, match in enumerate(matches):
        words.append(
            {
                "word": match.group(0),
                "start_seconds": total * idx / len(matches),
                "end_seconds": total * (idx + 1) / len(matches),
                "word_index": idx,
                "timestamp_source": "even_fallback",
            }
        )
    return words


def token_offsets_to_words(token_offsets: list[dict[str, Any]], fallback_text: str, duration_seconds: float) -> list[dict[str, Any]]:
    chars: list[str] = []
    char_times: list[tuple[float, float]] = []
    for offset in token_offsets:
        token = str(offset.get("token") or "")
        if not token:
            continue
        start = float(offset.get("start") or 0.0)
        end = float(offset.get("end") or start)
        span = max(end - start, 0.0)
        for i, ch in enumerate(token):
            chars.append(ch)
            c0 = start + span * i / max(len(token), 1)
            c1 = start + span * (i + 1) / max(len(token), 1)
            char_times.append((c0, c1))

    text = "".join(chars)
    if not text.strip():
        return even_timed_words(fallback_text, duration_seconds)

    words: list[dict[str, Any]] = []
    for idx, match in enumerate(re.finditer(r"\S+", text)):
        start_i = match.start()
        end_i = max(match.end() - 1, start_i)
        start = char_times[start_i][0] if start_i < len(char_times) else 0.0
        end = char_times[end_i][1] if end_i < len(char_times) else start
        words.append(
            {
                "word": match.group(0),
                "start_seconds": round(float(start), 3),
                "end_seconds": round(float(end), 3),
                "word_index": idx,
                "timestamp_source": "parakeet_tdt_token_offsets",
            }
        )
    return words or even_timed_words(fallback_text, duration_seconds)


class ParakeetRunner:
    def __init__(self, model_id: str, device_preference: str) -> None:
        import torch
        from transformers import AutoModelForTDT, AutoProcessor
        from transformers.utils import logging as transformers_logging

        transformers_logging.set_verbosity_error()
        os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
        self.torch = torch
        self.model_id = model_id
        self.device = choose_device(device_preference)
        self.dtype = torch.float16 if self.device in {"mps", "cuda"} else torch.float32
        log(f"loading Parakeet {model_id} on {self.device} ({self.dtype})")
        self.processor = AutoProcessor.from_pretrained(model_id)
        self.model = AutoModelForTDT.from_pretrained(model_id, dtype=self.dtype)
        self.model.eval().to(self.device)

    def _move_inputs(self, inputs: Any) -> dict[str, Any]:
        moved: dict[str, Any] = {}
        for key, value in dict(inputs).items():
            if hasattr(value, "to"):
                if self.torch.is_floating_point(value):
                    moved[key] = value.to(device=self.device, dtype=self.dtype)
                else:
                    moved[key] = value.to(device=self.device)
            else:
                moved[key] = value
        return moved

    def transcribe(self, audio: Any, sample_rate: int, duration_seconds: float) -> dict[str, Any]:
        inputs = self.processor([audio], sampling_rate=sample_rate, return_tensors="pt", padding=True)
        inputs = self._move_inputs(inputs)
        with self.torch.inference_mode():
            outputs = self.model.generate(**inputs, return_dict_in_generate=True)

        sequences = outputs.sequences.detach().cpu()
        text = self.processor.batch_decode(sequences, skip_special_tokens=True)[0].strip()
        token_offsets: list[dict[str, Any]] = []
        durations = getattr(outputs, "durations", None)
        if durations is not None:
            try:
                decoded = self.processor.decode(
                    sequences,
                    durations=durations.detach().cpu(),
                    skip_special_tokens=True,
                )
                if isinstance(decoded, tuple) and len(decoded) == 2:
                    token_batches = decoded[1]
                    if token_batches:
                        token_offsets = token_batches[0]
            except Exception as exc:  # noqa: BLE001
                log(f"Parakeet timestamp decode fell back to even timings: {type(exc).__name__}: {exc}")

        words = token_offsets_to_words(token_offsets, text, duration_seconds)
        return {
            "model": self.model_id,
            "device": self.device,
            "text": text,
            "words": words,
            "token_offsets": token_offsets,
        }


class PSSTRunner:
    def __init__(self, model_id: str, device_preference: str) -> None:
        import torch
        from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor
        from transformers.utils import logging as transformers_logging

        transformers_logging.set_verbosity_error()
        os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
        self.torch = torch
        self.model_id = model_id
        self.device = choose_device(device_preference)
        self.dtype = torch.float16 if self.device in {"mps", "cuda"} else torch.float32
        log(f"loading PSST {model_id} on {self.device} ({self.dtype})")
        self.processor = AutoProcessor.from_pretrained(model_id)
        self.model = AutoModelForSpeechSeq2Seq.from_pretrained(model_id, dtype=self.dtype)
        self.model.eval().to(self.device)

    def _chunk_text(self, audio: Any, sample_rate: int, max_new_tokens: int) -> str:
        inputs = self.processor(audio, sampling_rate=sample_rate, return_tensors="pt")
        input_features = inputs.input_features.to(device=self.device, dtype=self.dtype)
        kwargs: dict[str, Any] = {"max_new_tokens": max_new_tokens}
        with self.torch.inference_mode():
            ids = self.model.generate(input_features, **kwargs)
        return clean_psst_text(self.processor.batch_decode(ids, skip_special_tokens=True)[0])

    def transcribe_ius(
        self,
        audio: Any,
        sample_rate: int,
        chunk_seconds: float,
        max_new_tokens: int,
    ) -> dict[str, Any]:
        n = len(audio)
        chunk = max(1, int(sample_rate * chunk_seconds))
        texts: list[str] = []
        for start in range(0, n, chunk):
            part = audio[start : min(n, start + chunk)]
            if len(part) < int(sample_rate * 0.25):
                continue
            texts.append(self._chunk_text(part, sample_rate, max_new_tokens))
        text = " ".join(t for t in texts if t).strip()
        return {"model": self.model_id, "device": self.device, "text": text}


def split_psst_ius(text: str) -> list[str]:
    chunks = [re.sub(r"\s+", " ", c).strip() for c in IU_SEPARATOR_RE.split(clean_psst_text(text))]
    return [c for c in chunks if c]


def best_word_span(iu_tokens: list[str], word_tokens: list[str], cursor: int) -> tuple[int, int, float, str]:
    if not iu_tokens or not word_tokens:
        return cursor, cursor, 0.0, "empty"
    max_start = min(len(word_tokens), cursor + 120)
    best = (cursor, min(len(word_tokens), cursor + max(1, len(iu_tokens))), -1.0)
    for start in range(cursor, max_start):
        i = 0
        j = start
        matches = 0
        last_j = start
        max_j = min(len(word_tokens), start + len(iu_tokens) + 12)
        while i < len(iu_tokens) and j < max_j:
            if iu_tokens[i] == word_tokens[j]:
                matches += 1
                i += 1
                last_j = j + 1
            j += 1
        score = matches / max(len(iu_tokens), 1) - (start - cursor) * 0.003
        if score > best[2]:
            best = (start, max(last_j, start + 1), score)
    if best[2] >= 0.45:
        return best[0], best[1], round(max(0.0, best[2]), 3), "token_match"
    fallback_end = min(len(word_tokens), cursor + max(1, len(iu_tokens)))
    return cursor, fallback_end, round(max(0.0, best[2]), 3), "word_count_fallback"


def align_ius(psst_text: str, words: list[dict[str, Any]], duration_seconds: float) -> list[dict[str, Any]]:
    iu_texts = split_psst_ius(psst_text)
    if not iu_texts and psst_text.strip():
        iu_texts = [re.sub(r"\s+", " ", psst_text).strip()]
    if not iu_texts:
        return []

    word_tokens = [normalize_words(str(w.get("word") or ""))[0] if normalize_words(str(w.get("word") or "")) else "" for w in words]
    cursor = 0
    boundaries: list[dict[str, Any]] = []
    for idx, iu_text in enumerate(iu_texts):
        iu_tokens = normalize_words(iu_text)
        start_i, end_i, score, method = best_word_span(iu_tokens, word_tokens, cursor)
        cursor = max(end_i, cursor)
        if words and start_i < len(words) and end_i > start_i:
            start_s = float(words[start_i].get("start_seconds") or 0.0)
            end_s = float(words[min(end_i - 1, len(words) - 1)].get("end_seconds") or start_s)
        else:
            start_s = duration_seconds * idx / len(iu_texts)
            end_s = duration_seconds * (idx + 1) / len(iu_texts)
        boundaries.append(
            {
                "iu_index": idx,
                "text": iu_text,
                "start_seconds": round(max(0.0, start_s), 3),
                "end_seconds": round(max(start_s, min(duration_seconds, end_s)), 3),
                "start_word_index": int(start_i),
                "end_word_index_exclusive": int(end_i),
                "boundary_after_word_index": int(end_i - 1) if end_i > start_i else None,
                "alignment_score": score,
                "alignment_method": method,
            }
        )
    return boundaries


def db(value: float) -> float:
    return 20.0 * math.log10(max(value, 1e-8))


def frame_rms(audio: Any, sample_rate: int) -> tuple[Any, Any]:
    import numpy as np

    frame = max(1, int(sample_rate * 0.025))
    hop = max(1, int(sample_rate * 0.010))
    if len(audio) < frame:
        rms = np.array([float(np.sqrt(np.mean(np.square(audio)))) if len(audio) else 0.0], dtype="float32")
        times = np.array([0.0], dtype="float32")
        return rms, times
    squared = np.square(audio, dtype="float32")
    kernel = np.ones(frame, dtype="float32") / frame
    rms = np.sqrt(np.convolve(squared, kernel, mode="valid")[::hop])
    times = (np.arange(len(rms), dtype="float32") * hop + frame / 2) / sample_rate
    return rms.astype("float32", copy=False), times


def contiguous_silence_ms(rms: Any, times: Any, boundary: float, threshold: float, direction: str) -> float:
    if len(rms) == 0:
        return 0.0
    if direction == "after":
        idxs = [i for i, t in enumerate(times) if boundary <= float(t) <= boundary + 0.8]
    else:
        idxs = [i for i, t in enumerate(times) if boundary - 0.8 <= float(t) <= boundary]
        idxs.reverse()
    count = 0
    for i in idxs:
        if float(rms[i]) <= threshold:
            count += 1
        else:
            break
    return round(count * 10.0, 1)


def add_prosody_labels(audio: Any, sample_rate: int, boundaries: list[dict[str, Any]]) -> dict[str, Any]:
    import numpy as np

    rms, times = frame_rms(audio, sample_rate)
    if len(rms):
        silence_threshold = max(float(np.percentile(rms, 20)) * 1.5, float(np.median(rms)) * 0.25, 1e-4)
        silence_ratio = float(np.mean(rms <= silence_threshold))
        mean_rms_db = db(float(np.mean(rms)))
        peak_db = db(float(np.max(np.abs(audio))) if len(audio) else 0.0)
    else:
        silence_threshold = 1e-4
        silence_ratio = 1.0
        mean_rms_db = -160.0
        peak_db = -160.0

    for boundary in boundaries:
        start = float(boundary["start_seconds"])
        end = float(boundary["end_seconds"])
        mask = (times >= start) & (times <= end) if len(times) else []
        iu_rms = rms[mask] if len(times) else []
        word_count = max(0, int(boundary["end_word_index_exclusive"]) - int(boundary["start_word_index"]))
        duration = max(end - start, 0.01)
        energy_db = db(float(np.mean(iu_rms))) if len(iu_rms) else mean_rms_db
        pause_before = contiguous_silence_ms(rms, times, start, silence_threshold, "before")
        pause_after = contiguous_silence_ms(rms, times, end, silence_threshold, "after")
        if pause_after >= 400:
            strength = "major"
        elif pause_after >= 150:
            strength = "minor"
        else:
            strength = "continuing"
        boundary["prosody"] = {
            "iu_duration_seconds": round(duration, 3),
            "iu_word_count": word_count,
            "speech_rate_wps": round(word_count / duration, 3),
            "mean_energy_db": round(energy_db, 2),
            "pause_before_ms": pause_before,
            "pause_after_ms": pause_after,
            "psst_boundary_strength": strength,
        }

    return {
        "mean_rms_db": round(mean_rms_db, 2),
        "peak_db": round(peak_db, 2),
        "silence_threshold_rms": round(float(silence_threshold), 8),
        "silence_ratio": round(silence_ratio, 4),
    }


def process_one(
    rec: SegmentManifest,
    output_dir: Path,
    parakeet: ParakeetRunner,
    psst: PSSTRunner | None,
    args: argparse.Namespace,
) -> dict[str, Any]:
    stripped, stripped_probe = ensure_stripped_audio(
        rec,
        output_dir,
        sample_rate=args.sample_rate,
        keep_raw_cache=args.keep_raw_cache,
        gcs_timeout=args.gcs_timeout,
    )
    audio, sr = load_audio(stripped)
    duration_seconds = len(audio) / float(sr) if sr else rec.duration_seconds

    parakeet_result = parakeet.transcribe(audio, sr, duration_seconds)
    if psst is None:
        psst_result = {"model": None, "device": None, "text": ""}
    else:
        psst_result = psst.transcribe_ius(
            audio,
            sr,
            chunk_seconds=args.psst_chunk_seconds,
            max_new_tokens=args.psst_max_new_tokens,
        )

    iu_boundaries = align_ius(psst_result["text"], parakeet_result["words"], duration_seconds)
    segment_prosody = add_prosody_labels(audio, sr, iu_boundaries)

    return {
        "status": "ok",
        "processed_at": utc_now(),
        "segment_id": rec.segment_id,
        "source_id": rec.source_id,
        "source_name": rec.source_name,
        "country_code": rec.country_code,
        "recording_date": rec.date,
        "started_at": rec.started_at,
        "completed_at": rec.completed_at,
        "duration_seconds": round(float(duration_seconds), 3),
        "sample_rate": int(sr),
        "channels": 1,
        "audio_path_16k": str(stripped.resolve()),
        "audio_format": "flac",
        "raw_gcs_uri": rec.raw_gcs_uri,
        "raw_local_path": str(rec.raw_local_path.resolve()) if rec.raw_local_path else "",
        "manifest_path": str(rec.manifest_path.resolve()),
        "rights_status": rec.rights_status,
        "source_sample_rate_hz": rec.source_sample_rate_hz,
        "source_target_bit_depth": rec.target_bit_depth,
        "parakeet_model": parakeet_result["model"],
        "parakeet_device": parakeet_result["device"],
        "parakeet_text": parakeet_result["text"],
        "parakeet_words": parakeet_result["words"],
        "psst_model": psst_result["model"],
        "psst_device": psst_result["device"],
        "psst_text": psst_result["text"],
        "iu_boundaries": iu_boundaries,
        "segment_prosody": segment_prosody,
        "stripped_probe": stripped_probe,
    }


def output_records(output_dir: Path) -> list[Path]:
    return sorted((output_dir / "records").rglob("*.json")) if (output_dir / "records").exists() else []


def load_output_rows(output_dir: Path) -> list[dict[str, Any]]:
    rows = []
    for path in output_records(output_dir):
        try:
            row = read_json(path)
        except (OSError, json.JSONDecodeError):
            continue
        if row.get("status") == "ok":
            rows.append(row)
    rows.sort(key=lambda r: (r.get("source_id", ""), r.get("started_at", ""), r.get("segment_id", "")))
    return rows


def package_dataset(output_dir: Path, max_rows_per_shard: int) -> dict[str, Any]:
    import pyarrow as pa
    import pyarrow.parquet as pq

    rows = load_output_rows(output_dir)
    data_dir = output_dir / "hf_dataset" / "data"
    tmp_dir = output_dir / "hf_dataset" / "data.tmp"
    if tmp_dir.exists():
        shutil.rmtree(tmp_dir)
    tmp_dir.mkdir(parents=True, exist_ok=True)

    fields = [
        "segment_id",
        "source_id",
        "source_name",
        "country_code",
        "recording_date",
        "started_at",
        "duration_seconds",
        "sample_rate",
        "channels",
        "audio_format",
        "raw_gcs_uri",
        "manifest_path",
        "rights_status",
        "source_sample_rate_hz",
        "source_target_bit_depth",
        "parakeet_model",
        "parakeet_device",
        "parakeet_text",
        "psst_model",
        "psst_device",
        "psst_text",
        "iu_boundaries_json",
        "parakeet_words_json",
        "segment_prosody_json",
    ]

    shard_size = max(1, max_rows_per_shard)
    written = 0
    total_audio_bytes = 0
    for shard_idx in range(0, len(rows), shard_size):
        chunk = rows[shard_idx : shard_idx + shard_size]
        columns: dict[str, list[Any]] = {field: [] for field in fields}
        audio_bytes: list[bytes] = []
        audio_paths: list[str] = []
        for row in chunk:
            audio_path = Path(row["audio_path_16k"])
            data = audio_path.read_bytes()
            total_audio_bytes += len(data)
            audio_bytes.append(data)
            audio_paths.append(audio_path.name)
            for field in fields:
                if field == "iu_boundaries_json":
                    columns[field].append(json.dumps(row.get("iu_boundaries") or [], ensure_ascii=False))
                elif field == "parakeet_words_json":
                    columns[field].append(json.dumps(row.get("parakeet_words") or [], ensure_ascii=False))
                elif field == "segment_prosody_json":
                    columns[field].append(json.dumps(row.get("segment_prosody") or {}, ensure_ascii=False))
                else:
                    columns[field].append(row.get(field))

        audio_struct = pa.StructArray.from_arrays(
            [pa.array(audio_bytes, type=pa.binary()), pa.array(audio_paths, type=pa.string())],
            names=["bytes", "path"],
        )
        arrays = {"audio": audio_struct}
        for field, values in columns.items():
            arrays[field] = pa.array(values)
        table = pa.table(arrays)
        shard_path = tmp_dir / f"train-{written:05d}.parquet"
        pq.write_table(table, shard_path, compression="zstd")
        written += 1

    if data_dir.exists():
        shutil.rmtree(data_dir)
    tmp_dir.replace(data_dir)
    write_dataset_card(output_dir, rows, total_audio_bytes, written)
    return {
        "rows": len(rows),
        "shards": written,
        "total_hours": round(sum(float(r.get("duration_seconds") or 0.0) for r in rows) / 3600.0, 3),
        "embedded_audio_gb": round(total_audio_bytes / 1_000_000_000, 3),
        "dataset_dir": str((output_dir / "hf_dataset").resolve()),
    }


def iu_features_schema() -> Any:
    from datasets import Audio, Features, Value

    features = Features(
        {
            "audio": Audio(sampling_rate=16000, num_channels=1),
            "text": Value("string"),
            "iu_id": Value("string"),
            "segment_id": Value("string"),
            "source_id": Value("string"),
            "source_name": Value("string"),
            "country_code": Value("string"),
            "recording_date": Value("string"),
            "started_at": Value("string"),
            "raw_gcs_uri": Value("string"),
            "rights_status": Value("string"),
            "parakeet_model": Value("string"),
            "psst_model": Value("string"),
            "segment_duration_seconds": Value("float64"),
            "iu_index": Value("int32"),
            "start_seconds": Value("float64"),
            "end_seconds": Value("float64"),
            "duration_seconds": Value("float64"),
            "start_word_index": Value("int32"),
            "end_word_index_exclusive": Value("int32"),
            "boundary_after_word_index": Value("int32"),
            "alignment_score": Value("float64"),
            "alignment_method": Value("string"),
            "iu_word_count": Value("int32"),
            "speech_rate_wps": Value("float64"),
            "mean_energy_db": Value("float64"),
            "pause_before_ms": Value("float64"),
            "pause_after_ms": Value("float64"),
            "psst_boundary_strength": Value("string"),
            "words_json": Value("string"),
            "manifest_path": Value("string"),
        }
    )
    return features.arrow_schema


def empty_iu_columns() -> dict[str, list[Any]]:
    return {
        "audio": [],
        "text": [],
        "iu_id": [],
        "segment_id": [],
        "source_id": [],
        "source_name": [],
        "country_code": [],
        "recording_date": [],
        "started_at": [],
        "raw_gcs_uri": [],
        "rights_status": [],
        "parakeet_model": [],
        "psst_model": [],
        "segment_duration_seconds": [],
        "iu_index": [],
        "start_seconds": [],
        "end_seconds": [],
        "duration_seconds": [],
        "start_word_index": [],
        "end_word_index_exclusive": [],
        "boundary_after_word_index": [],
        "alignment_score": [],
        "alignment_method": [],
        "iu_word_count": [],
        "speech_rate_wps": [],
        "mean_energy_db": [],
        "pause_before_ms": [],
        "pause_after_ms": [],
        "psst_boundary_strength": [],
        "words_json": [],
        "manifest_path": [],
    }


def write_iu_shard(columns: dict[str, list[Any]], schema: Any, data_dir: Path, shard_index: int) -> int:
    import pyarrow as pa
    import pyarrow.parquet as pq

    if not columns["audio"]:
        return 0
    table = pa.Table.from_pydict(columns, schema=schema)
    path = data_dir / f"train-{shard_index:05d}.parquet"
    pq.write_table(table, path, compression="zstd")
    return table.num_rows


def encode_flac_clip(audio: Any, sample_rate: int) -> bytes:
    import soundfile as sf

    buf = io.BytesIO()
    sf.write(buf, audio, sample_rate, format="FLAC", subtype="PCM_16")
    return buf.getvalue()


def words_for_iu(row: dict[str, Any], boundary: dict[str, Any]) -> list[dict[str, Any]]:
    words = row.get("parakeet_words") or []
    start = int(boundary.get("start_word_index") or 0)
    end = int(boundary.get("end_word_index_exclusive") or start)
    if not words or end <= start:
        return []
    out = []
    for word in words[start:end]:
        out.append(
            {
                "word": word.get("word"),
                "start_seconds": word.get("start_seconds"),
                "end_seconds": word.get("end_seconds"),
                "word_index": word.get("word_index"),
                "timestamp_source": word.get("timestamp_source"),
            }
        )
    return out


def add_iu_row(
    columns: dict[str, list[Any]],
    row: dict[str, Any],
    boundary: dict[str, Any],
    audio: Any,
    sample_rate: int,
) -> int:
    import numpy as np

    text = re.sub(r"\s+", " ", str(boundary.get("text") or "")).strip()
    if not text:
        return 0

    segment_duration = float(row.get("duration_seconds") or 0.0)
    start_s = max(0.0, min(segment_duration, float(boundary.get("start_seconds") or 0.0)))
    end_s = max(start_s, min(segment_duration, float(boundary.get("end_seconds") or start_s)))
    if end_s - start_s < 0.04:
        end_s = min(segment_duration, start_s + 0.04)
    start_i = max(0, min(len(audio), int(round(start_s * sample_rate))))
    end_i = max(start_i + 1, min(len(audio), int(round(end_s * sample_rate))))
    clip = np.asarray(audio[start_i:end_i], dtype="float32")
    if clip.size == 0:
        return 0

    prosody = boundary.get("prosody") or {}
    iu_index = int(boundary.get("iu_index") or 0)
    iu_id = f"{row.get('segment_id')}#iu={iu_index:04d}"
    path = f"{row.get('source_id')}/{row.get('recording_date')}/{Path(str(row.get('audio_path_16k'))).stem}_iu{iu_index:04d}.flac"

    columns["audio"].append({"bytes": encode_flac_clip(clip, sample_rate), "path": path})
    columns["text"].append(text)
    columns["iu_id"].append(iu_id)
    columns["segment_id"].append(str(row.get("segment_id") or ""))
    columns["source_id"].append(str(row.get("source_id") or ""))
    columns["source_name"].append(str(row.get("source_name") or ""))
    columns["country_code"].append(str(row.get("country_code") or ""))
    columns["recording_date"].append(str(row.get("recording_date") or ""))
    columns["started_at"].append(str(row.get("started_at") or ""))
    columns["raw_gcs_uri"].append(str(row.get("raw_gcs_uri") or ""))
    columns["rights_status"].append(str(row.get("rights_status") or ""))
    columns["parakeet_model"].append(str(row.get("parakeet_model") or ""))
    columns["psst_model"].append(str(row.get("psst_model") or ""))
    columns["segment_duration_seconds"].append(segment_duration)
    columns["iu_index"].append(iu_index)
    columns["start_seconds"].append(round(start_s, 3))
    columns["end_seconds"].append(round(end_s, 3))
    columns["duration_seconds"].append(round(max(0.0, end_s - start_s), 3))
    columns["start_word_index"].append(int(boundary.get("start_word_index") or 0))
    columns["end_word_index_exclusive"].append(int(boundary.get("end_word_index_exclusive") or 0))
    after = boundary.get("boundary_after_word_index")
    columns["boundary_after_word_index"].append(int(after) if after is not None else -1)
    columns["alignment_score"].append(float(boundary.get("alignment_score") or 0.0))
    columns["alignment_method"].append(str(boundary.get("alignment_method") or ""))
    columns["iu_word_count"].append(int(prosody.get("iu_word_count") or 0))
    columns["speech_rate_wps"].append(float(prosody.get("speech_rate_wps") or 0.0))
    columns["mean_energy_db"].append(float(prosody.get("mean_energy_db") or 0.0))
    columns["pause_before_ms"].append(float(prosody.get("pause_before_ms") or 0.0))
    columns["pause_after_ms"].append(float(prosody.get("pause_after_ms") or 0.0))
    columns["psst_boundary_strength"].append(str(prosody.get("psst_boundary_strength") or ""))
    columns["words_json"].append(json.dumps(words_for_iu(row, boundary), ensure_ascii=False))
    columns["manifest_path"].append(str(row.get("manifest_path") or ""))
    return 1


def write_iu_dataset_card(output_dir: Path, summary: dict[str, Any]) -> None:
    ds_dir = output_dir / "hf_iu_dataset"
    sources = summary.get("sources") or []
    countries = summary.get("countries") or []
    readme = f"""---
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
license: other
pretty_name: English Sports Radio PSST Intonation Units
---

# English Sports Radio PSST Intonation Units

Each row is one PSST intonation unit split from the `!!!!!` boundary token. The `audio` column contains the corresponding clipped 16 kHz mono FLAC audio and is declared as a Hugging Face `Audio(sampling_rate=16000, num_channels=1)` feature in the parquet schema metadata.

## Contents

- IU rows: {summary['rows']}
- Source segment rows: {summary['segments']}
- Audio hours: {summary['total_hours']:.3f}
- Parquet shards: {summary['shards']}
- Embedded IU audio size: {summary['embedded_audio_bytes'] / 1_000_000_000:.3f} GB
- Countries: {", ".join(countries)}
- Sources: {", ".join(sources)}

## Main Columns

- `audio`: clipped IU audio bytes.
- `text`: PSST intonation-unit text.
- `segment_id`, `iu_index`, `start_seconds`, `end_seconds`: provenance and timing within the source segment.
- `pause_before_ms`, `pause_after_ms`, `mean_energy_db`, `speech_rate_wps`, `psst_boundary_strength`: prosodic labels.
- `words_json`: Parakeet word timestamps falling inside this IU.

## Rights

The broadcast rights status is unverified commercial radio. Keep this dataset private until licensing and redistribution rights are resolved.
"""
    (ds_dir / "README.md").write_text(readme, encoding="utf-8")


def package_iu_dataset(output_dir: Path, rows_per_shard: int, limit_segments: int = 0) -> dict[str, Any]:
    import soundfile as sf

    rows = load_output_rows(output_dir)
    if limit_segments:
        rows = rows[:limit_segments]
    ds_dir = output_dir / "hf_iu_dataset"
    data_tmp = ds_dir / "data.tmp"
    data_dir = ds_dir / "data"
    if data_tmp.exists():
        shutil.rmtree(data_tmp)
    data_tmp.mkdir(parents=True, exist_ok=True)

    schema = iu_features_schema()
    columns = empty_iu_columns()
    shard_index = 0
    iu_rows = 0
    total_seconds = 0.0
    embedded_audio_bytes = 0
    sources: set[str] = set()
    countries: set[str] = set()

    for segment_idx, row in enumerate(rows, start=1):
        audio_path = Path(str(row.get("audio_path_16k") or ""))
        if not audio_path.exists():
            log(f"missing stripped audio for {row.get('segment_id')}: {audio_path}")
            continue
        audio, sample_rate = sf.read(str(audio_path), dtype="float32", always_2d=False)
        if getattr(audio, "ndim", 1) > 1:
            import numpy as np

            audio = np.mean(audio, axis=1)
        sources.add(str(row.get("source_id") or ""))
        countries.add(str(row.get("country_code") or ""))
        for boundary in row.get("iu_boundaries") or []:
            before = len(columns["audio"])
            added = add_iu_row(columns, row, boundary, audio, int(sample_rate))
            if not added:
                continue
            embedded_audio_bytes += len(columns["audio"][-1]["bytes"])
            total_seconds += float(columns["duration_seconds"][-1])
            iu_rows += 1
            if len(columns["audio"]) >= rows_per_shard:
                written = write_iu_shard(columns, schema, data_tmp, shard_index)
                log(f"wrote IU shard {shard_index:05d} rows={written} total_iu_rows={iu_rows}")
                shard_index += 1
                columns = empty_iu_columns()
        if segment_idx % 100 == 0:
            log(f"IU packaging progress segments={segment_idx}/{len(rows)} iu_rows={iu_rows}")

    if columns["audio"]:
        written = write_iu_shard(columns, schema, data_tmp, shard_index)
        log(f"wrote IU shard {shard_index:05d} rows={written} total_iu_rows={iu_rows}")
        shard_index += 1

    if data_dir.exists():
        shutil.rmtree(data_dir)
    data_tmp.replace(data_dir)
    summary = {
        "rows": iu_rows,
        "segments": len(rows),
        "total_hours": total_seconds / 3600.0,
        "shards": shard_index,
        "embedded_audio_bytes": embedded_audio_bytes,
        "sources": sorted(s for s in sources if s),
        "countries": sorted(c for c in countries if c),
        "generated_at": utc_now(),
        "format": "iu-level parquet with Hugging Face Audio feature metadata",
    }
    write_json(ds_dir / "dataset_summary.json", summary)
    write_iu_dataset_card(output_dir, summary)
    return summary | {"dataset_dir": str(ds_dir.resolve())}


def clamp(value: Any, low: float, high: float) -> float:
    try:
        x = float(value)
    except (TypeError, ValueError):
        x = low
    return max(low, min(high, x))


def package_sampled_iu_dataset(
    output_dir: Path,
    per_station: int,
    top_pool_multiplier: int,
    seed: int,
) -> dict[str, Any]:
    import numpy as np
    import pandas as pd
    import pyarrow as pa
    import pyarrow.compute as pc
    import pyarrow.dataset as ds
    import pyarrow.parquet as pq

    source_dir = output_dir / "hf_iu_dataset" / "data"
    sampled_dir = output_dir / "hf_iu_sampled_dataset"
    data_tmp = sampled_dir / "data.tmp"
    data_dir = sampled_dir / "data"
    if data_tmp.exists():
        shutil.rmtree(data_tmp)
    data_tmp.mkdir(parents=True, exist_ok=True)

    dset = ds.dataset(str(source_dir), format="parquet")
    metadata_cols = [
        "iu_id",
        "source_id",
        "duration_seconds",
        "iu_word_count",
        "speech_rate_wps",
        "alignment_score",
        "mean_energy_db",
        "pause_before_ms",
        "pause_after_ms",
        "text",
    ]
    meta = dset.to_table(columns=metadata_cols).to_pandas()
    text_len = meta["text"].fillna("").str.len().clip(0, 120) / 120.0
    word_score = meta["iu_word_count"].fillna(0).clip(0, 24) / 24.0
    dur_score = meta["duration_seconds"].fillna(0).clip(0, 4.0) / 4.0
    align_score = meta["alignment_score"].fillna(0).clip(0, 1.0)
    energy_score = ((meta["mean_energy_db"].fillna(-60) + 45.0) / 30.0).clip(0, 1.0)
    rate = meta["speech_rate_wps"].fillna(0)
    rate_score = (1.0 - ((rate - 3.6).abs() / 4.0)).clip(0, 1.0)
    pause_penalty = (meta["pause_after_ms"].fillna(0).clip(0, 800) / 800.0) * 0.03
    meta["quality_score"] = (
        0.34 * word_score
        + 0.18 * dur_score
        + 0.18 * align_score
        + 0.14 * energy_score
        + 0.10 * rate_score
        + 0.06 * text_len
        - pause_penalty
    )

    rng = np.random.default_rng(seed)
    selected_ids: list[str] = []
    station_counts: dict[str, dict[str, Any]] = {}
    pool_size = max(per_station, per_station * top_pool_multiplier)
    for source_id, group in meta.groupby("source_id", sort=True):
        group = group.sort_values("quality_score", ascending=False)
        pool = group.head(min(len(group), pool_size))
        take = min(per_station, len(pool))
        if take < len(pool):
            chosen_idx = rng.choice(pool.index.to_numpy(), size=take, replace=False)
            chosen = pool.loc[chosen_idx].sort_values(["source_id", "quality_score"], ascending=[True, False])
        else:
            chosen = pool
        selected_ids.extend(chosen["iu_id"].tolist())
        station_counts[str(source_id)] = {
            "available_iu_rows": int(len(group)),
            "candidate_pool_rows": int(len(pool)),
            "selected_rows": int(take),
            "min_selected_quality_score": float(chosen["quality_score"].min()) if len(chosen) else 0.0,
            "mean_selected_quality_score": float(chosen["quality_score"].mean()) if len(chosen) else 0.0,
        }

    selected_set = set(selected_ids)
    selected_table = dset.to_table(filter=pc.field("iu_id").isin(selected_ids))
    selected_table = selected_table.filter(pc.is_in(selected_table["iu_id"], pa.array(selected_ids)))
    selected_df = selected_table.to_pandas()
    selected_df["_order"] = selected_df["iu_id"].map({iu_id: i for i, iu_id in enumerate(selected_ids)})
    selected_df = selected_df.sort_values("_order").drop(columns=["_order"])
    selected_table = pa.Table.from_pandas(selected_df, schema=iu_features_schema(), preserve_index=False)

    shard_path = data_tmp / "train-00000.parquet"
    pq.write_table(selected_table, shard_path, compression="zstd")
    if data_dir.exists():
        shutil.rmtree(data_dir)
    data_tmp.replace(data_dir)

    duration_col = selected_table["duration_seconds"].to_pylist()
    audio_col = selected_table["audio"].to_pylist()
    sources = sorted(set(selected_table["source_id"].to_pylist()))
    countries = sorted(set(selected_table["country_code"].to_pylist()))
    summary = {
        "rows": int(selected_table.num_rows),
        "target_rows_per_station": int(per_station),
        "stations": len(station_counts),
        "total_hours": float(sum(duration_col) / 3600.0),
        "shards": 1,
        "embedded_audio_bytes": int(sum(len(a["bytes"] or b"") for a in audio_col)),
        "sources": sources,
        "countries": countries,
        "station_counts": station_counts,
        "sample_seed": seed,
        "top_pool_multiplier": top_pool_multiplier,
        "selection_policy": (
            "For each station, rank IUs by a speech-richness/quality score "
            "(word count, duration, alignment score, energy, speech rate, text length), "
            "then sample without replacement from the top pool. Stations with fewer than "
            "the target rows contribute all available IUs."
        ),
        "generated_at": utc_now(),
        "format": "sampled IU-level parquet with Hugging Face Audio feature metadata",
    }
    write_json(sampled_dir / "dataset_summary.json", summary)
    write_sampled_iu_dataset_card(sampled_dir, summary)
    if len(selected_set) != selected_table.num_rows:
        log(f"warning: selected IDs={len(selected_set)} table rows={selected_table.num_rows}")
    return summary | {"dataset_dir": str(sampled_dir.resolve())}


def write_sampled_iu_dataset_card(sampled_dir: Path, summary: dict[str, Any]) -> None:
    station_lines = "\n".join(
        f"- `{source_id}`: {details['selected_rows']} selected / {details['available_iu_rows']} available"
        for source_id, details in sorted((summary.get("station_counts") or {}).items())
    )
    readme = f"""---
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
  - sampled
license: other
pretty_name: English Sports Radio PSST IU Sample
---

# English Sports Radio PSST IU Sample

This is a stratified, speech-rich IU-level sample of English sports radio. Each row is one PSST intonation unit split from the `!!!!!` boundary token, with its matching clipped 16 kHz mono FLAC audio in the `audio` column.

## Contents

- IU rows: {summary['rows']}
- Target rows per station: {summary['target_rows_per_station']}
- Audio hours: {summary['total_hours']:.3f}
- Parquet shards: {summary['shards']}
- Embedded audio size: {summary['embedded_audio_bytes'] / 1_000_000:.1f} MB
- Sample seed: {summary['sample_seed']}

## Station Counts

{station_lines}

## Selection

Rows were ranked per station by a speech-richness/quality score using IU word count, IU duration, Parakeet/PSST alignment score, energy, speech-rate plausibility, and text length. A deterministic random sample was drawn without replacement from each station's top-ranked pool. Stations with fewer than the target count contribute all available IU rows.

## Columns

- `audio`: clipped IU audio, declared as `Audio(sampling_rate=16000, num_channels=1)`.
- `text`: PSST intonation-unit text.
- `source_id`, `segment_id`, `iu_index`, `start_seconds`, `end_seconds`: provenance and timing.
- `pause_before_ms`, `pause_after_ms`, `mean_energy_db`, `speech_rate_wps`, `psst_boundary_strength`: prosodic labels.
- `words_json`: Parakeet word timestamps inside the IU.

## Rights

The broadcast rights status is unverified commercial radio. Keep this dataset private until licensing and redistribution rights are resolved.
"""
    (sampled_dir / "README.md").write_text(readme, encoding="utf-8")


def write_dataset_card(output_dir: Path, rows: list[dict[str, Any]], audio_bytes: int, shards: int) -> None:
    ds_dir = output_dir / "hf_dataset"
    ds_dir.mkdir(parents=True, exist_ok=True)
    total_hours = sum(float(r.get("duration_seconds") or 0.0) for r in rows) / 3600.0
    sources = sorted({str(r.get("source_id") or "") for r in rows if r.get("source_id")})
    countries = sorted({str(r.get("country_code") or "") for r in rows if r.get("country_code")})
    readme = f"""---
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
license: other
pretty_name: English Sports Radio with PSST IU Boundaries
---

# English Sports Radio with PSST IU Boundaries

This dataset contains English sports radio broadcast segments stripped to 16 kHz mono FLAC, transcribed locally with NVIDIA Parakeet, and annotated with PSST intonation-unit boundaries.

## Contents

- Rows: {len(rows)}
- Audio hours: {total_hours:.3f}
- Parquet shards: {shards}
- Embedded audio size: {audio_bytes / 1_000_000_000:.3f} GB
- Countries: {", ".join(countries) or "unknown"}
- Sources: {", ".join(sources) or "unknown"}

## Columns

- `audio`: 16 kHz mono FLAC bytes.
- `parakeet_text`: transcript from `{DEFAULT_PARAKEET_MODEL}`.
- `psst_text`: PSST output text containing IU separator tokens before splitting.
- `iu_boundaries_json`: JSON list of intonation units with start/end seconds, aligned word indices, alignment score, and prosodic pause/energy labels.
- `parakeet_words_json`: JSON list of word-level timestamps derived from Parakeet TDT token timing when available.
- `segment_prosody_json`: JSON segment-level energy and silence labels.
- `raw_gcs_uri` and `manifest_path`: provenance for the original 24 kHz / 24-bit capture.

## Rights

The broadcast rights status is unverified commercial radio. Keep this dataset private until licensing and redistribution rights are resolved.
"""
    (ds_dir / "README.md").write_text(readme, encoding="utf-8")
    info = {
        "rows": len(rows),
        "total_hours": total_hours,
        "shards": shards,
        "embedded_audio_bytes": audio_bytes,
        "sources": sources,
        "countries": countries,
        "generated_at": utc_now(),
    }
    write_json(ds_dir / "dataset_summary.json", info)


def upload_dataset(output_dir: Path, repo_id: str, private: bool) -> None:
    ds_dir = output_dir / "hf_dataset"
    if not ds_dir.exists():
        raise FileNotFoundError(f"Dataset directory does not exist: {ds_dir}")
    create_cmd = ["hf", "repos", "create", repo_id, "--type", "dataset", "--exist-ok"]
    if private:
        create_cmd.append("--private")
    proc = run_cmd(create_cmd, timeout=120)
    if proc.returncode != 0:
        raise RuntimeError(f"hf repo create failed: {proc.stderr[-1200:]}")
    upload_cmd = [
        "hf",
        "upload-large-folder",
        repo_id,
        str(ds_dir),
        "--type",
        "dataset",
        "--num-workers",
        "8",
    ]
    if private:
        upload_cmd.append("--private")
    proc = run_cmd(upload_cmd, timeout=None)
    if proc.returncode != 0:
        raise RuntimeError(f"hf upload failed: {proc.stderr[-2000:]}")


def cmd_inventory(args: argparse.Namespace) -> int:
    records = load_manifests(Path(args.manifest_root), set(args.source_id or []) or None)
    local = sum(1 for r in records if r.raw_local_path and r.raw_local_path.exists())
    gcs = sum(1 for r in records if r.raw_gcs_uri)
    by_source: dict[str, float] = defaultdict(float)
    for rec in records:
        by_source[rec.source_id] += rec.duration_seconds
    log(f"valid_segments={len(records)} local_audio={local} gcs_uris={gcs} hours={sum(by_source.values()) / 3600.0:.3f}")
    for source_id, seconds in sorted(by_source.items()):
        print(f"{source_id}\t{seconds / 3600.0:.3f}h")
    return 0


def cmd_status(args: argparse.Namespace) -> int:
    output_dir = Path(args.output_dir)
    rows = load_output_rows(output_dir)
    failures = sorted((output_dir / "failures").rglob("*.json")) if (output_dir / "failures").exists() else []
    packaged = sorted((output_dir / "hf_dataset" / "data").glob("*.parquet")) if (output_dir / "hf_dataset" / "data").exists() else []
    total_hours = sum(float(r.get("duration_seconds") or 0.0) for r in rows) / 3600.0
    log(f"processed_ok={len(rows)} failures={len(failures)} packaged_shards={len(packaged)} hours={total_hours:.3f}")
    return 0


def cmd_process(args: argparse.Namespace) -> int:
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    records = load_manifests(Path(args.manifest_root), set(args.source_id or []) or None)
    if args.local_first:
        local_records = [
            rec
            for rec in records
            if stripped_audio_path(output_dir, rec).exists()
            or raw_cache_path(output_dir, rec).exists()
            or (rec.raw_local_path is not None and rec.raw_local_path.exists())
        ]
        local_keys = {rec.segment_id for rec in local_records}
        remote_records = [rec for rec in records if rec.segment_id not in local_keys]
        records = interleave_records(local_records) + interleave_records(remote_records)
        log(f"local_first local_or_cached={len(local_records)} remote_later={len(remote_records)}")
    if args.limit:
        records = records[: args.limit]

    pending = [rec for rec in records if args.redo or not record_path(output_dir, rec).exists()]
    log(f"processing valid={len(records)} pending={len(pending)} output_dir={output_dir}")
    if not pending:
        if args.package:
            summary = package_dataset(output_dir, args.max_rows_per_shard)
            log(f"packaged {summary}")
        return 0

    parakeet = ParakeetRunner(args.parakeet_model, args.device)
    psst = None if args.skip_psst else PSSTRunner(args.psst_model, args.device)
    processed_since_package = 0
    ok = 0
    failed = 0
    for idx, rec in enumerate(pending, start=1):
        started = time.perf_counter()
        try:
            row = process_one(rec, output_dir, parakeet, psst, args)
            write_json(record_path(output_dir, rec), row)
            ok += 1
            processed_since_package += 1
            log(
                f"ok {idx}/{len(pending)} {rec.segment_id} "
                f"{row['duration_seconds']:.1f}s ius={len(row['iu_boundaries'])} "
                f"elapsed={time.perf_counter() - started:.1f}s"
            )
        except Exception as exc:  # noqa: BLE001
            failed += 1
            payload = {
                "status": "failed",
                "failed_at": utc_now(),
                "segment_id": rec.segment_id,
                "manifest_path": str(rec.manifest_path.resolve()),
                "error_type": type(exc).__name__,
                "error": str(exc),
            }
            write_json(failure_path(output_dir, rec), payload)
            log(f"failed {idx}/{len(pending)} {rec.segment_id}: {type(exc).__name__}: {exc}")
            if args.fail_fast:
                raise

        if args.package_every and processed_since_package >= args.package_every:
            summary = package_dataset(output_dir, args.max_rows_per_shard)
            log(f"packaged checkpoint {summary}")
            processed_since_package = 0
            if args.upload_repo:
                upload_dataset(output_dir, args.upload_repo, args.private)
                log(f"uploaded checkpoint to {args.upload_repo}")

    if args.package:
        summary = package_dataset(output_dir, args.max_rows_per_shard)
        log(f"packaged final {summary}")
        if args.upload_repo:
            upload_dataset(output_dir, args.upload_repo, args.private)
            log(f"uploaded final to {args.upload_repo}")
    log(f"done ok={ok} failed={failed}")
    return 0 if failed == 0 else 2


def prefetch_one(rec: SegmentManifest, output_dir: Path, timeout: int, overwrite: bool) -> dict[str, Any]:
    if rec.raw_local_path and rec.raw_local_path.exists():
        return {"status": "local", "segment_id": rec.segment_id}
    dest = raw_cache_path(output_dir, rec)
    if dest.exists() and dest.stat().st_size > 0 and not overwrite:
        return {"status": "cached", "segment_id": rec.segment_id}
    if not rec.raw_gcs_uri:
        return {"status": "skipped", "segment_id": rec.segment_id, "error": "missing gcs uri"}
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(".tmp.wav")
    gstmp = Path(str(tmp) + "_.gstmp")
    for path in (tmp, gstmp):
        if path.exists():
            path.unlink()
    proc = run_cmd(["gcloud", "storage", "cp", rec.raw_gcs_uri, str(tmp)], timeout=timeout)
    if proc.returncode != 0:
        return {
            "status": "failed",
            "segment_id": rec.segment_id,
            "error": (proc.stderr or proc.stdout)[-1000:],
        }
    tmp.replace(dest)
    return {"status": "ok", "segment_id": rec.segment_id, "bytes": dest.stat().st_size}


def cmd_prefetch(args: argparse.Namespace) -> int:
    output_dir = Path(args.output_dir)
    records = load_manifests(Path(args.manifest_root), set(args.source_id or []) or None)
    if args.missing_only:
        records = [
            rec
            for rec in records
            if not (rec.raw_local_path is not None and rec.raw_local_path.exists())
            and not raw_cache_path(output_dir, rec).exists()
        ]
    records = [rec for rec in records if rec.raw_gcs_uri]
    if args.limit:
        records = records[: args.limit]
    records = interleave_records(records)
    log(f"prefetch queued={len(records)} workers={args.workers} timeout={args.timeout}s")
    counts: dict[str, int] = defaultdict(int)
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = [pool.submit(prefetch_one, rec, output_dir, args.timeout, args.overwrite) for rec in records]
        for idx, future in enumerate(concurrent.futures.as_completed(futures), start=1):
            result = future.result()
            status = str(result.get("status") or "unknown")
            counts[status] += 1
            if status in {"ok", "failed"} or idx % 25 == 0:
                log(f"prefetch {idx}/{len(futures)} {status} {result.get('segment_id')} counts={dict(counts)}")
    log(f"prefetch done counts={dict(counts)}")
    return 0 if counts.get("failed", 0) == 0 else 2


def cmd_package(args: argparse.Namespace) -> int:
    summary = package_dataset(Path(args.output_dir), args.max_rows_per_shard)
    log(f"packaged {summary}")
    return 0


def cmd_package_ius(args: argparse.Namespace) -> int:
    summary = package_iu_dataset(Path(args.output_dir), args.rows_per_shard, args.limit_segments)
    log(f"packaged IU dataset {summary}")
    return 0


def cmd_sample_ius(args: argparse.Namespace) -> int:
    summary = package_sampled_iu_dataset(
        Path(args.output_dir),
        args.per_station,
        args.top_pool_multiplier,
        args.seed,
    )
    log(f"packaged sampled IU dataset {summary}")
    return 0


def cmd_upload(args: argparse.Namespace) -> int:
    upload_dataset(Path(args.output_dir), args.repo_id, args.private)
    log(f"uploaded to {args.repo_id}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build English sports radio ASR + PSST IU dataset.")
    sub = parser.add_subparsers(dest="cmd", required=True)

    inv = sub.add_parser("inventory")
    inv.add_argument("--manifest-root", default="runs/eng_sports_live/segments")
    inv.add_argument("--source-id", action="append")
    inv.set_defaults(func=cmd_inventory)

    status = sub.add_parser("status")
    status.add_argument("--output-dir", default="runs/eng_sports_processing")
    status.set_defaults(func=cmd_status)

    process = sub.add_parser("process")
    process.add_argument("--manifest-root", default="runs/eng_sports_live/segments")
    process.add_argument("--output-dir", default="runs/eng_sports_processing")
    process.add_argument("--source-id", action="append")
    process.add_argument("--limit", type=int, default=0)
    process.add_argument("--sample-rate", type=int, default=16000)
    process.add_argument("--parakeet-model", default=DEFAULT_PARAKEET_MODEL)
    process.add_argument("--psst-model", default=DEFAULT_PSST_MODEL)
    process.add_argument("--device", default="auto", choices=["auto", "cpu", "mps", "cuda"])
    process.add_argument("--psst-chunk-seconds", type=float, default=30.0)
    process.add_argument("--psst-max-new-tokens", type=int, default=256)
    process.add_argument("--skip-psst", action="store_true")
    process.add_argument("--keep-raw-cache", action="store_true")
    process.add_argument("--gcs-timeout", type=int, default=180)
    process.add_argument("--local-first", action="store_true")
    process.add_argument("--redo", action="store_true")
    process.add_argument("--fail-fast", action="store_true")
    process.add_argument("--package", action="store_true")
    process.add_argument("--package-every", type=int, default=0)
    process.add_argument("--max-rows-per-shard", type=int, default=200)
    process.add_argument("--upload-repo", default="")
    process.add_argument("--private", action=argparse.BooleanOptionalAction, default=True)
    process.set_defaults(func=cmd_process)

    prefetch = sub.add_parser("prefetch")
    prefetch.add_argument("--manifest-root", default="runs/eng_sports_live/segments")
    prefetch.add_argument("--output-dir", default="runs/eng_sports_processing")
    prefetch.add_argument("--source-id", action="append")
    prefetch.add_argument("--limit", type=int, default=0)
    prefetch.add_argument("--workers", type=int, default=12)
    prefetch.add_argument("--timeout", type=int, default=600)
    prefetch.add_argument("--missing-only", action=argparse.BooleanOptionalAction, default=True)
    prefetch.add_argument("--overwrite", action="store_true")
    prefetch.set_defaults(func=cmd_prefetch)

    package = sub.add_parser("package")
    package.add_argument("--output-dir", default="runs/eng_sports_processing")
    package.add_argument("--max-rows-per-shard", type=int, default=200)
    package.set_defaults(func=cmd_package)

    package_ius = sub.add_parser("package-ius")
    package_ius.add_argument("--output-dir", default="runs/eng_sports_processing")
    package_ius.add_argument("--rows-per-shard", type=int, default=10000)
    package_ius.add_argument("--limit-segments", type=int, default=0)
    package_ius.set_defaults(func=cmd_package_ius)

    sample_ius = sub.add_parser("sample-ius")
    sample_ius.add_argument("--output-dir", default="runs/eng_sports_processing")
    sample_ius.add_argument("--per-station", type=int, default=300)
    sample_ius.add_argument("--top-pool-multiplier", type=int, default=3)
    sample_ius.add_argument("--seed", type=int, default=20260530)
    sample_ius.set_defaults(func=cmd_sample_ius)

    upload = sub.add_parser("upload")
    upload.add_argument("--output-dir", default="runs/eng_sports_processing")
    upload.add_argument("--repo-id", default=DEFAULT_HF_REPO)
    upload.add_argument("--private", action=argparse.BooleanOptionalAction, default=True)
    upload.set_defaults(func=cmd_upload)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    exit_code = main()
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(exit_code)
