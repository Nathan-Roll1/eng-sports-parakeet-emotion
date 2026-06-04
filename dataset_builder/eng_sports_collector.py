from __future__ import annotations

import argparse
import concurrent.futures
import dataclasses
import json
import os
import shutil
import signal
import subprocess
import sys
import time
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


@dataclasses.dataclass(frozen=True)
class SportsSource:
    id: str
    name: str
    country_code: str
    homepage: str
    source_page: str
    stream_url: str
    content_type: str
    language: str
    discovered_via: str
    technical_status: str
    rights_status: str
    rights_notes: str
    probe: dict[str, Any]


@dataclasses.dataclass(frozen=True)
class CollectorConfig:
    bucket: str
    work_dir: Path
    duration_seconds: int
    target_sample_rate: int
    min_source_sample_rate: int
    target_channels: int
    upload: bool
    delete_after_upload: bool
    max_local_gb: float
    min_duration_ratio: float
    probe_timeout: int


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def stamp_for_path() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def date_part() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%d")


def load_sources(path: Path) -> list[SportsSource]:
    raw = json.loads(path.read_text(encoding="utf-8"))
    return [SportsSource(**row) for row in raw]


def ffprobe(target: str, timeout: int = 20, remote: bool = False) -> dict[str, Any]:
    cmd = [
        FFPROBE_BIN,
        "-hide_banner",
        "-v",
        "error",
    ]
    if remote:
        cmd += ["-rw_timeout", str(timeout * 1_000_000)]
    cmd += [
        "-select_streams",
        "a:0",
        "-show_entries",
        "stream=codec_name,codec_type,sample_rate,channels,channel_layout,bit_rate,bits_per_raw_sample,bits_per_sample,sample_fmt:format=format_name,bit_rate,duration",
        "-of",
        "json",
        target,
    ]
    started = time.perf_counter()
    try:
        proc = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=timeout + 5,
            check=False,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        return {
            "ok": False,
            "error": str(exc),
            "elapsed_seconds": round(time.perf_counter() - started, 3),
            "command": cmd,
        }
    payload: dict[str, Any] = {}
    if proc.stdout.strip():
        try:
            payload = json.loads(proc.stdout)
        except json.JSONDecodeError as exc:
            payload = {"parse_error": str(exc), "stdout": proc.stdout[:1000]}
    payload["ok"] = proc.returncode == 0
    payload["stderr"] = proc.stderr.strip()[:2000]
    payload["elapsed_seconds"] = round(time.perf_counter() - started, 3)
    payload["command"] = cmd
    return payload


def first_audio_stream(probe: dict[str, Any]) -> dict[str, Any]:
    for stream in probe.get("streams") or []:
        if stream.get("codec_type") == "audio":
            return stream
    return {}


def parse_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def parse_int(value: Any, default: int = 0) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def source_quality_gate(
    source: SportsSource,
    source_probe: dict[str, Any],
    min_sample_rate: int,
) -> tuple[bool, str, int]:
    stream = first_audio_stream(source_probe) if source_probe.get("ok") else {}
    sample_rate = parse_int(stream.get("sample_rate"))
    if sample_rate <= 0:
        sample_rate = parse_int(source.probe.get("sample_rate_hz"))
    if sample_rate <= 0:
        return False, "source sample rate could not be established", sample_rate
    if sample_rate < min_sample_rate:
        return (
            False,
            f"source sample_rate={sample_rate}, below minimum {min_sample_rate}",
            sample_rate,
        )
    return True, "ok", sample_rate


def local_segment_bytes(work_dir: Path) -> int:
    total = 0
    segments = work_dir / "segments"
    if not segments.exists():
        return 0
    for path in segments.rglob("*"):
        if path.is_file():
            try:
                total += path.stat().st_size
            except OSError:
                pass
    return total


def ensure_bucket_uri(bucket: str) -> str:
    if bucket.startswith("gs://"):
        return bucket.rstrip("/")
    return f"gs://{bucket.strip('/')}"


def gcs_uri(bucket: str, prefix: str, source_id: str, local_path: Path) -> str:
    return (
        f"{ensure_bucket_uri(bucket)}/{prefix}/source_id={source_id}/"
        f"date={date_part()}/{local_path.name}"
    )


def upload_file(
    local_path: Path,
    uri: str,
    timeout: int = 300,
    no_clobber: bool = True,
) -> dict[str, Any]:
    cmd = [
        "gcloud",
        "storage",
        "cp",
        str(local_path),
        uri,
    ]
    if no_clobber:
        cmd.insert(3, "--no-clobber")
    started = time.perf_counter()
    try:
        proc = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=timeout,
            check=False,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        return {
            "ok": False,
            "error": str(exc),
            "elapsed_seconds": round(time.perf_counter() - started, 3),
            "command": cmd,
        }
    return {
        "ok": proc.returncode == 0,
        "stdout": proc.stdout.strip()[-2000:],
        "stderr": proc.stderr.strip()[-4000:],
        "elapsed_seconds": round(time.perf_counter() - started, 3),
        "command": cmd,
    }


def validate_output(
    output_probe: dict[str, Any],
    expected_duration: int,
    target_sample_rate: int,
    min_duration_ratio: float,
) -> tuple[bool, str]:
    if not output_probe.get("ok"):
        return False, output_probe.get("error") or output_probe.get("stderr") or "ffprobe failed"
    stream = first_audio_stream(output_probe)
    sample_rate = int(stream.get("sample_rate") or 0)
    codec = str(stream.get("codec_name") or "")
    bits = int(stream.get("bits_per_sample") or stream.get("bits_per_raw_sample") or 0)
    duration = parse_float((output_probe.get("format") or {}).get("duration"))
    if sample_rate != target_sample_rate:
        return False, f"sample_rate={sample_rate}, expected {target_sample_rate}"
    if codec != "pcm_s24le" and bits < 24:
        return False, f"codec={codec}, bits_per_sample={bits}; expected 24-bit PCM"
    min_duration = expected_duration * min_duration_ratio
    if duration and duration < min_duration:
        return False, f"duration={duration:.2f}s below minimum {min_duration:.2f}s"
    return True, "ok"


def record_one(source: SportsSource, cfg: CollectorConfig) -> dict[str, Any]:
    started_at = utc_now()
    source_dir = cfg.work_dir / "segments" / source.id / date_part()
    source_dir.mkdir(parents=True, exist_ok=True)
    basename = f"{source.id}_{stamp_for_path()}_24k_s24le.wav"
    audio_path = source_dir / basename
    metadata_path = audio_path.with_suffix(".json")

    source_probe = ffprobe(source.stream_url, timeout=cfg.probe_timeout, remote=True)
    source_ok, source_message, source_sample_rate = source_quality_gate(
        source,
        source_probe,
        cfg.min_source_sample_rate,
    )
    if not source_ok:
        metadata_path.write_text(
            json.dumps(
                {
                    "source": dataclasses.asdict(source),
                    "capture": {
                        "started_at": started_at,
                        "completed_at": utc_now(),
                        "duration_requested_seconds": cfg.duration_seconds,
                        "elapsed_seconds": 0.0,
                        "ffmpeg_returncode": None,
                        "ffmpeg_error": source_message,
                        "ffmpeg_command": [],
                    },
                    "local": {
                        "audio_path": str(audio_path),
                        "metadata_path": str(metadata_path),
                        "file_size_bytes": 0,
                    },
                    "quality": {
                        "source_probe": source_probe,
                        "source_sample_rate_hz": source_sample_rate,
                        "source_quality_valid": False,
                        "source_quality_message": source_message,
                        "output_probe": {},
                        "valid": False,
                        "validation_message": source_message,
                        "target_sample_rate_hz": cfg.target_sample_rate,
                        "target_bit_depth": 24,
                        "target_channels": cfg.target_channels,
                    },
                    "gcs": {
                        "bucket": ensure_bucket_uri(cfg.bucket),
                        "audio_uri": gcs_uri(cfg.bucket, "raw_24k_s24le", source.id, audio_path),
                        "metadata_uri": gcs_uri(cfg.bucket, "manifests", source.id, metadata_path),
                        "audio_upload": {"ok": False, "skipped": True},
                        "metadata_upload": {"ok": False, "skipped": True},
                    },
                },
                indent=2,
            ),
            encoding="utf-8",
        )
        return json.loads(metadata_path.read_text(encoding="utf-8"))

    cmd = [
        FFMPEG_BIN,
        "-hide_banner",
        "-nostdin",
        "-loglevel",
        "warning",
        "-reconnect",
        "1",
        "-reconnect_streamed",
        "1",
        "-reconnect_at_eof",
        "1",
        "-reconnect_on_network_error",
        "1",
        "-reconnect_on_http_error",
        "429,500,502,503,504",
        "-reconnect_delay_max",
        "30",
        "-re",
        "-i",
        source.stream_url,
        "-t",
        str(cfg.duration_seconds),
        "-map",
        "0:a:0",
        "-vn",
        "-ac",
        str(cfg.target_channels),
        "-ar",
        str(cfg.target_sample_rate),
        "-c:a",
        "pcm_s24le",
        "-f",
        "wav",
        "-y",
        str(audio_path),
    ]

    record_started = time.perf_counter()
    try:
        proc = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=cfg.duration_seconds + 90,
            check=False,
        )
        record_error = None if proc.returncode == 0 else proc.stderr.strip()[-4000:]
    except (OSError, subprocess.SubprocessError) as exc:
        proc = None
        record_error = str(exc)

    completed_at = utc_now()
    file_size = audio_path.stat().st_size if audio_path.exists() else 0
    output_probe = ffprobe(str(audio_path), timeout=cfg.probe_timeout) if file_size else {}
    valid, validation_message = validate_output(
        output_probe,
        cfg.duration_seconds,
        cfg.target_sample_rate,
        cfg.min_duration_ratio,
    )

    audio_uri = gcs_uri(cfg.bucket, "raw_24k_s24le", source.id, audio_path)
    metadata_uri = gcs_uri(cfg.bucket, "manifests", source.id, metadata_path)
    audio_upload: dict[str, Any] = {"ok": False, "skipped": not cfg.upload}
    metadata_upload: dict[str, Any] = {"ok": False, "skipped": not cfg.upload}

    record = {
        "source": dataclasses.asdict(source),
        "capture": {
            "started_at": started_at,
            "completed_at": completed_at,
            "duration_requested_seconds": cfg.duration_seconds,
            "elapsed_seconds": round(time.perf_counter() - record_started, 3),
            "ffmpeg_returncode": proc.returncode if proc else None,
            "ffmpeg_error": record_error,
            "ffmpeg_command": cmd,
        },
        "local": {
            "audio_path": str(audio_path),
            "metadata_path": str(metadata_path),
            "file_size_bytes": file_size,
        },
        "quality": {
            "source_probe": source_probe,
            "source_sample_rate_hz": source_sample_rate,
            "source_quality_valid": source_ok,
            "source_quality_message": source_message,
            "output_probe": output_probe,
            "valid": valid,
            "validation_message": validation_message,
            "target_sample_rate_hz": cfg.target_sample_rate,
            "target_bit_depth": 24,
            "target_channels": cfg.target_channels,
            "native_bit_depth_note": "MP3/AAC/HLS radio streams generally do not expose a PCM bit depth; this file is a decoded 24-bit PCM capture, not proof of native 24-bit source fidelity.",
        },
        "gcs": {
            "bucket": ensure_bucket_uri(cfg.bucket),
            "audio_uri": audio_uri,
            "metadata_uri": metadata_uri,
            "audio_upload": audio_upload,
            "metadata_upload": metadata_upload,
        },
    }

    metadata_path.write_text(json.dumps(record, indent=2), encoding="utf-8")

    if cfg.upload and valid and file_size:
        audio_upload = upload_file(audio_path, audio_uri, no_clobber=True)
        metadata_upload = upload_file(metadata_path, metadata_uri, no_clobber=False)
        record["gcs"]["audio_upload"] = audio_upload
        record["gcs"]["metadata_upload"] = metadata_upload
        metadata_path.write_text(json.dumps(record, indent=2), encoding="utf-8")
        if metadata_upload.get("ok"):
            record["gcs"]["metadata_upload"] = upload_file(
                metadata_path,
                metadata_uri,
                no_clobber=False,
            )
            metadata_path.write_text(json.dumps(record, indent=2), encoding="utf-8")
        if (
            cfg.delete_after_upload
            and audio_upload.get("ok")
            and record["gcs"]["metadata_upload"].get("ok")
        ):
            audio_path.unlink(missing_ok=True)

    return record


def iter_manifest_paths(work_dir: Path) -> list[Path]:
    segments = work_dir / "segments"
    if not segments.exists():
        return []
    return sorted(segments.rglob("*.json"))


def retry_pending_uploads(cfg: CollectorConfig, limit: int | None = None) -> dict[str, int]:
    stats = {"seen": 0, "attempted": 0, "uploaded": 0, "failed": 0, "skipped": 0}
    for metadata_path in iter_manifest_paths(cfg.work_dir):
        if limit is not None and stats["attempted"] >= limit:
            break
        stats["seen"] += 1
        try:
            record = json.loads(metadata_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            stats["skipped"] += 1
            continue
        if not (record.get("quality") or {}).get("valid"):
            stats["skipped"] += 1
            continue
        gcs = record.get("gcs") or {}
        audio_ok = bool((gcs.get("audio_upload") or {}).get("ok"))
        metadata_ok = bool((gcs.get("metadata_upload") or {}).get("ok"))
        if audio_ok and metadata_ok:
            stats["uploaded"] += 1
            continue

        local = record.get("local") or {}
        audio_path = Path(local.get("audio_path") or "")
        if not audio_path.is_file():
            stats["skipped"] += 1
            continue

        stats["attempted"] += 1
        record.setdefault("retry", {})["last_retry_at"] = utc_now()
        if not audio_ok:
            record["gcs"]["audio_upload"] = upload_file(
                audio_path,
                record["gcs"]["audio_uri"],
                no_clobber=True,
            )
        metadata_path.write_text(json.dumps(record, indent=2), encoding="utf-8")
        record["gcs"]["metadata_upload"] = upload_file(
            metadata_path,
            record["gcs"]["metadata_uri"],
            no_clobber=False,
        )
        metadata_path.write_text(json.dumps(record, indent=2), encoding="utf-8")
        if record["gcs"]["metadata_upload"].get("ok"):
            record["gcs"]["metadata_upload"] = upload_file(
                metadata_path,
                record["gcs"]["metadata_uri"],
                no_clobber=False,
            )
            metadata_path.write_text(json.dumps(record, indent=2), encoding="utf-8")

        if (
            record["gcs"]["audio_upload"].get("ok")
            and record["gcs"]["metadata_upload"].get("ok")
        ):
            stats["uploaded"] += 1
            if cfg.delete_after_upload:
                audio_path.unlink(missing_ok=True)
        else:
            stats["failed"] += 1
    return stats


def run_batch(sources: list[SportsSource], cfg: CollectorConfig, parallel: int) -> list[dict[str, Any]]:
    results: list[dict[str, Any]] = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=parallel) as pool:
        futs = [pool.submit(record_one, source, cfg) for source in sources]
        for fut in concurrent.futures.as_completed(futs):
            row = fut.result()
            results.append(row)
            status = "ok" if row["quality"]["valid"] else "failed"
            upload = row["gcs"]["audio_upload"]
            upload_status = "uploaded" if upload.get("ok") else "upload pending"
            print(
                f"{utc_now()} {status} {row['source']['id']} "
                f"{row['local']['file_size_bytes']} bytes; {upload_status}",
                flush=True,
            )
    return results


def choose_sources(
    sources: list[SportsSource],
    source_ids: list[str] | None,
    allow_unverified_rights: bool,
) -> list[SportsSource]:
    chosen = sources
    if source_ids:
        wanted = set(source_ids)
        chosen = [source for source in chosen if source.id in wanted]
    if not allow_unverified_rights:
        chosen = [
            source
            for source in chosen
            if source.rights_status in {"licensed", "public-domain", "open-license"}
        ]
    return chosen


def build_parser() -> argparse.ArgumentParser:
    default_sources = Path(__file__).with_name("eng_sports_sources.json")
    p = argparse.ArgumentParser(description="Collect English sports radio audio to GCS.")
    p.add_argument("--sources", type=Path, default=default_sources)
    p.add_argument("--source-id", action="append", default=None)
    p.add_argument("--bucket", default="eng_sports")
    p.add_argument("--work-dir", type=Path, default=Path("eng_sports_spool"))
    p.add_argument("--duration", type=int, default=300)
    p.add_argument("--parallel", type=int, default=2)
    p.add_argument("--sample-rate", type=int, default=24000)
    p.add_argument("--min-source-sample-rate", type=int, default=24000)
    p.add_argument("--channels", type=int, default=1)
    p.add_argument("--continuous", action="store_true")
    p.add_argument("--max-segments", type=int, default=None)
    p.add_argument("--skip-upload", action="store_true")
    p.add_argument("--delete-after-upload", action="store_true")
    p.add_argument("--retry-pending", action="store_true")
    p.add_argument("--retry-pending-only", action="store_true")
    p.add_argument("--retry-limit", type=int, default=None)
    p.add_argument("--max-local-gb", type=float, default=5.0)
    p.add_argument("--min-duration-ratio", type=float, default=0.75)
    p.add_argument("--probe-timeout", type=int, default=20)
    p.add_argument(
        "--allow-unverified-rights",
        action="store_true",
        help="Allow technically public streams whose collection rights have not been verified.",
    )
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    all_sources = load_sources(args.sources)
    sources = choose_sources(
        all_sources,
        args.source_id,
        allow_unverified_rights=args.allow_unverified_rights,
    )
    if not sources and not args.retry_pending_only:
        print(
            "No sources selected. The bundled sports streams have unverified audio rights; "
            "pass --allow-unverified-rights only after confirming your intended use is permitted.",
            file=sys.stderr,
        )
        return 2

    cfg = CollectorConfig(
        bucket=args.bucket,
        work_dir=args.work_dir,
        duration_seconds=args.duration,
        target_sample_rate=args.sample_rate,
        min_source_sample_rate=args.min_source_sample_rate,
        target_channels=args.channels,
        upload=not args.skip_upload,
        delete_after_upload=args.delete_after_upload,
        max_local_gb=args.max_local_gb,
        min_duration_ratio=args.min_duration_ratio,
        probe_timeout=args.probe_timeout,
    )
    cfg.work_dir.mkdir(parents=True, exist_ok=True)

    if args.retry_pending or args.retry_pending_only:
        stats = retry_pending_uploads(cfg, limit=args.retry_limit)
        print(f"{utc_now()} pending upload retry: {stats}", flush=True)
        if args.retry_pending_only:
            return 0 if stats["failed"] == 0 else 1

    stop = False

    def request_stop(signum: int, _frame: Any) -> None:
        nonlocal stop
        stop = True
        print(f"{utc_now()} received signal {signum}; stopping after current batch", flush=True)

    signal.signal(signal.SIGTERM, request_stop)
    signal.signal(signal.SIGINT, request_stop)

    print(
        f"{utc_now()} starting eng_sports collector: sources={len(sources)} "
        f"duration={cfg.duration_seconds}s parallel={args.parallel} "
        f"bucket={ensure_bucket_uri(cfg.bucket)} upload={cfg.upload}",
        flush=True,
    )

    completed = 0
    while True:
        cap_bytes = int(cfg.max_local_gb * (1024 ** 3))
        if cfg.upload:
            stats = retry_pending_uploads(cfg, limit=args.retry_limit)
            if stats["attempted"]:
                print(f"{utc_now()} pending upload retry: {stats}", flush=True)
        pending_bytes = local_segment_bytes(cfg.work_dir)
        if cap_bytes > 0 and pending_bytes >= cap_bytes:
            print(
                f"{utc_now()} local spool is {pending_bytes} bytes, "
                f"above cap {cap_bytes}; sleeping before next batch",
                flush=True,
            )
            time.sleep(60)
            if stop:
                break
            continue

        remaining = None if args.max_segments is None else args.max_segments - completed
        if remaining is not None and remaining <= 0:
            break
        batch_sources = sources if remaining is None else sources[:remaining]
        results = run_batch(batch_sources, cfg, max(1, int(args.parallel)))
        completed += len(results)

        if stop or not args.continuous:
            break

    print(f"{utc_now()} collector stopped after {completed} segment attempts", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
