from __future__ import annotations

import re
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path

from dataset_builder.config import (
    FFMPEG_BIN,
    RecordingConfig,
    RecordingResult,
    StationInfo,
    get_output_path,
)

_DURATION_RE = re.compile(r"Duration:\s*(\d+):(\d+):(\d+)\.(\d+)")


def _iso_timestamp() -> str:
    return datetime.now(timezone.utc).isoformat()


def _estimate_duration(filepath: str) -> float:
    try:
        proc = subprocess.run(
            [FFMPEG_BIN, "-hide_banner", "-i", filepath, "-f", "null", "-"],
            stdin=subprocess.DEVNULL,
            capture_output=True,
            timeout=60,
            check=False,
        )
        text = (proc.stderr or b"").decode(errors="replace")
        m = _DURATION_RE.search(text)
        if not m:
            return 0.0
        h, mn, s, cs = int(m.group(1)), int(m.group(2)), int(m.group(3)), int(m.group(4))
        return h * 3600 + mn * 60 + s + cs / 100.0
    except (OSError, subprocess.SubprocessError, ValueError):
        return 0.0


_STREAM_RE = re.compile(r"Stream #\d+:\d+.*Audio:\s*(\w+).*?(\d+)\s*Hz")
_BITRATE_RE = re.compile(r"bitrate:\s*(\d+)\s*kb/s")


def probe_stream(url: str, timeout: float = 10.0) -> dict:
    if not (url or "").strip():
        return {}
    try:
        proc = subprocess.run(
            [FFMPEG_BIN, "-hide_banner", "-i", url.strip(), "-f", "null", "-"],
            stdin=subprocess.DEVNULL,
            capture_output=True,
            timeout=timeout,
            check=False,
        )
        text = (proc.stderr or b"").decode(errors="replace")
        out: dict = {"codec": None, "bitrate": None, "sample_rate": None}
        sm = _STREAM_RE.search(text)
        if sm:
            out["codec"] = sm.group(1)
            out["sample_rate"] = int(sm.group(2))
        bm = _BITRATE_RE.search(text)
        if bm:
            out["bitrate"] = int(bm.group(1)) * 1000
        return out
    except (OSError, subprocess.SubprocessError):
        return {}


def _stderr_message(stderr: bytes | str | None, fallback: str) -> str:
    if stderr is None:
        return fallback
    if isinstance(stderr, bytes):
        text = stderr.decode(errors="replace").strip()
    else:
        text = str(stderr).strip()
    return text or fallback


def record_station(station: StationInfo, config: RecordingConfig) -> RecordingResult:
    started_at = _iso_timestamp()
    wall_t0 = time.perf_counter()
    out_path = get_output_path(station, config)
    out_str = str(out_path)
    stream_url = (station.url_resolved or "").strip() or (station.url or "").strip()

    def base_failure(
        error: str,
        *,
        filepath: str | None = None,
        duration_actual: float = 0.0,
        file_size_bytes: int = 0,
    ) -> RecordingResult:
        return RecordingResult(
            station=station,
            filepath=filepath,
            duration_actual=duration_actual,
            file_size_bytes=file_size_bytes,
            started_at=started_at,
            completed_at=_iso_timestamp(),
            success=False,
            error=error,
            avg_bitrate_kbps=0.0,
            silence_ratio=0.0,
            peak_amplitude=0.0,
        )

    if not stream_url:
        return base_failure("empty stream URL")

    try:
        out_path.parent.mkdir(parents=True, exist_ok=True)
    except OSError as e:
        return base_failure(f"cannot create output directory: {e}")

    cmd = [
        FFMPEG_BIN,
        "-reconnect",
        "1",
        "-reconnect_streamed",
        "1",
        "-reconnect_delay_max",
        "5",
        "-i",
        stream_url,
        "-t",
        str(config.duration_seconds),
        "-f",
        "flac",
        "-ar",
        str(config.sample_rate),
        "-ac",
        str(config.channels),
        "-loglevel",
        "error",
        "-y",
        out_str,
    ]

    try:
        completed = subprocess.run(
            cmd,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
            timeout=float(config.ffmpeg_timeout),
            check=False,
        )
    except subprocess.TimeoutExpired as e:
        err = _stderr_message(
            e.stderr, f"ffmpeg timed out after {config.ffmpeg_timeout}s"
        )
        fp = out_str if Path(out_str).is_file() else None
        return base_failure(err, filepath=fp)
    except (OSError, subprocess.SubprocessError) as e:
        return base_failure(str(e))

    if completed.returncode != 0:
        err = _stderr_message(
            completed.stderr, f"ffmpeg exited with code {completed.returncode}"
        )
        fp = out_str if Path(out_str).is_file() else None
        return base_failure(err, filepath=fp)

    path = Path(out_str)
    if not path.is_file():
        return base_failure("output file missing after ffmpeg")

    try:
        file_size_bytes = path.stat().st_size
    except OSError as e:
        return base_failure(f"cannot stat output file: {e}", filepath=out_str)

    duration_actual = _estimate_duration(out_str)
    if duration_actual <= 0:
        duration_actual = max(time.perf_counter() - wall_t0, 1e-6)

    avg_bitrate_kbps = (file_size_bytes * 8.0) / (duration_actual * 1000.0)

    return RecordingResult(
        station=station,
        filepath=out_str,
        duration_actual=duration_actual,
        file_size_bytes=file_size_bytes,
        started_at=started_at,
        completed_at=_iso_timestamp(),
        success=True,
        error=None,
        avg_bitrate_kbps=avg_bitrate_kbps,
        silence_ratio=0.0,
        peak_amplitude=0.0,
    )
