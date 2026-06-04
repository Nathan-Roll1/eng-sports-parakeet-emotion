from __future__ import annotations

import os
import re
import subprocess
from pathlib import Path

from dataset_builder.config import FFMPEG_BIN

_SILENCE_START_RE = re.compile(r"silence_start:\s*([\d.]+)")
_SILENCE_END_RE = re.compile(r"silence_end:\s*([\d.]+)")
_MAX_VOLUME_RE = re.compile(r"max_volume:\s*([-+]?[\d.]+)\s*dB")

_AUDIO_SUFFIXES = frozenset({".flac", ".wav", ".mp3"})


_DURATION_RE = re.compile(r"Duration:\s*(\d+):(\d+):(\d+)\.(\d+)")


def get_duration(filepath: str) -> float:
    r = subprocess.run(
        [FFMPEG_BIN, "-hide_banner", "-i", filepath, "-f", "null", "-"],
        capture_output=True,
    )
    text = (r.stderr or b"").decode(errors="replace")
    m = _DURATION_RE.search(text)
    if not m:
        return 0.0
    h, mn, s, cs = int(m.group(1)), int(m.group(2)), int(m.group(3)), int(m.group(4))
    return h * 3600 + mn * 60 + s + cs / 100.0


def get_peak_amplitude(filepath: str) -> float:
    r = subprocess.run(
        [
            FFMPEG_BIN,
            "-hide_banner",
            "-nostats",
            "-i",
            filepath,
            "-af",
            "volumedetect",
            "-f",
            "null",
            "-",
        ],
        capture_output=True,
    )
    text = (r.stderr or b"").decode(errors="replace")
    m = _MAX_VOLUME_RE.search(text)
    if not m:
        return 0.0
    try:
        db = float(m.group(1))
    except ValueError:
        return 0.0
    return float(10 ** (db / 20.0))


def detect_silence(
    filepath: str, threshold_db: float = -50.0, min_duration: float = 2.0
) -> list[tuple[float, float]]:
    af = f"silencedetect=noise={threshold_db}dB:d={min_duration}"
    r = subprocess.run(
        [
            FFMPEG_BIN,
            "-hide_banner",
            "-nostats",
            "-i",
            filepath,
            "-af",
            af,
            "-f",
            "null",
            "-",
        ],
        capture_output=True,
    )
    text = (r.stderr or b"").decode(errors="replace")
    intervals: list[tuple[float, float]] = []
    pending_start: float | None = None
    for line in text.splitlines():
        ms = _SILENCE_START_RE.search(line)
        if ms is not None:
            pending_start = float(ms.group(1))
            continue
        me = _SILENCE_END_RE.search(line)
        if me is not None and pending_start is not None:
            end = float(me.group(1))
            intervals.append((pending_start, end))
            pending_start = None
    if pending_start is not None:
        dur = get_duration(filepath)
        if dur > pending_start:
            intervals.append((pending_start, dur))
    return intervals


def validate_recording(
    filepath: str, min_duration: float = 1700.0, max_silence_ratio: float = 0.5
) -> tuple[bool, str]:
    p = Path(filepath)
    if not p.is_file():
        return False, "file not found"
    if p.stat().st_size <= 0:
        return False, "empty file"

    total_duration = get_duration(filepath)
    if total_duration <= 0:
        return False, "could not read duration"
    if total_duration < min_duration:
        return False, f"duration {total_duration:.1f}s below minimum {min_duration}s"

    silence_segments = detect_silence(filepath)
    total_silence = sum(max(0.0, e - s) for s, e in silence_segments)
    silence_ratio = total_silence / total_duration if total_duration > 0 else 1.0
    if silence_ratio > max_silence_ratio:
        return (
            False,
            f"silence ratio {silence_ratio:.2f} exceeds max {max_silence_ratio}",
        )

    peak = get_peak_amplitude(filepath)
    if peak < 0.001:
        return False, f"peak amplitude {peak:.6f} below threshold"

    return True, "ok"


def batch_validate(
    directory: str, min_duration: float = 1700.0, max_silence_ratio: float = 0.5
) -> dict:
    results: list[dict] = []
    for dirpath, _, filenames in os.walk(directory):
        for name in filenames:
            if Path(name).suffix.lower() not in _AUDIO_SUFFIXES:
                continue
            fp = str(Path(dirpath) / name)
            valid, reason = validate_recording(
                fp, min_duration=min_duration, max_silence_ratio=max_silence_ratio
            )
            results.append({"filepath": fp, "valid": valid, "reason": reason})
    results.sort(key=lambda x: x["filepath"])
    valid_n = sum(1 for r in results if r["valid"])
    return {
        "total": len(results),
        "valid": valid_n,
        "invalid": len(results) - valid_n,
        "results": results,
    }
