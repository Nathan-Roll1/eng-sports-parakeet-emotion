import glob
import math
import os
import re
import subprocess
from concurrent.futures import ThreadPoolExecutor

from dataset_builder.config import FFMPEG_BIN, RecordingConfig

_AUDIO_EXTENSIONS = frozenset({
    ".wav",
    ".wave",
    ".mp3",
    ".flac",
    ".m4a",
    ".aac",
    ".ogg",
    ".opus",
    ".wma",
    ".aiff",
    ".aif",
    ".caf",
})


def _stderr_text(result: subprocess.CompletedProcess) -> str:
    err = result.stderr
    if not err:
        return ""
    if isinstance(err, bytes):
        return err.decode("utf-8", errors="replace")
    return str(err)


def normalize_audio(input_path: str, output_path: str, config: RecordingConfig) -> bool:
    out_dir = os.path.dirname(output_path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    cmd = [
        FFMPEG_BIN,
        "-y",
        "-i",
        input_path,
        "-af",
        "loudnorm=I=-23:TP=-1:LRA=11",
        "-ar",
        str(config.sample_rate),
        "-ac",
        str(config.channels),
        "-f",
        config.output_format,
        output_path,
    ]
    r = subprocess.run(cmd, capture_output=True)
    return r.returncode == 0


def _parse_duration_seconds(text: str) -> float:
    m = re.search(r"Duration:\s*(\d{2}):(\d{2}):(\d{2}\.\d+)", text)
    if not m:
        return 0.0
    h, mi, s = m.groups()
    return int(h) * 3600 + int(mi) * 60 + float(s)


def _parse_sample_rate_channels(text: str) -> tuple[int, int]:
    for line in text.splitlines():
        if "Audio:" not in line and "Stream #" not in line:
            continue
        m = re.search(r"(\d+)\s+Hz,\s*(mono|stereo)", line, re.IGNORECASE)
        if m:
            sr = int(m.group(1))
            ch = 1 if m.group(2).lower() == "mono" else 2
            return sr, ch
        m = re.search(r"(\d+)\s+Hz,\s*(\d+)\s+channels?", line, re.IGNORECASE)
        if m:
            return int(m.group(1)), int(m.group(2))
    return 0, 0


def _db_to_linear(db: float) -> float:
    if not math.isfinite(db) or db <= -100.0:
        return 0.0
    return 10.0 ** (db / 20.0)


def _total_silence_seconds(stderr_text: str) -> float:
    starts = [float(x) for x in re.findall(r"silence_start:\s*([\d.]+)", stderr_text)]
    ends = [float(x) for x in re.findall(r"silence_end:\s*([\d.]+)", stderr_text)]
    total = 0.0
    for i in range(min(len(starts), len(ends))):
        total += max(0.0, ends[i] - starts[i])
    return total


def compute_audio_stats(filepath: str) -> dict:
    try:
        size = os.path.getsize(filepath)
    except OSError:
        size = 0

    cmd = [
        FFMPEG_BIN,
        "-hide_banner",
        "-nostats",
        "-i",
        filepath,
        "-af",
        "silencedetect=noise=-50dB:d=1,volumedetect",
        "-f",
        "null",
        "-",
    ]
    r = subprocess.run(cmd, capture_output=True)
    text = _stderr_text(r)

    duration = _parse_duration_seconds(text)
    sample_rate, channels = _parse_sample_rate_channels(text)

    mean_db = None
    max_db = None
    mm = re.search(r"mean_volume:\s*([-\d.]+)\s*dB", text)
    if mm:
        try:
            mean_db = float(mm.group(1))
        except ValueError:
            pass
    mm = re.search(r"max_volume:\s*([-\d.]+)\s*dB", text)
    if mm:
        try:
            max_db = float(mm.group(1))
        except ValueError:
            pass

    mean_amp = _db_to_linear(mean_db) if mean_db is not None else 0.0
    peak_amp = _db_to_linear(max_db) if max_db is not None else 0.0

    silence_dur = _total_silence_seconds(text)
    silence_ratio = (silence_dur / duration) if duration > 0 else 0.0
    silence_ratio = min(1.0, max(0.0, silence_ratio))

    return {
        "duration_seconds": duration,
        "peak_amplitude": peak_amp,
        "mean_amplitude": mean_amp,
        "silence_ratio": silence_ratio,
        "sample_rate": sample_rate,
        "channels": channels,
        "file_size_bytes": size,
    }


def split_into_chunks(
    filepath: str,
    chunk_duration_seconds: int = 30,
    output_dir: str | None = None,
) -> list[str]:
    if chunk_duration_seconds <= 0:
        return []

    base = os.path.splitext(os.path.basename(filepath))[0]
    ext = os.path.splitext(filepath)[1] or ".wav"
    od = output_dir if output_dir is not None else (os.path.dirname(filepath) or ".")
    os.makedirs(od, exist_ok=True)

    pattern = os.path.join(od, f"{base}_chunk_%03d{ext}")
    cmd = [
        FFMPEG_BIN,
        "-y",
        "-i",
        filepath,
        "-f",
        "segment",
        "-segment_time",
        str(chunk_duration_seconds),
        "-reset_timestamps",
        "1",
        "-map",
        "0:a",
        "-c",
        "copy",
        pattern,
    ]
    r = subprocess.run(cmd, capture_output=True)
    if r.returncode != 0:
        return []

    chunks = sorted(glob.glob(os.path.join(od, f"{base}_chunk_*{ext}")))
    return chunks


def batch_normalize(
    input_dir: str,
    output_dir: str,
    config: RecordingConfig,
    max_workers: int = 4,
) -> list[tuple[str, bool]]:
    input_dir = os.path.abspath(input_dir)
    paths: list[str] = []
    for root, _, files in os.walk(input_dir):
        for name in files:
            ext = os.path.splitext(name)[1].lower()
            if ext not in _AUDIO_EXTENSIONS:
                continue
            paths.append(os.path.join(root, name))
    paths.sort()

    def run_one(src: str) -> tuple[str, bool]:
        rel = os.path.relpath(src, input_dir)
        stem = os.path.splitext(rel)[0]
        fmt = config.output_format.lstrip(".")
        out_path = os.path.join(output_dir, f"{stem}.{fmt}")
        ok = normalize_audio(src, out_path, config)
        return (src, ok)

    workers = max(1, max_workers)
    results: list[tuple[str, bool]] = []
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = [pool.submit(run_one, p) for p in paths]
        for fut in futures:
            results.append(fut.result())
    results.sort(key=lambda x: x[0])
    return results
