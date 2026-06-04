from __future__ import annotations

import random
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone

import httpx

from dataset_builder.config import BandwidthProfile, RecordingConfig, StationInfo


def _station_stream_url(station: StationInfo) -> str:
    return station.url_resolved or station.url


def test_single_stream(url: str, duration_seconds: int = 10) -> dict:
    bytes_received = 0
    content_type = ""
    error: str | None = None
    success = False
    t0 = time.perf_counter()
    try:
        with httpx.Client(timeout=15, follow_redirects=True) as client:
            with client.stream("GET", url) as response:
                content_type = response.headers.get("content-type") or ""
                try:
                    response.raise_for_status()
                except httpx.HTTPStatusError as e:
                    duration = time.perf_counter() - t0
                    return {
                        "url": url,
                        "bytes_received": 0,
                        "duration": duration,
                        "bitrate_kbps": 0.0,
                        "success": False,
                        "error": str(e),
                        "content_type": content_type,
                    }
                start_read = time.perf_counter()
                for chunk in response.iter_bytes():
                    bytes_received += len(chunk)
                    if time.perf_counter() - start_read >= duration_seconds:
                        break
                duration = time.perf_counter() - start_read
        success = True
    except Exception as e:
        duration = time.perf_counter() - t0
        error = str(e)
    bitrate_kbps = (
        (bytes_received * 8) / duration / 1000.0 if duration > 0 else 0.0
    )
    return {
        "url": url,
        "bytes_received": bytes_received,
        "duration": duration,
        "bitrate_kbps": bitrate_kbps,
        "success": success,
        "error": error,
        "content_type": content_type,
    }


def test_bandwidth(
    stations: list[StationInfo],
    num_samples: int = 20,
    concurrent: int = 10,
    duration: int = 10,
) -> BandwidthProfile:
    if not stations:
        now = datetime.now(timezone.utc).isoformat()
        return BandwidthProfile(
            tested_at=now,
            download_speed_mbps=0.0,
            recommended_concurrent=1,
            estimated_total_hours=0.0,
            total_stations=0,
        )

    n = min(num_samples, len(stations))
    sample = random.sample(stations, n)
    wall0 = time.perf_counter()
    results: list[dict] = []
    with ThreadPoolExecutor(max_workers=concurrent) as ex:
        futs = [
            ex.submit(
                test_single_stream, _station_stream_url(s), duration
            )
            for s in sample
        ]
        for f in as_completed(futs):
            results.append(f.result())
    wall_elapsed = time.perf_counter() - wall0

    total_bytes = sum(r["bytes_received"] for r in results)
    download_speed_mbps = (
        (total_bytes * 8) / wall_elapsed / 1_000_000 if wall_elapsed > 0 else 0.0
    )

    ok = [r for r in results if r["success"] and r["duration"] > 0]
    if ok:
        avg_stream_bitrate_mbps = (
            sum(r["bitrate_kbps"] for r in ok) / len(ok) / 1000.0
        )
    else:
        avg_stream_bitrate_mbps = 0.128

    if avg_stream_bitrate_mbps > 0:
        raw = int((download_speed_mbps * 0.7) / avg_stream_bitrate_mbps)
        recommended = max(1, min(200, raw))
    else:
        recommended = 1

    total_stations = len(stations)
    cfg = RecordingConfig()
    per_batch_hours = cfg.duration_seconds / 3600.0
    estimated_total_hours = (total_stations / recommended) * per_batch_hours

    return BandwidthProfile(
        tested_at=datetime.now(timezone.utc).isoformat(),
        download_speed_mbps=download_speed_mbps,
        recommended_concurrent=recommended,
        estimated_total_hours=estimated_total_hours,
        total_stations=total_stations,
    )


def test_concurrent_scaling(
    stations: list[StationInfo],
    levels: list[int] | None = None,
    duration: int = 10,
) -> list[dict]:
    if levels is None:
        levels = [10, 25, 50, 100]
    if not stations:
        return [
            {
                "concurrent": level,
                "total_mbps": 0.0,
                "per_stream_kbps": 0.0,
                "success_rate": 0.0,
                "errors": level,
            }
            for level in levels
        ]

    out: list[dict] = []
    for level in levels:
        L = min(level, len(stations))
        if L < level:
            picked = random.choices(stations, k=level)
        else:
            picked = random.sample(stations, L)
        wall0 = time.perf_counter()
        results: list[dict] = []
        with ThreadPoolExecutor(max_workers=level) as ex:
            futs = [
                ex.submit(
                    test_single_stream, _station_stream_url(s), duration
                )
                for s in picked
            ]
            for f in as_completed(futs):
                results.append(f.result())
        wall_elapsed = time.perf_counter() - wall0

        total_bytes = sum(r["bytes_received"] for r in results)
        total_mbps = (
            (total_bytes * 8) / wall_elapsed / 1_000_000
            if wall_elapsed > 0
            else 0.0
        )
        per_stream_kbps = (
            (total_mbps * 1000.0 / level) if level > 0 else 0.0
        )
        successes = sum(1 for r in results if r["success"])
        success_rate = successes / level if level > 0 else 0.0
        errors = level - successes
        out.append(
            {
                "concurrent": level,
                "total_mbps": total_mbps,
                "per_stream_kbps": per_stream_kbps,
                "success_rate": success_rate,
                "errors": errors,
            }
        )
    return out


def estimate_storage(
    stations: list[StationInfo], duration_seconds: int = 1800
) -> dict:
    total_stations = len(stations)
    if total_stations == 0:
        return {
            "total_stations": 0,
            "raw_pcm_gb": 0.0,
            "estimated_flac_gb": 0.0,
            "estimated_mp3_gb": 0.0,
            "duration_hours_total": 0.0,
        }

    rates = [s.bitrate for s in stations if s.bitrate and s.bitrate > 0]
    avg_kbps = sum(rates) / len(rates) if rates else 128.0

    cfg = RecordingConfig()
    sr = cfg.sample_rate
    ch = cfg.channels
    bytes_per_sample = 2
    raw_pcm_bytes = total_stations * duration_seconds * sr * ch * bytes_per_sample
    raw_pcm_gb = raw_pcm_bytes / (1024**3)
    estimated_flac_gb = raw_pcm_gb * 0.5

    mp3_bytes_per_station = duration_seconds * (avg_kbps * 1000 / 8)
    estimated_mp3_gb = (mp3_bytes_per_station * total_stations) / (1024**3)

    duration_hours_total = total_stations * duration_seconds / 3600.0

    return {
        "total_stations": total_stations,
        "raw_pcm_gb": raw_pcm_gb,
        "estimated_flac_gb": estimated_flac_gb,
        "estimated_mp3_gb": estimated_mp3_gb,
        "duration_hours_total": duration_hours_total,
    }
