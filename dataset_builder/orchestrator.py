from __future__ import annotations

import asyncio
import dataclasses
import inspect
import math
import os
from collections.abc import Awaitable, Callable
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Optional, Union

from dataset_builder.checkpoint import CheckpointManager
from dataset_builder.config import RecordingConfig, RecordingResult, StationInfo
from dataset_builder.recorder import record_station
from dataset_builder.validator import validate_recording


def estimate_completion_time(total_stations: int, concurrent: int, duration_per: int = 1800) -> float:
    if concurrent <= 0:
        raise ValueError("concurrent must be positive")
    return math.ceil(total_stations / concurrent) * duration_per / 3600.0


def _station_label(station: StationInfo) -> str:
    name = getattr(station, "name", "") or "unknown"
    lang = getattr(station, "language", None) or getattr(station, "iso_639", None) or ""
    return f"{name} ({lang})" if lang else name


def _failure_message(result: RecordingResult) -> str:
    err = getattr(result, "error", None)
    if err:
        return str(err)
    return "recording failed"


class RecordingOrchestrator:
    def __init__(self, config: RecordingConfig, checkpoint: CheckpointManager) -> None:
        self._config = config
        self._checkpoint = checkpoint
        self._results: list[RecordingResult] = []

    @property
    def results(self) -> list[RecordingResult]:
        return self._results

    def get_stats(self) -> dict[str, Any]:
        completed = [r for r in self._results if getattr(r, "success", False)]
        failed = len(self._results) - len(completed)
        total_bytes = sum(int(getattr(r, "file_size_bytes", 0) or 0) for r in completed)
        total_duration = sum(float(getattr(r, "duration_actual", 0.0) or 0.0) for r in completed)
        avg_speed = (total_bytes / total_duration) if total_duration > 0 else 0.0
        return {
            "total": len(self._results),
            "completed": len(completed),
            "failed": failed,
            "total_bytes": total_bytes,
            "total_duration": total_duration,
            "avg_speed": avg_speed,
        }

    def run_sync(
        self,
        stations: list[StationInfo],
        on_complete: Optional[Callable[..., Union[None, Awaitable[None]]]] = None,
    ) -> list[RecordingResult]:
        return asyncio.run(self.run(stations, on_complete))

    async def run(
        self,
        stations: list[StationInfo],
        on_complete: Optional[Callable[..., Union[None, Awaitable[None]]]] = None,
    ) -> list[RecordingResult]:
        self._results = []
        total = len(stations)
        progress_lock = asyncio.Lock()
        progress_done = sum(1 for s in stations if self._checkpoint.is_completed(s.stationuuid))
        semaphore = asyncio.Semaphore(self._config.max_concurrent)
        loop = asyncio.get_running_loop()
        to_record = [s for s in stations if not self._checkpoint.is_completed(s.stationuuid)]

        async def bump_progress() -> int:
            nonlocal progress_done
            async with progress_lock:
                progress_done += 1
                return progress_done

        async def notify(station: StationInfo, result: RecordingResult) -> None:
            if on_complete is None:
                return
            out = on_complete(station, result)
            if inspect.isawaitable(out):
                await out

        def record_blocking(st: StationInfo) -> RecordingResult:
            return record_station(st, self._config)

        async def process_one(station: StationInfo) -> None:
            async with semaphore:
                result = await loop.run_in_executor(executor, record_blocking, station)
                fp = getattr(result, "filepath", "") or ""
                if getattr(result, "success", False) and fp and os.path.isfile(fp):
                    ok, msg = validate_recording(
                        fp,
                        float(self._config.min_duration),
                        float(self._config.max_silence_ratio),
                    )
                    if not ok:
                        if dataclasses.is_dataclass(result) and not isinstance(
                            result, type
                        ):
                            result = dataclasses.replace(result, success=False, error=msg)
                        else:
                            result.success = False
                            result.error = msg
                self._checkpoint.mark_completed(station.stationuuid, result)
                self._results.append(result)
                idx = await bump_progress()
                label = _station_label(station)
                if getattr(result, "success", False):
                    dur = int(round(float(getattr(result, "duration_actual", 0.0) or 0.0)))
                    mb = float(getattr(result, "file_size_bytes", 0) or 0) / (1024 * 1024)
                    print(f"[{idx}/{total}] ✓ {label} - {dur}s, {mb:.1f}MB")
                else:
                    print(f"[{idx}/{total}] ✗ {label} - {_failure_message(result)}")
                await notify(station, result)

        with ThreadPoolExecutor(max_workers=self._config.max_concurrent) as executor:
            tasks = [asyncio.create_task(process_one(s)) for s in to_record]
            try:
                await asyncio.gather(*tasks)
            except KeyboardInterrupt:
                for t in tasks:
                    if not t.done():
                        t.cancel()
                await asyncio.gather(*tasks, return_exceptions=True)
                raise

        return self._results
