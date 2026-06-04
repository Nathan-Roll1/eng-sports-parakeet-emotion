from __future__ import annotations

import json
import os
import tempfile
import threading
from dataclasses import asdict
from datetime import datetime, timezone

from dataset_builder.config import RecordingConfig, RecordingResult


class CheckpointManager:
    def __init__(self, checkpoint_file: str = "checkpoint.json") -> None:
        self.checkpoint_file = checkpoint_file
        self._lock = threading.Lock()
        self._state: dict = self._empty_state()
        if os.path.isfile(checkpoint_file):
            self.load()

    def _empty_state(self) -> dict:
        return {
            "completed": {},
            "failed": {},
            "in_progress": set(),
            "started_at": datetime.now(timezone.utc).isoformat(),
            "config": {},
        }

    def _max_retries_from_state(self) -> int:
        cfg = self._state.get("config") or {}
        return int(cfg.get("max_retries", 3))

    def is_completed(self, stationuuid: str) -> bool:
        with self._lock:
            return stationuuid in self._state["completed"]

    def is_failed(self, stationuuid: str) -> bool:
        with self._lock:
            entry = self._state["failed"].get(stationuuid)
            if not entry:
                return False
            return entry.get("attempts", 0) >= self._max_retries_from_state()

    def should_skip(self, stationuuid: str, max_retries: int = 3) -> bool:
        with self._lock:
            if stationuuid in self._state["completed"]:
                return True
            entry = self._state["failed"].get(stationuuid)
            if entry and entry.get("attempts", 0) >= max_retries:
                return True
            return False

    def mark_in_progress(self, stationuuid: str) -> None:
        with self._lock:
            self._state["in_progress"].add(stationuuid)

    def mark_completed(self, stationuuid: str, result: RecordingResult) -> None:
        with self._lock:
            self._state["in_progress"].discard(stationuuid)
            self._state["failed"].pop(stationuuid, None)
            self._state["completed"][stationuuid] = asdict(result)
        self.save()

    def mark_failed(self, stationuuid: str, error: str) -> None:
        with self._lock:
            self._state["in_progress"].discard(stationuuid)
            prev = self._state["failed"].get(stationuuid, {})
            attempts = int(prev.get("attempts", 0)) + 1
            self._state["failed"][stationuuid] = {
                "error": error,
                "attempts": attempts,
                "last_attempt": datetime.now(timezone.utc).isoformat(),
            }
        self.save()

    def get_progress(self) -> dict:
        with self._lock:
            completed = self._state["completed"]
            failed = self._state["failed"]
            in_prog = self._state["in_progress"]
            max_r = self._max_retries_from_state()
            failed_perm = sum(
                1 for e in failed.values() if int(e.get("attempts", 0)) >= max_r
            )
            total_attempted = len(completed) + sum(
                int(e.get("attempts", 0)) for e in failed.values()
            )
            total_bytes = sum(
                int(r.get("file_size_bytes", 0)) for r in completed.values()
            )
            total_duration_hours = (
                sum(float(r.get("duration_actual", 0.0)) for r in completed.values())
                / 3600.0
            )
            return {
                "completed": len(completed),
                "failed": failed_perm,
                "in_progress": len(in_prog),
                "total_attempted": total_attempted,
                "total_bytes": total_bytes,
                "total_duration_hours": total_duration_hours,
            }

    def save(self) -> None:
        with self._lock:
            payload = {
                "completed": self._state["completed"],
                "failed": self._state["failed"],
                "in_progress": sorted(self._state["in_progress"]),
                "started_at": self._state["started_at"],
                "config": self._state["config"],
            }
            data = json.dumps(payload, indent=2, sort_keys=True)
            dir_name = os.path.dirname(os.path.abspath(self.checkpoint_file)) or "."
            fd, tmp_path = tempfile.mkstemp(
                dir=dir_name, prefix=".checkpoint_", suffix=".tmp"
            )
            try:
                with os.fdopen(fd, "w", encoding="utf-8") as f:
                    f.write(data)
                    f.flush()
                    os.fsync(f.fileno())
                os.replace(tmp_path, self.checkpoint_file)
            except Exception:
                try:
                    os.unlink(tmp_path)
                except OSError:
                    pass
                raise

    def load(self) -> None:
        with open(self.checkpoint_file, encoding="utf-8") as f:
            raw = json.load(f)
        with self._lock:
            self._state = {
                "completed": dict(raw.get("completed") or {}),
                "failed": dict(raw.get("failed") or {}),
                "in_progress": set(raw.get("in_progress") or []),
                "started_at": str(raw.get("started_at") or ""),
                "config": dict(raw.get("config") or {}),
            }

    def get_remaining(self, all_uuids: list[str]) -> list[str]:
        with self._lock:
            max_r = self._max_retries_from_state()
            out: list[str] = []
            for u in all_uuids:
                if u in self._state["completed"]:
                    continue
                fe = self._state["failed"].get(u)
                if fe and int(fe.get("attempts", 0)) >= max_r:
                    continue
                out.append(u)
            return out

    def reset(self) -> None:
        with self._lock:
            self._state = self._empty_state()
        self.save()
