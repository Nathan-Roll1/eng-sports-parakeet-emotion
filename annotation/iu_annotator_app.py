"""IU emotion annotation UI mounted into the Railway FastAPI service."""

from __future__ import annotations

import csv
import hashlib
import io
import json
import os
import sqlite3
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from fastapi import APIRouter, FastAPI, HTTPException, Query, Request
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse, Response
from pydantic import BaseModel, Field


PREFIX = os.environ.get("IU_ANNOTATOR_PREFIX", "/iu-annotator").rstrip("/") or "/iu-annotator"
DATA_DIR = Path(os.environ.get("IU_ANNOTATOR_DATA_DIR", "/data/iu_annotator"))
DB_PATH = Path(os.environ.get("IU_ANNOTATOR_DB_PATH", str(DATA_DIR / "annotations.db")))
AUDIO_DIR = DATA_DIR / "audio"
HF_CACHE_DIR = DATA_DIR / "hf_cache"
HF_REPO_ID = os.environ.get("IU_HF_REPO_ID", "NathanRoll/eng-sports-radio-psst-iu")
HF_PARQUET_PATH = os.environ.get("IU_HF_PARQUET_PATH", "data/train-00000.parquet")
ACCESS_KEY = os.environ.get("IU_ANNOTATOR_ACCESS_KEY", "").strip()
BOOTSTRAP_ON_START = os.environ.get("IU_BOOTSTRAP_ON_START", "1").strip().lower() not in {
    "0",
    "false",
    "no",
    "off",
}

EMOTIONS = ("neutral", "joy", "sadness", "anger", "fear", "disgust", "surprise", "low_quality")
BOOTSTRAP_BATCH_SIZE = int(os.environ.get("IU_BOOTSTRAP_BATCH_SIZE", "64"))

_BOOTSTRAP_LOCK = threading.Lock()
_BOOTSTRAP_THREAD: threading.Thread | None = None
_BOOTSTRAP_STATUS: dict[str, Any] = {
    "state": "idle",
    "imported_rows": 0,
    "expected_rows": 0,
    "message": "Waiting to import dataset.",
    "error": "",
    "started_at": "",
    "updated_at": "",
    "completed_at": "",
}


class AnnotationIn(BaseModel):
    iu_id: str = Field(min_length=1)
    annotator: str = Field(min_length=1, max_length=120)
    emotion: str
    confidence: int = Field(default=3, ge=1, le=5)
    notes: str = Field(default="", max_length=2000)


class SeenIn(BaseModel):
    iu_id: str = Field(min_length=1)
    annotator: str = Field(min_length=1, max_length=120)


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _connect() -> sqlite3.Connection:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(DB_PATH), timeout=30)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA foreign_keys=ON")
    return conn


def _set_bootstrap_status(**updates: Any) -> None:
    with _BOOTSTRAP_LOCK:
        _BOOTSTRAP_STATUS.update(updates)
        _BOOTSTRAP_STATUS["updated_at"] = _utc_now()


def init_annotator_db() -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    AUDIO_DIR.mkdir(parents=True, exist_ok=True)
    HF_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    with _connect() as conn:
        conn.executescript(
            """
            CREATE TABLE IF NOT EXISTS bootstrap_meta (
                key TEXT PRIMARY KEY,
                value TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS iu_samples (
                iu_id TEXT PRIMARY KEY,
                text TEXT NOT NULL,
                source_id TEXT NOT NULL,
                source_name TEXT NOT NULL,
                country_code TEXT,
                recording_date TEXT,
                started_at TEXT,
                segment_id TEXT,
                raw_gcs_uri TEXT,
                rights_status TEXT,
                parakeet_model TEXT,
                psst_model TEXT,
                segment_duration_seconds REAL,
                iu_index INTEGER,
                start_seconds REAL,
                end_seconds REAL,
                duration_seconds REAL,
                start_word_index INTEGER,
                end_word_index_exclusive INTEGER,
                boundary_after_word_index INTEGER,
                alignment_score REAL,
                alignment_method TEXT,
                iu_word_count INTEGER,
                speech_rate_wps REAL,
                mean_energy_db REAL,
                pause_before_ms REAL,
                pause_after_ms REAL,
                psst_boundary_strength TEXT,
                words_json TEXT,
                manifest_path TEXT,
                audio_path TEXT NOT NULL,
                imported_at TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS annotations (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                iu_id TEXT NOT NULL REFERENCES iu_samples(iu_id) ON DELETE CASCADE,
                annotator TEXT NOT NULL,
                emotion TEXT NOT NULL,
                confidence INTEGER NOT NULL DEFAULT 3,
                low_quality INTEGER NOT NULL DEFAULT 0,
                notes TEXT NOT NULL DEFAULT '',
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL,
                UNIQUE(iu_id, annotator)
            );

            CREATE TABLE IF NOT EXISTS user_sample_state (
                user_key TEXT NOT NULL,
                iu_id TEXT NOT NULL REFERENCES iu_samples(iu_id) ON DELETE CASCADE,
                first_seen_at TEXT NOT NULL,
                last_seen_at TEXT NOT NULL,
                seen_count INTEGER NOT NULL DEFAULT 1,
                PRIMARY KEY(user_key, iu_id)
            );

            CREATE INDEX IF NOT EXISTS idx_iu_samples_source ON iu_samples(source_id);
            CREATE INDEX IF NOT EXISTS idx_iu_samples_duration ON iu_samples(duration_seconds);
            CREATE INDEX IF NOT EXISTS idx_annotations_annotator ON annotations(annotator);
            CREATE INDEX IF NOT EXISTS idx_annotations_emotion ON annotations(emotion);
            CREATE INDEX IF NOT EXISTS idx_user_sample_state_user ON user_sample_state(user_key);
            CREATE INDEX IF NOT EXISTS idx_user_sample_state_iu ON user_sample_state(iu_id);
            """
        )
        annotation_columns = {row["name"] for row in conn.execute("PRAGMA table_info(annotations)").fetchall()}
        if "low_quality" not in annotation_columns:
            conn.execute("ALTER TABLE annotations ADD COLUMN low_quality INTEGER NOT NULL DEFAULT 0")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_annotations_low_quality ON annotations(low_quality)")
        conn.execute("UPDATE annotations SET emotion = 'low_quality', low_quality = 0 WHERE low_quality = 1")


def _get_meta(conn: sqlite3.Connection, key: str) -> str | None:
    row = conn.execute("SELECT value FROM bootstrap_meta WHERE key = ?", (key,)).fetchone()
    return str(row["value"]) if row else None


def _set_meta(conn: sqlite3.Connection, key: str, value: str) -> None:
    conn.execute(
        "INSERT INTO bootstrap_meta(key, value) VALUES(?, ?) "
        "ON CONFLICT(key) DO UPDATE SET value = excluded.value",
        (key, value),
    )


def _hf_token() -> str | None:
    return os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN") or None


def _current_hf_sha(token: str | None) -> str:
    try:
        from huggingface_hub import HfApi

        info = HfApi(token=token).dataset_info(HF_REPO_ID)
        return str(getattr(info, "sha", "") or "")
    except Exception:
        return ""


def _sample_count(conn: sqlite3.Connection) -> int:
    return int(conn.execute("SELECT COUNT(*) AS n FROM iu_samples").fetchone()["n"])


def _require_access(request: Request) -> None:
    if not ACCESS_KEY:
        return
    provided = request.headers.get("x-annotator-key") or request.query_params.get("key") or ""
    if provided != ACCESS_KEY:
        raise HTTPException(status_code=401, detail="annotation access key required")


def _clean_annotator(value: str | None) -> str:
    value = (value or "anonymous").strip()
    return value[:120] or "anonymous"


def _require_user_key_value(value: str | None) -> str:
    key = (value or "").strip()[:120]
    if not key:
        raise HTTPException(status_code=422, detail="user key required")
    return key


def _row_to_sample(row: sqlite3.Row | None, annotation: sqlite3.Row | None = None) -> dict[str, Any] | None:
    if row is None:
        return None
    sample = {key: row[key] for key in row.keys() if key != "audio_path"}
    sample["audio_url"] = f"{PREFIX}/api/audio/{row['iu_id']}"
    sample["words"] = []
    if sample.get("words_json"):
        try:
            sample["words"] = json.loads(sample["words_json"])
        except json.JSONDecodeError:
            sample["words"] = []
    if annotation:
        sample["annotation"] = {key: annotation[key] for key in annotation.keys()}
    else:
        sample["annotation"] = None
    return sample


def _audio_target(row: dict[str, Any]) -> Path:
    iu_id = str(row.get("iu_id") or "")
    source_id = "".join(ch if ch.isalnum() or ch in {"_", "-"} else "_" for ch in str(row.get("source_id") or "source"))
    digest = hashlib.sha1(iu_id.encode("utf-8")).hexdigest()
    return AUDIO_DIR / source_id / f"{digest}.flac"


def _coerce_float(value: Any, default: float = 0.0) -> float:
    try:
        if value is None:
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def _coerce_int(value: Any, default: int = 0) -> int:
    try:
        if value is None:
            return default
        return int(value)
    except (TypeError, ValueError):
        return default


def _prepare_sample_row(row: dict[str, Any], audio_path: Path) -> tuple[Any, ...]:
    imported_at = _utc_now()
    return (
        str(row.get("iu_id") or ""),
        str(row.get("text") or "").strip(),
        str(row.get("source_id") or ""),
        str(row.get("source_name") or row.get("source_id") or ""),
        str(row.get("country_code") or ""),
        str(row.get("recording_date") or ""),
        str(row.get("started_at") or ""),
        str(row.get("segment_id") or ""),
        str(row.get("raw_gcs_uri") or ""),
        str(row.get("rights_status") or ""),
        str(row.get("parakeet_model") or ""),
        str(row.get("psst_model") or ""),
        _coerce_float(row.get("segment_duration_seconds")),
        _coerce_int(row.get("iu_index")),
        _coerce_float(row.get("start_seconds")),
        _coerce_float(row.get("end_seconds")),
        _coerce_float(row.get("duration_seconds")),
        _coerce_int(row.get("start_word_index")),
        _coerce_int(row.get("end_word_index_exclusive")),
        _coerce_int(row.get("boundary_after_word_index"), -1),
        _coerce_float(row.get("alignment_score")),
        str(row.get("alignment_method") or ""),
        _coerce_int(row.get("iu_word_count")),
        _coerce_float(row.get("speech_rate_wps")),
        _coerce_float(row.get("mean_energy_db")),
        _coerce_float(row.get("pause_before_ms")),
        _coerce_float(row.get("pause_after_ms")),
        str(row.get("psst_boundary_strength") or ""),
        str(row.get("words_json") or ""),
        str(row.get("manifest_path") or ""),
        str(audio_path),
        imported_at,
    )


def bootstrap_from_hf(force: bool = False) -> dict[str, Any]:
    init_annotator_db()
    token = _hf_token()
    current_sha = _current_hf_sha(token)
    with _connect() as conn:
        if force:
            conn.execute("DELETE FROM annotations")
            conn.execute("DELETE FROM iu_samples")
            conn.execute("DELETE FROM bootstrap_meta")
            conn.commit()
        complete = _get_meta(conn, "sample_import_complete") == "1"
        count = _sample_count(conn)
        imported_sha = _get_meta(conn, "hf_repo_sha") or ""
        if complete and count:
            if not current_sha or imported_sha == current_sha:
                _set_bootstrap_status(
                    state="complete",
                    imported_rows=count,
                    expected_rows=count,
                    message="Dataset is already imported.",
                    error="",
                    completed_at=_get_meta(conn, "sample_import_completed_at") or "",
                )
                return dict(_BOOTSTRAP_STATUS)
            _set_bootstrap_status(
                state="refreshing",
                imported_rows=count,
                expected_rows=count,
                message="HF dataset changed; refreshing samples while preserving annotations.",
                error="",
                completed_at="",
            )

    _set_bootstrap_status(
        state="downloading",
        imported_rows=0,
        expected_rows=0,
        message=f"Downloading {HF_REPO_ID}/{HF_PARQUET_PATH}.",
        error="",
        started_at=_utc_now(),
        completed_at="",
    )

    try:
        from huggingface_hub import hf_hub_download
        import pyarrow.parquet as pq

        parquet_path = hf_hub_download(
            repo_id=HF_REPO_ID,
            filename=HF_PARQUET_PATH,
            repo_type="dataset",
            token=token or None,
            revision=current_sha or None,
            cache_dir=str(HF_CACHE_DIR),
        )
        pf = pq.ParquetFile(parquet_path)
        expected_rows = int(pf.metadata.num_rows)
        _set_bootstrap_status(
            state="importing",
            expected_rows=expected_rows,
            message=f"Importing {expected_rows} IU clips into persistent storage.",
        )

        insert_sql = """
            INSERT INTO iu_samples (
                iu_id, text, source_id, source_name, country_code, recording_date, started_at,
                segment_id, raw_gcs_uri, rights_status, parakeet_model, psst_model,
                segment_duration_seconds, iu_index, start_seconds, end_seconds, duration_seconds,
                start_word_index, end_word_index_exclusive, boundary_after_word_index,
                alignment_score, alignment_method, iu_word_count, speech_rate_wps, mean_energy_db,
                pause_before_ms, pause_after_ms, psst_boundary_strength, words_json, manifest_path,
                audio_path, imported_at
            ) VALUES (
                ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?
            )
            ON CONFLICT(iu_id) DO UPDATE SET
                text = excluded.text,
                source_id = excluded.source_id,
                source_name = excluded.source_name,
                country_code = excluded.country_code,
                recording_date = excluded.recording_date,
                started_at = excluded.started_at,
                segment_id = excluded.segment_id,
                raw_gcs_uri = excluded.raw_gcs_uri,
                rights_status = excluded.rights_status,
                parakeet_model = excluded.parakeet_model,
                psst_model = excluded.psst_model,
                segment_duration_seconds = excluded.segment_duration_seconds,
                iu_index = excluded.iu_index,
                start_seconds = excluded.start_seconds,
                end_seconds = excluded.end_seconds,
                duration_seconds = excluded.duration_seconds,
                start_word_index = excluded.start_word_index,
                end_word_index_exclusive = excluded.end_word_index_exclusive,
                boundary_after_word_index = excluded.boundary_after_word_index,
                alignment_score = excluded.alignment_score,
                alignment_method = excluded.alignment_method,
                iu_word_count = excluded.iu_word_count,
                speech_rate_wps = excluded.speech_rate_wps,
                mean_energy_db = excluded.mean_energy_db,
                pause_before_ms = excluded.pause_before_ms,
                pause_after_ms = excluded.pause_after_ms,
                psst_boundary_strength = excluded.psst_boundary_strength,
                words_json = excluded.words_json,
                manifest_path = excluded.manifest_path,
                audio_path = excluded.audio_path,
                imported_at = excluded.imported_at
        """

        imported = 0
        with _connect() as conn:
            for batch in pf.iter_batches(batch_size=BOOTSTRAP_BATCH_SIZE):
                rows = batch.to_pylist()
                prepared: list[tuple[Any, ...]] = []
                for row in rows:
                    iu_id = str(row.get("iu_id") or "")
                    text = str(row.get("text") or "").strip()
                    audio = row.get("audio") or {}
                    audio_bytes = audio.get("bytes") if isinstance(audio, dict) else None
                    if not iu_id or not audio_bytes:
                        continue
                    target = _audio_target(row)
                    target.parent.mkdir(parents=True, exist_ok=True)
                    if not target.exists() or target.stat().st_size != len(audio_bytes):
                        target.write_bytes(audio_bytes)
                    prepared.append(_prepare_sample_row(row, target))
                if prepared:
                    conn.executemany(insert_sql, prepared)
                    conn.commit()
                    imported = _sample_count(conn)
                    _set_bootstrap_status(
                        state="importing",
                        imported_rows=imported,
                        expected_rows=expected_rows,
                        message=f"Imported {imported} of {expected_rows} IU clips.",
                    )

            _set_meta(conn, "sample_import_complete", "1")
            _set_meta(conn, "sample_import_completed_at", _utc_now())
            _set_meta(conn, "hf_repo_id", HF_REPO_ID)
            _set_meta(conn, "hf_parquet_path", HF_PARQUET_PATH)
            if current_sha:
                _set_meta(conn, "hf_repo_sha", current_sha)
            conn.commit()
            final_count = _sample_count(conn)

        _set_bootstrap_status(
            state="complete",
            imported_rows=final_count,
            expected_rows=expected_rows,
            message=f"Ready with {final_count} IU clips.",
            error="",
            completed_at=_utc_now(),
        )
    except Exception as exc:  # pragma: no cover - reported through status endpoint in production.
        _set_bootstrap_status(
            state="error",
            message="Dataset import failed.",
            error=f"{type(exc).__name__}: {exc}",
        )
    return dict(_BOOTSTRAP_STATUS)


def ensure_bootstrap_started() -> None:
    global _BOOTSTRAP_THREAD
    if _BOOTSTRAP_THREAD and _BOOTSTRAP_THREAD.is_alive():
        return
    with _BOOTSTRAP_LOCK:
        if _BOOTSTRAP_THREAD and _BOOTSTRAP_THREAD.is_alive():
            return
        _BOOTSTRAP_THREAD = threading.Thread(target=bootstrap_from_hf, daemon=True, name="iu-annotator-bootstrap")
        _BOOTSTRAP_THREAD.start()


def _current_status() -> dict[str, Any]:
    init_annotator_db()
    with _connect() as conn:
        sample_count = _sample_count(conn)
        annotation_count = int(conn.execute("SELECT COUNT(*) AS n FROM annotations").fetchone()["n"])
        sources = [
            dict(row)
            for row in conn.execute(
                """
                SELECT source_id, source_name, country_code, COUNT(*) AS total
                FROM iu_samples
                GROUP BY source_id, source_name, country_code
                ORDER BY source_id
                """
            ).fetchall()
        ]
        status = dict(_BOOTSTRAP_STATUS)
        if sample_count and status.get("state") in {"idle", "downloading"}:
            status.update(
                {
                    "state": "complete",
                    "imported_rows": sample_count,
                    "expected_rows": sample_count,
                    "message": f"Ready with {sample_count} IU clips.",
                }
            )
        return {
            "bootstrap": status,
            "sample_count": sample_count,
            "annotation_count": annotation_count,
            "sources": sources,
            "emotions": list(EMOTIONS),
            "repo_id": HF_REPO_ID,
        }


def _annotation_for(conn: sqlite3.Connection, iu_id: str, annotator: str) -> sqlite3.Row | None:
    return conn.execute(
        "SELECT * FROM annotations WHERE iu_id = ? AND annotator = ?",
        (iu_id, annotator),
    ).fetchone()


def _mark_seen(conn: sqlite3.Connection, iu_id: str, user_key: str) -> None:
    now = _utc_now()
    conn.execute(
        """
        INSERT INTO user_sample_state(user_key, iu_id, first_seen_at, last_seen_at, seen_count)
        VALUES(?, ?, ?, ?, 1)
        ON CONFLICT(user_key, iu_id) DO UPDATE SET
            last_seen_at = excluded.last_seen_at,
            seen_count = user_sample_state.seen_count + 1
        """,
        (user_key, iu_id, now, now),
    )


def _fetch_sample(conn: sqlite3.Connection, iu_id: str, annotator: str | None = None) -> dict[str, Any] | None:
    row = conn.execute("SELECT * FROM iu_samples WHERE iu_id = ?", (iu_id,)).fetchone()
    annotation = _annotation_for(conn, iu_id, _clean_annotator(annotator)) if row and annotator else None
    return _row_to_sample(row, annotation)


def _next_sample(
    conn: sqlite3.Connection,
    annotator: str,
    source_id: str,
    mode: str,
    mark_seen: bool = True,
) -> dict[str, Any] | None:
    joins: list[str] = []
    where = ["NOT EXISTS (SELECT 1 FROM annotations any_a WHERE any_a.iu_id = s.iu_id)"]
    params: list[Any] = []
    if mode != "random":
        joins.append("LEFT JOIN user_sample_state u ON u.iu_id = s.iu_id AND u.user_key = ?")
        params.append(annotator)
        where.append("u.iu_id IS NULL")
    if source_id:
        where.append("s.source_id = ?")
        params.append(source_id)
    join_sql = "\n        ".join(joins)
    where_sql = f"WHERE {' AND '.join(where)}" if where else ""
    row = conn.execute(
        f"""
        SELECT s.*
        FROM iu_samples s
        {join_sql}
        {where_sql}
        ORDER BY RANDOM()
        LIMIT 1
        """,
        params,
    ).fetchone()
    if not row:
        return None
    if mark_seen:
        _mark_seen(conn, row["iu_id"], annotator)
        conn.commit()
    annotation = _annotation_for(conn, row["iu_id"], annotator)
    return _row_to_sample(row, annotation)


def _stats_for(conn: sqlite3.Connection, annotator: str) -> dict[str, Any]:
    total = _sample_count(conn)
    global_annotated = int(
        conn.execute("SELECT COUNT(DISTINCT iu_id) AS n FROM annotations").fetchone()["n"]
    )
    annotated = int(
        conn.execute(
            "SELECT COUNT(*) AS n FROM annotations WHERE annotator = ?",
            (annotator,),
        ).fetchone()["n"]
    )
    heard = int(
        conn.execute(
            "SELECT COUNT(*) AS n FROM user_sample_state WHERE user_key = ?",
            (annotator,),
        ).fetchone()["n"]
    )
    by_emotion = {
        row["emotion"]: int(row["total"])
        for row in conn.execute(
            """
            SELECT emotion, COUNT(*) AS total
            FROM annotations
            WHERE annotator = ?
            GROUP BY emotion
            ORDER BY emotion
            """,
            (annotator,),
        ).fetchall()
    }
    by_source = [
        dict(row)
        for row in conn.execute(
            """
            SELECT
                s.source_id,
                s.source_name,
                s.country_code,
                COUNT(DISTINCT s.iu_id) AS total,
                COUNT(DISTINCT u.iu_id) AS heard,
                COUNT(DISTINCT a.iu_id) AS annotated,
                COUNT(DISTINCT ga.iu_id) AS global_annotated
            FROM iu_samples s
            LEFT JOIN user_sample_state u ON u.iu_id = s.iu_id AND u.user_key = ?
            LEFT JOIN annotations a ON a.iu_id = s.iu_id AND a.annotator = ?
            LEFT JOIN annotations ga ON ga.iu_id = s.iu_id
            GROUP BY s.source_id, s.source_name, s.country_code
            ORDER BY s.source_id
            """,
            (annotator, annotator),
        ).fetchall()
    ]
    return {
        "total": total,
        "annotated": annotated,
        "global_annotated": global_annotated,
        "global_remaining": max(0, total - global_annotated),
        "heard": heard,
        "remaining": max(0, total - annotated),
        "unheard": max(0, total - heard),
        "completion": round(annotated / total, 4) if total else 0.0,
        "global_completion": round(global_annotated / total, 4) if total else 0.0,
        "heard_completion": round(heard / total, 4) if total else 0.0,
        "by_emotion": by_emotion,
        "by_source": by_source,
    }


def _html() -> str:
    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>IU Emotion Annotator</title>
  <style>
    :root {{
      color-scheme: light;
      --ink: #18212f;
      --muted: #657184;
      --line: #d7dce5;
      --panel: #ffffff;
      --page: #f6f7f9;
      --accent: #116a72;
      --accent-ink: #ffffff;
      --danger: #a43d3d;
      --focus: #2d79c7;
      font-family: Inter, ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
    }}
    * {{ box-sizing: border-box; }}
    body {{
      margin: 0;
      background: var(--page);
      color: var(--ink);
      min-height: 100vh;
    }}
    button, input, select, textarea {{
      font: inherit;
    }}
    button {{
      border: 1px solid var(--line);
      background: #fff;
      color: var(--ink);
      border-radius: 6px;
      min-height: 38px;
      padding: 0 12px;
      cursor: pointer;
    }}
    button:hover {{ border-color: #9ca8b8; }}
    button:focus-visible, input:focus-visible, select:focus-visible, textarea:focus-visible {{
      outline: 2px solid var(--focus);
      outline-offset: 2px;
    }}
    .app-hidden {{
      display: none !important;
    }}
    .key-gate {{
      min-height: 100vh;
      display: grid;
      place-items: center;
      padding: 22px;
      background: var(--page);
    }}
    .key-panel {{
      width: min(430px, 100%);
      background: #fff;
      border: 1px solid var(--line);
      border-radius: 8px;
      padding: 22px;
      display: grid;
      gap: 16px;
    }}
    .key-panel h1 {{
      font-size: 24px;
    }}
    .shell {{
      display: grid;
      grid-template-rows: auto auto 1fr;
      min-height: 100vh;
    }}
    header {{
      display: flex;
      align-items: center;
      justify-content: space-between;
      gap: 20px;
      padding: 16px 22px;
      background: #fff;
      border-bottom: 1px solid var(--line);
    }}
    h1 {{
      margin: 0;
      font-size: 20px;
      letter-spacing: 0;
    }}
    .status {{
      color: var(--muted);
      font-size: 14px;
      text-align: right;
    }}
    .header-right {{
      display: grid;
      gap: 6px;
      justify-items: end;
      min-width: 280px;
    }}
    .top-progress {{
      display: flex;
      flex-wrap: wrap;
      justify-content: flex-end;
      gap: 8px;
      font-size: 13px;
    }}
    .top-progress span {{
      display: inline-flex;
      align-items: center;
      min-height: 26px;
      padding: 0 8px;
      border: 1px solid var(--line);
      border-radius: 999px;
      background: #f9fafb;
    }}
    .top-progress strong {{
      font-weight: 800;
    }}
    .toolbar {{
      display: grid;
      grid-template-columns: minmax(180px, 1fr) minmax(160px, 240px) auto auto auto;
      gap: 10px;
      padding: 12px 22px;
      background: #fbfcfd;
      border-bottom: 1px solid var(--line);
      align-items: end;
    }}
    .key-display {{
      display: grid;
      gap: 5px;
      color: var(--muted);
      font-size: 12px;
      font-weight: 650;
      text-transform: uppercase;
    }}
    .key-display strong {{
      display: flex;
      align-items: center;
      min-height: 38px;
      color: var(--ink);
      background: #fff;
      border: 1px solid var(--line);
      border-radius: 6px;
      padding: 0 10px;
      overflow: hidden;
      text-overflow: ellipsis;
      white-space: nowrap;
      text-transform: none;
    }}
    label {{
      display: grid;
      gap: 5px;
      color: var(--muted);
      font-size: 12px;
      font-weight: 650;
      text-transform: uppercase;
    }}
    input, select, textarea {{
      width: 100%;
      border: 1px solid var(--line);
      border-radius: 6px;
      padding: 9px 10px;
      color: var(--ink);
      background: #fff;
    }}
    main {{
      display: grid;
      grid-template-columns: minmax(0, 1.6fr) minmax(320px, 0.8fr);
      gap: 16px;
      padding: 16px 22px 22px;
    }}
    .panel {{
      background: var(--panel);
      border: 1px solid var(--line);
      border-radius: 8px;
      min-width: 0;
    }}
    .sample {{
      display: grid;
      grid-template-rows: auto auto 1fr auto;
      min-height: 520px;
    }}
    .sample-head, .annotation-head, .stats-head {{
      padding: 14px 16px;
      border-bottom: 1px solid var(--line);
      display: flex;
      align-items: center;
      justify-content: space-between;
      gap: 12px;
    }}
    .station {{
      font-weight: 750;
    }}
    .meta {{
      display: flex;
      gap: 8px;
      flex-wrap: wrap;
      color: var(--muted);
      font-size: 13px;
    }}
    .chip {{
      display: inline-flex;
      align-items: center;
      min-height: 26px;
      padding: 0 8px;
      border: 1px solid var(--line);
      border-radius: 999px;
      background: #f9fafb;
      white-space: nowrap;
    }}
    audio {{
      width: calc(100% - 32px);
      margin: 16px;
    }}
    .transcript {{
      margin: 4px 16px 16px;
      padding: 18px;
      border: 1px solid var(--line);
      border-radius: 8px;
      background: #fbfbf8;
      min-height: 190px;
      font-size: clamp(22px, 2.4vw, 34px);
      line-height: 1.28;
      overflow-wrap: anywhere;
    }}
    .metrics {{
      display: grid;
      grid-template-columns: repeat(4, minmax(120px, 1fr));
      gap: 0;
      border-top: 1px solid var(--line);
    }}
    .metric {{
      padding: 13px 16px;
      border-right: 1px solid var(--line);
    }}
    .metric:last-child {{ border-right: 0; }}
    .metric span {{
      display: block;
      color: var(--muted);
      font-size: 12px;
      text-transform: uppercase;
      font-weight: 700;
      margin-bottom: 4px;
    }}
    .metric strong {{
      font-size: 18px;
    }}
    aside {{
      display: grid;
      grid-template-rows: auto auto 1fr;
      gap: 16px;
    }}
    .annotation-body {{
      padding: 16px;
      display: grid;
      gap: 14px;
    }}
    .emotion-grid {{
      display: grid;
      grid-template-columns: repeat(2, minmax(0, 1fr));
      gap: 8px;
    }}
    .emotion {{
      justify-content: center;
      min-height: 48px;
      font-weight: 750;
    }}
    .emotion.active {{
      border-color: var(--accent);
      background: var(--accent);
      color: var(--accent-ink);
    }}
    .emotion[data-emotion="neutral"] {{ border-left: 5px solid #79818f; }}
    .emotion[data-emotion="joy"] {{ border-left: 5px solid #c28b18; }}
    .emotion[data-emotion="sadness"] {{ border-left: 5px solid #3f75a2; }}
    .emotion[data-emotion="anger"] {{ border-left: 5px solid #b24a42; }}
    .emotion[data-emotion="fear"] {{ border-left: 5px solid #73589a; }}
    .emotion[data-emotion="disgust"] {{ border-left: 5px solid #5a8d55; }}
    .emotion[data-emotion="surprise"] {{ border-left: 5px solid #d06c2e; }}
    .emotion[data-emotion="low_quality"] {{ border-left: 5px solid #303846; }}
    textarea {{
      resize: vertical;
      min-height: 92px;
    }}
    .actions {{
      display: grid;
      grid-template-columns: 1fr 1fr;
      gap: 8px;
    }}
    .primary {{
      border-color: var(--accent);
      background: var(--accent);
      color: var(--accent-ink);
    }}
    .danger {{
      color: var(--danger);
      border-color: #d7a6a6;
    }}
    .stats-body {{
      padding: 12px 16px 16px;
      display: grid;
      gap: 10px;
      max-height: 310px;
      overflow: auto;
    }}
    .bar {{
      display: grid;
      gap: 5px;
      font-size: 13px;
    }}
    .bar-line {{
      height: 7px;
      background: #e9edf2;
      border-radius: 999px;
      overflow: hidden;
    }}
    .bar-fill {{
      height: 100%;
      width: 0%;
      background: var(--accent);
    }}
    .empty {{
      color: var(--muted);
      display: grid;
      place-items: center;
      padding: 60px 20px;
      text-align: center;
    }}
    @media (max-width: 920px) {{
      .toolbar {{
        grid-template-columns: 1fr 1fr;
      }}
      main {{
        grid-template-columns: 1fr;
      }}
      .metrics {{
        grid-template-columns: repeat(2, minmax(0, 1fr));
      }}
      .metric:nth-child(2) {{
        border-right: 0;
      }}
    }}
    @media (max-width: 620px) {{
      header {{
        align-items: flex-start;
        flex-direction: column;
      }}
      .header-right {{
        justify-items: start;
        min-width: 0;
        width: 100%;
      }}
      .status {{
        text-align: left;
      }}
      .top-progress {{
        justify-content: flex-start;
      }}
      .toolbar {{
        grid-template-columns: 1fr;
        padding: 12px;
      }}
      main {{
        padding: 12px;
      }}
      .emotion-grid, .actions {{
        grid-template-columns: 1fr;
      }}
      .transcript {{
        font-size: 22px;
      }}
    }}
  </style>
</head>
<body>
  <div class="key-gate" id="key-gate">
    <form class="key-panel" id="key-form">
      <h1>IU Emotion Annotator</h1>
      <label>User key
        <input id="gate-key" autocomplete="off" placeholder="type your assigned key" required>
      </label>
      <button class="primary" type="submit">Continue</button>
    </form>
  </div>
  <div class="shell app-hidden" id="app-shell">
    <header>
      <h1>IU Emotion Annotator</h1>
      <div class="header-right">
        <div class="status" id="status">Connecting...</div>
        <div class="top-progress" aria-live="polite">
          <span><strong id="top-done">0 / 0</strong>&nbsp;labeled</span>
          <span id="top-percent">0%</span>
          <span id="top-heard">0 by you</span>
        </div>
      </div>
    </header>
    <section class="toolbar" aria-label="annotation controls">
      <label>Station
        <select id="source"><option value="">All stations</option></select>
      </label>
      <div class="key-display"><span>User key</span><strong id="current-key">-</strong></div>
      <button id="change-key" type="button">Change key</button>
      <button id="next" type="button">Next</button>
      <button id="export" type="button">Export</button>
    </section>
    <main>
      <section class="panel sample">
        <div class="sample-head">
          <div>
            <div class="station" id="station">No IU loaded</div>
            <div class="meta" id="meta"></div>
          </div>
          <span class="chip" id="sample-id">-</span>
        </div>
        <audio id="audio" controls preload="metadata"></audio>
        <div class="transcript" id="transcript">Waiting for a sample...</div>
        <div class="metrics">
          <div class="metric"><span>Duration</span><strong id="duration">-</strong></div>
          <div class="metric"><span>Words</span><strong id="words">-</strong></div>
          <div class="metric"><span>Rate</span><strong id="rate">-</strong></div>
          <div class="metric"><span>Boundary</span><strong id="boundary">-</strong></div>
        </div>
      </section>
      <aside>
        <section class="panel">
          <div class="annotation-head">
            <strong>Emotion</strong>
            <span class="chip" id="saved-state">Unsaved</span>
          </div>
          <div class="annotation-body">
            <div class="emotion-grid" id="emotions"></div>
            <label>Notes
              <textarea id="notes" placeholder=""></textarea>
            </label>
            <div class="actions">
              <button id="save" class="primary" type="button">Save</button>
              <button id="skip" type="button">Skip</button>
            </div>
          </div>
        </section>
        <section class="panel">
          <div class="stats-head">
            <strong>Progress</strong>
            <span class="chip" id="progress-chip">0 / 0</span>
          </div>
          <div class="stats-body" id="stats"></div>
        </section>
      </aside>
    </main>
  </div>
  <script>
    const base = {json.dumps(PREFIX)};
    const emotions = {json.dumps(list(EMOTIONS))};
    const state = {{
      sample: null,
      emotion: null,
      statusTimer: null,
      userKey: "",
      prefetched: null,
      prefetching: null,
      prefetchSignature: "",
      currentAudioObjectUrl: ""
    }};
    const $ = (id) => document.getElementById(id);
    const els = {{
      keyGate: $("key-gate"), keyForm: $("key-form"), gateKey: $("gate-key"), appShell: $("app-shell"),
      status: $("status"), source: $("source"), currentKey: $("current-key"), changeKey: $("change-key"),
      next: $("next"), export: $("export"), station: $("station"), meta: $("meta"),
      sampleId: $("sample-id"), audio: $("audio"), transcript: $("transcript"),
      duration: $("duration"), words: $("words"), rate: $("rate"), boundary: $("boundary"),
      emotions: $("emotions"), notes: $("notes"), save: $("save"), skip: $("skip"), savedState: $("saved-state"),
      stats: $("stats"), progressChip: $("progress-chip"),
      topDone: $("top-done"), topPercent: $("top-percent"), topHeard: $("top-heard")
    }};

    function annotator() {{
      const value = state.userKey.trim();
      if (value) localStorage.setItem("iuUserKey", value);
      return value;
    }}

    function setUserKey(value) {{
      state.userKey = (value || "").trim();
      if (state.userKey) localStorage.setItem("iuUserKey", state.userKey);
      els.currentKey.textContent = state.userKey || "-";
    }}

    async function enterApp(value) {{
      const key = (value || "").trim();
      if (!key) return;
      setUserKey(key);
      els.keyGate.classList.add("app-hidden");
      els.appShell.classList.remove("app-hidden");
      const status = await loadStatus();
      if (status.bootstrap?.state !== "complete") {{
        state.statusTimer = setInterval(async () => {{
          const latest = await loadStatus();
          if (latest.bootstrap?.state === "complete" && !state.sample) {{
            await loadStats();
            await nextSample();
          }}
        }}, 4000);
        await loadStats();
        return;
      }}
      await loadStats();
      await nextSample();
    }}

    function renderEmpty(message) {{
      state.sample = null;
      state.emotion = null;
      els.station.textContent = "No IU loaded";
      els.meta.innerHTML = "";
      els.sampleId.textContent = "-";
      if (state.currentAudioObjectUrl) {{
        revokeObjectUrl(state.currentAudioObjectUrl);
        state.currentAudioObjectUrl = "";
      }}
      els.audio.removeAttribute("src");
      els.transcript.textContent = message;
      els.duration.textContent = "-";
      els.words.textContent = "-";
      els.rate.textContent = "-";
      els.boundary.textContent = "-";
      els.notes.value = "";
      chooseEmotion(null);
    }}

    function accessKey() {{
      return localStorage.getItem("iuAnnotatorKey") || "";
    }}

    function sampleAudioUrl(sample) {{
      const keyParam = accessKey() ? `?key=${{encodeURIComponent(accessKey())}}` : "";
      return `${{base}}/api/audio/${{encodeURIComponent(sample.iu_id)}}${{keyParam}}`;
    }}

    function queueSignature() {{
      return `${{annotator()}}|${{els.source.value || ""}}`;
    }}

    function revokeObjectUrl(url) {{
      if (url) URL.revokeObjectURL(url);
    }}

    function clearPrefetch() {{
      if (state.prefetched?.audio_blob_url) revokeObjectUrl(state.prefetched.audio_blob_url);
      state.prefetched = null;
      state.prefetching = null;
      state.prefetchSignature = "";
    }}

    async function api(path, options = {{}}) {{
      const headers = new Headers(options.headers || {{}});
      if (accessKey()) headers.set("x-annotator-key", accessKey());
      if (options.body && !headers.has("content-type")) headers.set("content-type", "application/json");
      const response = await fetch(base + path, {{ ...options, headers }});
      if (response.status === 401) {{
        const key = window.prompt("Access key");
        if (key) {{
          localStorage.setItem("iuAnnotatorKey", key);
          return api(path, options);
        }}
      }}
      if (!response.ok) {{
        let detail = response.statusText;
        try {{
          const body = await response.json();
          detail = body.detail || body.error || detail;
        }} catch (_err) {{}}
        throw new Error(detail);
      }}
      return response.json();
    }}

    function fmt(value, suffix = "") {{
      const num = Number(value);
      if (!Number.isFinite(num)) return "-";
      return `${{num.toFixed(num >= 10 ? 1 : 2)}}${{suffix}}`;
    }}

    function setStatus(text) {{
      els.status.textContent = text;
    }}

    async function hydrateSampleAudio(sample) {{
      if (!sample) return sample;
      try {{
        const response = await fetch(sampleAudioUrl(sample));
        if (response.ok) {{
          const blob = await response.blob();
          sample.audio_blob_url = URL.createObjectURL(blob);
        }}
      }} catch (_err) {{}}
      return sample;
    }}

    async function prefetchNext(force = false) {{
      const key = annotator();
      if (!key) return null;
      const signature = queueSignature();
      if (!force && state.prefetched && state.prefetchSignature === signature) return state.prefetched;
      if (!force && state.prefetching && state.prefetchSignature === signature) return state.prefetching;
      clearPrefetch();
      state.prefetchSignature = signature;
      state.prefetching = (async () => {{
        const params = new URLSearchParams({{
          annotator: key,
          source_id: els.source.value,
          mark_seen: "false"
        }});
        const payload = await api(`/api/next?${{params.toString()}}`);
        const sample = await hydrateSampleAudio(payload.sample);
        if (sample?.iu_id && state.sample?.iu_id === sample.iu_id) {{
          if (sample.audio_blob_url) revokeObjectUrl(sample.audio_blob_url);
          if (state.prefetchSignature === signature) {{
            state.prefetched = null;
            state.prefetching = null;
          }}
          return null;
        }}
        if (state.prefetchSignature === signature) {{
          state.prefetched = sample;
          state.prefetching = null;
        }} else if (sample?.audio_blob_url) {{
          revokeObjectUrl(sample.audio_blob_url);
        }}
        return sample;
      }})().catch((error) => {{
        if (state.prefetchSignature === signature) state.prefetching = null;
        console.warn("prefetch failed", error);
        return null;
      }});
      return state.prefetching;
    }}

    async function markSeen(sample) {{
      if (!sample) return;
      try {{
        await api("/api/seen", {{
          method: "POST",
          body: JSON.stringify({{ iu_id: sample.iu_id, annotator: annotator() }})
        }});
      }} catch (error) {{
        console.warn("mark seen failed", error);
      }}
    }}

    function renderEmotionButtons() {{
      els.emotions.innerHTML = "";
      emotions.forEach((emotion, index) => {{
        const button = document.createElement("button");
        button.type = "button";
        button.className = "emotion";
        button.dataset.emotion = emotion;
        button.textContent = `${{index + 1}}  ${{emotion.replaceAll("_", " ")}}`;
        button.addEventListener("click", async () => {{
          chooseEmotion(emotion);
          await saveAnnotation();
        }});
        els.emotions.appendChild(button);
      }});
    }}

    function chooseEmotion(emotion) {{
      state.emotion = emotion;
      [...els.emotions.children].forEach((button) => {{
        button.classList.toggle("active", button.dataset.emotion === emotion);
      }});
      els.savedState.textContent = "Unsaved";
    }}

    function populateSources(sources) {{
      const current = els.source.value;
      els.source.innerHTML = '<option value="">All stations</option>';
      sources.forEach((source) => {{
        const option = document.createElement("option");
        option.value = source.source_id;
        option.textContent = `${{source.source_id}} (${{source.total}})`;
        els.source.appendChild(option);
      }});
      els.source.value = current;
    }}

    function renderSample(sample) {{
      state.sample = sample;
      state.emotion = sample?.annotation?.emotion || null;
      if (!sample) {{
        renderEmpty("No rows are left in this queue.");
        return;
      }}
      els.station.textContent = sample.source_name || sample.source_id;
      els.sampleId.textContent = `IU ${{sample.iu_index}}`;
      els.meta.innerHTML = "";
      [sample.source_id, sample.country_code, sample.recording_date].filter(Boolean).forEach((value) => {{
        const chip = document.createElement("span");
        chip.className = "chip";
        chip.textContent = value;
        els.meta.appendChild(chip);
      }});
      const audioSrc = sample.audio_blob_url || sampleAudioUrl(sample);
      if (state.currentAudioObjectUrl && state.currentAudioObjectUrl !== audioSrc) revokeObjectUrl(state.currentAudioObjectUrl);
      state.currentAudioObjectUrl = sample.audio_blob_url || "";
      els.audio.src = audioSrc;
      els.transcript.textContent = sample.text || "";
      els.duration.textContent = fmt(sample.duration_seconds, "s");
      els.words.textContent = sample.iu_word_count ?? "-";
      els.rate.textContent = fmt(sample.speech_rate_wps, "/s");
      els.boundary.textContent = sample.psst_boundary_strength || "-";
      els.notes.value = sample.annotation?.notes || "";
      chooseEmotion(state.emotion);
      els.savedState.textContent = sample.annotation ? "Saved" : "Unsaved";
    }}

    async function loadStatus() {{
      const status = await api("/api/status");
      populateSources(status.sources || []);
      const boot = status.bootstrap || {{}};
      if (boot.state === "complete") {{
        setStatus(`${{status.sample_count}} IUs ready · ${{status.annotation_count}} annotations`);
        if (state.statusTimer) {{
          clearInterval(state.statusTimer);
          state.statusTimer = null;
        }}
      }} else if (boot.state === "error") {{
        setStatus(`Import error · ${{boot.error || "check logs"}}`);
      }} else {{
        setStatus(`${{boot.message || "Importing dataset"}}`);
      }}
      return status;
    }}

    async function loadStats() {{
      const key = annotator();
      if (!key) {{
        els.progressChip.textContent = "0 / 0";
        els.topDone.textContent = "0 / 0";
        els.topPercent.textContent = "0%";
        els.topHeard.textContent = "0 heard";
        els.stats.innerHTML = "";
        const empty = document.createElement("div");
        empty.className = "empty";
        empty.textContent = "Type your user key to continue.";
        els.stats.appendChild(empty);
        return null;
      }}
      const stats = await api(`/api/stats?annotator=${{encodeURIComponent(key)}}`);
      els.progressChip.textContent = `${{stats.global_annotated}} / ${{stats.total}}`;
      els.topDone.textContent = `${{stats.global_annotated}} / ${{stats.total}}`;
      els.topPercent.textContent = `${{Math.round((stats.global_completion || 0) * 100)}}%`;
      els.topHeard.textContent = `${{stats.annotated || 0}} by you`;
      els.stats.innerHTML = "";
      const progress = document.createElement("div");
      progress.className = "bar";
      progress.innerHTML = `<div>${{Math.round((stats.global_completion || 0) * 100)}}% globally labeled · ${{stats.global_remaining || 0}} unlabeled · ${{stats.annotated || 0}} by you</div><div class="bar-line"><div class="bar-fill" style="width:${{Math.round((stats.global_completion || 0) * 100)}}%"></div></div>`;
      els.stats.appendChild(progress);
      Object.entries(stats.by_emotion || {{}}).forEach(([emotion, total]) => {{
        const row = document.createElement("div");
        row.className = "bar";
        row.textContent = `${{emotion}} · ${{total}}`;
        els.stats.appendChild(row);
      }});
      (stats.by_source || []).forEach((source) => {{
        const pct = source.total ? Math.round(source.global_annotated / source.total * 100) : 0;
        const row = document.createElement("div");
        row.className = "bar";
        row.innerHTML = `<div>${{source.source_id}} · ${{source.global_annotated}}/${{source.total}} globally labeled · ${{source.annotated}} by you</div><div class="bar-line"><div class="bar-fill" style="width:${{pct}}%"></div></div>`;
        els.stats.appendChild(row);
      }});
      return stats;
    }}

    async function nextSample() {{
      const key = annotator();
      if (!key) {{
        renderEmpty("Type your user key to start.");
        await loadStats();
        return;
      }}
      els.next.disabled = true;
      try {{
        const signature = queueSignature();
        if (state.prefetched && state.prefetchSignature === signature) {{
          const sample = state.prefetched;
          state.prefetched = null;
          state.prefetchSignature = "";
          renderSample(sample);
          markSeen(sample).then(loadStats);
          prefetchNext(true);
          return;
        }}
        const params = new URLSearchParams({{
          annotator: key,
          source_id: els.source.value,
          mark_seen: "true"
        }});
        const sample = await api(`/api/next?${{params.toString()}}`);
        renderSample(sample.sample);
        await loadStats();
        prefetchNext(true);
      }} finally {{
        els.next.disabled = false;
      }}
    }}

    async function saveAnnotation() {{
      const key = annotator();
      if (!key) {{
        renderEmpty("Type your user key to start.");
        await loadStats();
        return;
      }}
      if (!state.sample || !state.emotion) return;
      els.save.disabled = true;
      const payload = {{
        iu_id: state.sample.iu_id,
        annotator: key,
        emotion: state.emotion,
        notes: els.notes.value
      }};
      const usedPrefetch = Boolean(state.prefetched && state.prefetchSignature === queueSignature());
      if (usedPrefetch) {{
        await nextSample();
      }}
      try {{
        await api("/api/annotations", {{
          method: "POST",
          body: JSON.stringify(payload)
        }});
        els.savedState.textContent = "Saved";
        if (!usedPrefetch) await nextSample();
        else await loadStats();
      }} finally {{
        els.save.disabled = false;
      }}
    }}

    async function exportCsv() {{
      const key = annotator();
      if (!key) {{
        renderEmpty("Type your user key to export your annotations.");
        await loadStats();
        return;
      }}
      const keyParam = accessKey() ? `&key=${{encodeURIComponent(accessKey())}}` : "";
      window.location.href = `${{base}}/api/export.csv?annotator=${{encodeURIComponent(key)}}${{keyParam}}`;
    }}

    function bind() {{
      els.gateKey.value = localStorage.getItem("iuUserKey") || localStorage.getItem("iuAnnotatorName") || "";
      els.keyForm.addEventListener("submit", (event) => {{
        event.preventDefault();
        enterApp(els.gateKey.value).catch((error) => {{
          setStatus(error.message || String(error));
        }});
      }});
      els.changeKey.addEventListener("click", () => {{
        els.appShell.classList.add("app-hidden");
        els.keyGate.classList.remove("app-hidden");
        els.gateKey.value = state.userKey || localStorage.getItem("iuUserKey") || "";
        els.gateKey.focus();
      }});
      els.next.addEventListener("click", nextSample);
      els.skip.addEventListener("click", nextSample);
      els.save.addEventListener("click", saveAnnotation);
      els.export.addEventListener("click", exportCsv);
      els.source.addEventListener("change", nextSample);
      document.addEventListener("keydown", (event) => {{
        if (event.target && ["INPUT", "TEXTAREA", "SELECT"].includes(event.target.tagName)) return;
        const idx = Number(event.key) - 1;
        if (idx >= 0 && idx < emotions.length) chooseEmotion(emotions[idx]);
        if (event.key === "Enter") saveAnnotation();
        if (event.key === " ") {{
          event.preventDefault();
          els.audio.paused ? els.audio.play() : els.audio.pause();
        }}
      }});
    }}

    async function start() {{
      renderEmotionButtons();
      bind();
      els.gateKey.focus();
    }}

    start().catch((error) => {{
      setStatus(error.message || String(error));
      const empty = document.createElement("div");
      empty.className = "empty";
      empty.textContent = error.message || String(error);
      els.stats.innerHTML = "";
      els.stats.appendChild(empty);
    }});
  </script>
</body>
</html>"""


def mount_iu_annotator(app: FastAPI) -> None:
    init_annotator_db()
    router = APIRouter(prefix=PREFIX, tags=["iu-annotator"])

    @router.get("", include_in_schema=False)
    @router.get("/", include_in_schema=False)
    async def index(request: Request) -> HTMLResponse:
        _require_access(request)
        return HTMLResponse(_html())

    @router.get("/api/status")
    async def status(request: Request) -> dict[str, Any]:
        _require_access(request)
        if BOOTSTRAP_ON_START:
            ensure_bootstrap_started()
        return _current_status()

    @router.post("/api/bootstrap")
    async def bootstrap(request: Request, force: bool = Query(default=False)) -> dict[str, Any]:
        _require_access(request)
        if force:
            return bootstrap_from_hf(force=True)
        ensure_bootstrap_started()
        return _current_status()

    @router.get("/api/next")
    async def next_sample(
        request: Request,
        annotator: str = Query(default=""),
        source_id: str = Query(default=""),
        mode: str = Query(default="unlabeled"),
        mark_seen: bool = Query(default=True),
    ) -> dict[str, Any]:
        _require_access(request)
        annotator = _require_user_key_value(annotator)
        mode = "random" if mode == "random" else "unlabeled"
        with _connect() as conn:
            if not _sample_count(conn):
                raise HTTPException(status_code=503, detail="dataset import has not completed")
            return {"sample": _next_sample(conn, annotator, source_id.strip(), mode, mark_seen=mark_seen)}

    @router.post("/api/seen")
    async def seen(request: Request, payload: SeenIn) -> dict[str, Any]:
        _require_access(request)
        annotator = _require_user_key_value(payload.annotator)
        with _connect() as conn:
            exists = conn.execute("SELECT 1 FROM iu_samples WHERE iu_id = ?", (payload.iu_id,)).fetchone()
            if not exists:
                raise HTTPException(status_code=404, detail="sample not found")
            _mark_seen(conn, payload.iu_id, annotator)
            conn.commit()
        return {"ok": True}

    @router.get("/api/samples/{iu_id:path}")
    async def sample(request: Request, iu_id: str, annotator: str = Query(default="")) -> dict[str, Any]:
        _require_access(request)
        with _connect() as conn:
            payload = _fetch_sample(conn, iu_id, annotator or None)
            if not payload:
                raise HTTPException(status_code=404, detail="sample not found")
            return {"sample": payload}

    @router.get("/api/audio/{iu_id:path}")
    async def audio(request: Request, iu_id: str) -> FileResponse:
        _require_access(request)
        with _connect() as conn:
            row = conn.execute("SELECT audio_path FROM iu_samples WHERE iu_id = ?", (iu_id,)).fetchone()
            if not row:
                raise HTTPException(status_code=404, detail="sample not found")
            path = Path(row["audio_path"])
        if not path.exists():
            raise HTTPException(status_code=404, detail="audio file not found")
        return FileResponse(str(path), media_type="audio/flac")

    @router.post("/api/annotations")
    async def annotate(request: Request, payload: AnnotationIn) -> dict[str, Any]:
        _require_access(request)
        emotion = payload.emotion.strip().lower()
        if emotion not in EMOTIONS:
            raise HTTPException(status_code=422, detail=f"emotion must be one of: {', '.join(EMOTIONS)}")
        annotator = _require_user_key_value(payload.annotator)
        now = _utc_now()
        with _connect() as conn:
            exists = conn.execute("SELECT 1 FROM iu_samples WHERE iu_id = ?", (payload.iu_id,)).fetchone()
            if not exists:
                raise HTTPException(status_code=404, detail="sample not found")
            conn.execute(
                """
                INSERT INTO annotations(iu_id, annotator, emotion, confidence, low_quality, notes, created_at, updated_at)
                VALUES(?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(iu_id, annotator) DO UPDATE SET
                    emotion = excluded.emotion,
                    confidence = excluded.confidence,
                    low_quality = excluded.low_quality,
                    notes = excluded.notes,
                    updated_at = excluded.updated_at
                """,
                (
                    payload.iu_id,
                    annotator,
                    emotion,
                    3,
                    0,
                    payload.notes.strip(),
                    now,
                    now,
                ),
            )
            _mark_seen(conn, payload.iu_id, annotator)
            conn.commit()
            sample_payload = _fetch_sample(conn, payload.iu_id, annotator)
        return {"ok": True, "sample": sample_payload}

    @router.get("/api/stats")
    async def stats(request: Request, annotator: str = Query(default="")) -> dict[str, Any]:
        _require_access(request)
        annotator = _require_user_key_value(annotator)
        with _connect() as conn:
            return _stats_for(conn, annotator)

    @router.get("/api/export.csv")
    async def export_csv(request: Request, annotator: str = Query(default="")) -> Response:
        _require_access(request)
        query_params: tuple[Any, ...] = ()
        where = ""
        if annotator.strip():
            where = "WHERE a.annotator = ?"
            query_params = (_clean_annotator(annotator),)
        with _connect() as conn:
            rows = conn.execute(
                f"""
                SELECT
                    a.annotator, a.emotion, a.notes, a.created_at, a.updated_at,
                    s.iu_id, s.text, s.source_id, s.source_name, s.country_code, s.recording_date,
                    s.duration_seconds, s.speech_rate_wps, s.mean_energy_db, s.pause_before_ms,
                    s.pause_after_ms, s.psst_boundary_strength, s.alignment_score, s.alignment_method
                FROM annotations a
                JOIN iu_samples s ON s.iu_id = a.iu_id
                {where}
                ORDER BY a.updated_at DESC
                """,
                query_params,
            ).fetchall()
        out = io.StringIO()
        writer = csv.writer(out)
        columns = [
            "annotator",
            "emotion",
            "notes",
            "created_at",
            "updated_at",
            "iu_id",
            "text",
            "source_id",
            "source_name",
            "country_code",
            "recording_date",
            "duration_seconds",
            "speech_rate_wps",
            "mean_energy_db",
            "pause_before_ms",
            "pause_after_ms",
            "psst_boundary_strength",
            "alignment_score",
            "alignment_method",
        ]
        writer.writerow(columns)
        for row in rows:
            writer.writerow([row[column] for column in columns])
        filename = f"iu_annotations_{_clean_annotator(annotator) if annotator else 'all'}.csv"
        return Response(
            content=out.getvalue(),
            media_type="text/csv",
            headers={"Content-Disposition": f'attachment; filename="{filename}"'},
        )

    app.include_router(router)
    if BOOTSTRAP_ON_START:
        ensure_bootstrap_started()
