from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import asdict, fields, replace
from pathlib import Path
from typing import Any

if __name__ == "__main__" and __package__ is None:
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from dataset_builder.bandwidth import (
    estimate_storage,
    test_bandwidth,
    test_concurrent_scaling,
)
from dataset_builder.checkpoint import CheckpointManager
from dataset_builder.config import RecordingConfig, RecordingResult, StationInfo
from dataset_builder.discovery import (
    best_per_country,
    best_per_language,
    deduplicate_stations,
    discover_all_news_stations,
    filter_stations,
)
from dataset_builder.hf_dataset import create_dataset, export_stats
from dataset_builder.orchestrator import RecordingOrchestrator, estimate_completion_time
from dataset_builder.validator import batch_validate

SCALING_LEVELS = [10, 25, 50, 100, 150, 200]


def _split_csv(s: str | None) -> list[str] | None:
    if not s or not s.strip():
        return None
    return [p.strip() for p in s.split(",") if p.strip()]


def _station_from_dict(d: dict[str, Any]) -> StationInfo:
    field_names = {f.name for f in fields(StationInfo)}
    return StationInfo(**{k: v for k, v in d.items() if k in field_names})


def _result_from_dict(d: dict[str, Any]) -> RecordingResult:
    station = _station_from_dict(d["station"])
    field_names = {f.name for f in fields(RecordingResult)}
    kwargs = {k: v for k, v in d.items() if k in field_names and k != "station"}
    kwargs["station"] = station
    return RecordingResult(**kwargs)


def load_results_from_checkpoint(checkpoint_path: str) -> list[RecordingResult]:
    raw = json.loads(Path(checkpoint_path).read_text(encoding="utf-8"))
    completed = raw.get("completed") or {}
    return [_result_from_dict(v) for v in completed.values()]


def resolve_audio_paths(
    results: list[RecordingResult], input_dir: Path
) -> list[RecordingResult]:
    out: list[RecordingResult] = []
    for r in results:
        fp = r.filepath
        if fp and not Path(fp).is_absolute():
            out.append(replace(r, filepath=str((input_dir / fp).resolve())))
        else:
            out.append(r)
    return out


def cmd_discover(args: argparse.Namespace) -> int:
    t0 = time.perf_counter()
    try:
        print(f"Discovering news stations (min_votes>={args.min_votes})...")
        raw = discover_all_news_stations()
        deduped = deduplicate_stations(raw)
        stations = filter_stations(
            deduped,
            min_votes=args.min_votes,
            languages=_split_csv(args.languages),
            countries=_split_csv(args.countries),
            limit=args.limit,
        )
        langs = len({s.language_canonical for s in stations})
        print(
            f"Summary: raw={len(raw)}, deduped={len(deduped)}, "
            f"after_filters={len(stations)}, distinct_languages={langs}"
        )
        for s in stations[:20]:
            print(
                f"  {s.name} | {s.language_canonical} | {s.country} | votes={s.votes}"
            )
        if len(stations) > 20:
            print(f"  ... and {len(stations) - 20} more")
        if args.output:
            path = Path(args.output)
            path.write_text(
                json.dumps([asdict(s) for s in stations], indent=2),
                encoding="utf-8",
            )
            print(f"Wrote {len(stations)} stations to {path}")
    except KeyboardInterrupt:
        print("\nInterrupted.")
        return 130
    print(f"Finished in {time.perf_counter() - t0:.2f}s")
    return 0


def cmd_bandwidth(args: argparse.Namespace) -> int:
    t0 = time.perf_counter()
    try:
        print("Loading stations for bandwidth probe...")
        stations = deduplicate_stations(discover_all_news_stations())
        if len(stations) < args.samples:
            print(f"Only {len(stations)} stations available (requested {args.samples}).")
        profile = test_bandwidth(
            stations,
            num_samples=args.samples,
            concurrent=args.concurrent,
            duration=args.duration,
        )
        print(
            f"Bandwidth: {profile.download_speed_mbps:.2f} Mbps effective; "
            f"recommended_concurrent={profile.recommended_concurrent}; "
            f"estimated_total_hours≈{profile.estimated_total_hours:.2f} "
            f"(for {profile.total_stations} stations)"
        )
        if args.scaling:
            print(f"Running scaling sweep at {SCALING_LEVELS}...")
            rows = test_concurrent_scaling(
                stations, levels=SCALING_LEVELS, duration=args.duration
            )
            for row in rows:
                print(
                    f"  concurrent={row['concurrent']}: "
                    f"{row['total_mbps']:.2f} Mbps total, "
                    f"per_stream≈{row['per_stream_kbps']:.1f} kbps, "
                    f"success_rate={row['success_rate']:.2%}, errors={row['errors']}"
                )
    except KeyboardInterrupt:
        print("\nInterrupted.")
        return 130
    print(f"Finished in {time.perf_counter() - t0:.2f}s")
    return 0


def _prepare_stations_for_record(args: argparse.Namespace) -> list[StationInfo]:
    stations = deduplicate_stations(discover_all_news_stations())
    stations = filter_stations(
        stations,
        min_votes=args.min_votes,
        languages=_split_csv(args.languages),
        countries=_split_csv(args.countries),
        limit=None,
    )
    if args.best_per_language:
        stations = list(best_per_language(stations).values())
    if args.best_per_country:
        stations = list(best_per_country(stations).values())
    stations = sorted(stations, key=lambda s: s.votes, reverse=True)
    if args.limit is not None:
        stations = stations[: args.limit]
    return stations


def cmd_record(args: argparse.Namespace) -> int:
    t0 = time.perf_counter()
    try:
        print("Discovering and filtering stations...")
        stations = _prepare_stations_for_record(args)
        n = len(stations)
        ckpt = CheckpointManager(args.checkpoint)
        to_run = sum(1 for s in stations if not ckpt.should_skip(s.stationuuid))
        already = n - to_run
        est_h = estimate_completion_time(to_run, args.concurrent, args.duration)
        storage = estimate_storage(stations, duration_seconds=args.duration)
        print(
            f"Summary: stations={n}, already_done_or_skipped={already}, "
            f"remaining_to_capture≈{to_run}, concurrent={args.concurrent}, "
            f"duration_per_station={args.duration}s, est_wall_time≈{est_h:.2f}h"
        )
        print(
            f"Storage estimate: flac≈{storage['estimated_flac_gb']:.2f} GB, "
            f"raw_pcm≈{storage['raw_pcm_gb']:.2f} GB, "
            f"hours_audio_total≈{storage['duration_hours_total']:.2f}"
        )
        if already:
            print(
                f"Checkpoint {args.checkpoint!r} will skip {already} stations "
                f"(completed or retries exhausted)."
            )
        if args.resume and already:
            print("Resume: continuing with existing checkpoint state.")
        if not args.yes:
            ans = input(f"Proceed with recording ({to_run} new jobs)? [y/N]: ").strip().lower()
            if ans not in ("y", "yes"):
                print("Aborted.")
                return 1
        config = RecordingConfig(
            duration_seconds=args.duration,
            sample_rate=args.sample_rate,
            max_concurrent=args.concurrent,
            output_dir=args.output_dir,
            checkpoint_file=args.checkpoint,
        )
        orchestrator = RecordingOrchestrator(config, ckpt)
        print(f"Starting recording run...")
        orchestrator.run_sync(stations)
        run_stats = orchestrator.get_stats()
        cum = ckpt.get_progress()
        print(
            f"This run: attempts={run_stats.get('total')}, "
            f"ok={run_stats.get('completed')}, failed={run_stats.get('failed')}"
        )
        print(
            f"Checkpoint totals: completed={cum['completed']}, "
            f"failed_permanent={cum['failed']}, total_bytes={cum['total_bytes']}, "
            f"audio_hours≈{cum['total_duration_hours']:.2f}"
        )
    except KeyboardInterrupt:
        print("\nInterrupted; checkpoint should reflect progress.")
        return 130
    print(f"Finished in {time.perf_counter() - t0:.2f}s")
    return 0


def cmd_validate(args: argparse.Namespace) -> int:
    t0 = time.perf_counter()
    try:
        print(f"Validating audio under {args.input_dir!r}...")
        report = batch_validate(
            args.input_dir,
            min_duration=args.min_duration,
            max_silence_ratio=args.max_silence,
        )
        print(
            f"Summary: total={report['total']}, valid={report['valid']}, "
            f"invalid={report['invalid']}"
        )
        for row in report.get("results", [])[:30]:
            status = "ok" if row["valid"] else row.get("reason", "fail")
            print(f"  {row['filepath']}: {status}")
        rest = len(report.get("results", [])) - 30
        if rest > 0:
            print(f"  ... {rest} more rows omitted")
    except KeyboardInterrupt:
        print("\nInterrupted.")
        return 130
    print(f"Finished in {time.perf_counter() - t0:.2f}s")
    return 0


def cmd_build_dataset(args: argparse.Namespace) -> int:
    t0 = time.perf_counter()
    try:
        input_dir = Path(args.input_dir)
        print(f"Loading results from checkpoint {args.checkpoint!r}...")
        results = load_results_from_checkpoint(args.checkpoint)
        results = resolve_audio_paths(results, input_dir.resolve())
        out = create_dataset(results, output_dir=args.output_dir)
        stats = export_stats(results)
        print(
            f"Dataset written to {out}: recordings={stats['total_recordings']}, "
            f"ok={stats['successful']}, failed={stats['failed']}, "
            f"hours≈{stats['total_hours_audio']:.2f}, "
            f"storage≈{stats['storage_used_bytes'] / (1024 ** 3):.2f} GiB"
        )
    except KeyboardInterrupt:
        print("\nInterrupted.")
        return 130
    print(f"Finished in {time.perf_counter() - t0:.2f}s")
    return 0


def cmd_status(args: argparse.Namespace) -> int:
    t0 = time.perf_counter()
    try:
        ckpt = CheckpointManager(args.checkpoint)
        p = ckpt.get_progress()
        print(
            f"Checkpoint {args.checkpoint!r}: "
            f"completed={p['completed']}, failed={p['failed']}, "
            f"in_progress={p['in_progress']}, total_attempted={p['total_attempted']}, "
            f"total_bytes={p['total_bytes']}, total_duration_hours={p['total_duration_hours']:.4f}"
        )
    except KeyboardInterrupt:
        print("\nInterrupted.")
        return 130
    print(f"Finished in {time.perf_counter() - t0:.2f}s")
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="dataset_builder")
    sub = p.add_subparsers(dest="command", required=True)

    d = sub.add_parser("discover", help="Find and list news stations")
    d.add_argument("--min-votes", type=int, default=0)
    d.add_argument("--languages", type=str, default=None)
    d.add_argument("--countries", type=str, default=None)
    d.add_argument("--limit", type=int, default=None)
    d.add_argument("--output", type=str, default=None)
    d.set_defaults(func=cmd_discover)

    b = sub.add_parser("bandwidth", help="Test bandwidth and capacity")
    b.add_argument("--samples", type=int, default=20)
    b.add_argument("--concurrent", type=int, default=10)
    b.add_argument("--duration", type=int, default=10)
    b.add_argument(
        "--scaling",
        action="store_true",
        help="Run concurrent scaling at [10,25,50,100,150,200]",
    )
    b.set_defaults(func=cmd_bandwidth)

    r = sub.add_parser("record", help="Record audio from stations")
    r.add_argument("--duration", type=int, default=1800)
    r.add_argument("--concurrent", type=int, default=50)
    r.add_argument("--sample-rate", type=int, default=16000)
    r.add_argument("--output-dir", type=str, default="recordings")
    r.add_argument("--min-votes", type=int, default=0)
    r.add_argument("--languages", type=str, default=None)
    r.add_argument("--countries", type=str, default=None)
    r.add_argument("--limit", type=int, default=None)
    r.add_argument("--resume", action="store_true")
    r.add_argument("--checkpoint", type=str, default="checkpoint.json")
    r.add_argument("--best-per-language", action="store_true")
    r.add_argument("--best-per-country", action="store_true")
    r.add_argument("--yes", action="store_true", help="Skip confirmation prompt")
    r.set_defaults(func=cmd_record)

    v = sub.add_parser("validate", help="Validate recorded audio")
    v.add_argument("--input-dir", type=str, default="recordings")
    v.add_argument("--min-duration", type=float, default=1700.0)
    v.add_argument("--max-silence", type=float, default=0.5)
    v.set_defaults(func=cmd_validate)

    bd = sub.add_parser("build-dataset", help="Build HuggingFace dataset from checkpoint")
    bd.add_argument("--input-dir", type=str, required=True)
    bd.add_argument("--output-dir", type=str, default="hf_dataset")
    bd.add_argument("--checkpoint", type=str, default="checkpoint.json")
    bd.set_defaults(func=cmd_build_dataset)

    st = sub.add_parser("status", help="Show checkpoint progress")
    st.add_argument("--checkpoint", type=str, default="checkpoint.json")
    st.set_defaults(func=cmd_status)

    return p


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
