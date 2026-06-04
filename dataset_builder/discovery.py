from __future__ import annotations

from dataclasses import fields

from pyradios import RadioBrowser

from dataset_builder.config import StationInfo, normalize_language


def _station_from_row(row: dict) -> StationInfo:
    station_field_names = {f.name for f in fields(StationInfo)}
    kwargs = {
        k: v
        for k, v in row.items()
        if k in station_field_names and k != "language_canonical"
    }
    raw_language = row.get("language") or ""
    kwargs["language_canonical"] = normalize_language(
        raw_language if isinstance(raw_language, str) else str(raw_language)
    )
    return StationInfo(**kwargs)


def discover_all_news_stations() -> list[StationInfo]:
    rb = RadioBrowser()
    rows = rb.search(tag="news", limit=100000, hidebroken=True)
    return [_station_from_row(row) for row in rows]


def deduplicate_stations(stations: list[StationInfo]) -> list[StationInfo]:
    nonempty = [s for s in stations if (s.url_resolved or "").strip()]
    seen: set[str] = set()
    out: list[StationInfo] = []
    for s in sorted(nonempty, key=lambda x: int(x.votes or 0), reverse=True):
        u = (s.url_resolved or "").strip()
        if u in seen:
            continue
        seen.add(u)
        out.append(s)
    return out


def best_per_language(stations: list[StationInfo]) -> dict[str, StationInfo]:
    best: dict[str, StationInfo] = {}
    for s in stations:
        key = s.language_canonical
        cur = best.get(key)
        if cur is None or int(s.votes or 0) > int(cur.votes or 0):
            best[key] = s
    return best


def best_per_country(stations: list[StationInfo]) -> dict[str, StationInfo]:
    best: dict[str, StationInfo] = {}
    for s in stations:
        key = s.countrycode or ""
        cur = best.get(key)
        if cur is None or int(s.votes or 0) > int(cur.votes or 0):
            best[key] = s
    return best


def filter_stations(
    stations: list[StationInfo],
    min_votes: int = 0,
    languages: list[str] | None = None,
    countries: list[str] | None = None,
    limit: int | None = None,
) -> list[StationInfo]:
    out = [s for s in stations if int(s.votes or 0) >= min_votes]
    if languages is not None:
        allowed = {normalize_language(x) for x in languages}
        out = [s for s in out if s.language_canonical in allowed]
    if countries is not None:
        allowed_cc = {c.upper() for c in countries}
        out = [s for s in out if (s.countrycode or "").upper() in allowed_cc]
    out.sort(key=lambda s: int(s.votes or 0), reverse=True)
    if limit is not None:
        out = out[:limit]
    return out
