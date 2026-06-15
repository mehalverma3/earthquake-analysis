"""
USGS earthquake data fetching.

Pulls event data from the USGS FDSN Event web service and returns a clean
pandas DataFrame. Handles the API's 20,000-event-per-request cap by splitting
large time windows into chunks, and caches results to disk to avoid hammering
the service on repeated runs.

API docs: https://earthquake.usgs.gov/fdsnws/event/1/
"""

from __future__ import annotations

import time
from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd
import requests

USGS_ENDPOINT = "https://earthquake.usgs.gov/fdsnws/event/1/query"
USGS_COUNT_ENDPOINT = "https://earthquake.usgs.gov/fdsnws/event/1/count"

# USGS rejects single requests that would return more than this many events.
MAX_EVENTS_PER_REQUEST = 20_000


def _count_events(params: dict) -> int:
    """Ask USGS how many events a query would return (cheap, no payload)."""
    p = dict(params)
    p["format"] = "text"
    resp = requests.get(USGS_COUNT_ENDPOINT, params=p, timeout=60)
    resp.raise_for_status()
    return int(resp.text.strip())


def _geojson_to_records(geojson: dict) -> list[dict]:
    """Flatten a USGS GeoJSON FeatureCollection into row dicts."""
    records = []
    for feature in geojson.get("features", []):
        props = feature.get("properties", {}) or {}
        geom = feature.get("geometry") or {}
        coords = geom.get("coordinates") or [None, None, None]
        lon, lat, depth = (coords + [None, None, None])[:3]
        records.append(
            {
                "id": feature.get("id"),
                "time": props.get("time"),       # epoch ms, UTC
                "magnitude": props.get("mag"),
                "mag_type": props.get("magType"),
                "place": props.get("place"),
                "longitude": lon,
                "latitude": lat,
                "depth_km": depth,
                "felt": props.get("felt"),
                "tsunami": props.get("tsunami"),
                "type": props.get("type"),
            }
        )
    return records


def _fetch_window(starttime: str, endtime: str, min_magnitude: float) -> list[dict]:
    """Fetch a single time window, recursively bisecting if it's too large."""
    params = {
        "format": "geojson",
        "starttime": starttime,
        "endtime": endtime,
        "minmagnitude": min_magnitude,
        "orderby": "time-asc",
    }

    count = _count_events(params)
    if count == 0:
        return []

    if count <= MAX_EVENTS_PER_REQUEST:
        resp = requests.get(USGS_ENDPOINT, params=params, timeout=120)
        resp.raise_for_status()
        return _geojson_to_records(resp.json())

    # Too many events: split the window in half and recurse.
    start_dt = datetime.fromisoformat(starttime)
    end_dt = datetime.fromisoformat(endtime)
    mid_dt = start_dt + (end_dt - start_dt) / 2
    mid = mid_dt.isoformat(timespec="seconds")
    print(f"  window {starttime}..{endtime} has {count} events; splitting at {mid}")
    time.sleep(0.5)  # be polite to the service
    left = _fetch_window(starttime, mid, min_magnitude)
    right = _fetch_window(mid, endtime, min_magnitude)
    return left + right


def fetch_earthquakes(
    starttime: str,
    endtime: str,
    min_magnitude: float = 2.5,
    cache_path: str | Path | None = "outputs/earthquakes_cache.csv",
    refresh: bool = False,
) -> pd.DataFrame:
    """
    Fetch earthquakes from USGS between two dates.

    Parameters
    ----------
    starttime, endtime : str
        ISO dates, e.g. "2024-01-01" or "2024-01-01T00:00:00".
    min_magnitude : float
        Minimum magnitude to retrieve. Lower values mean far more events.
    cache_path : str | Path | None
        Where to cache the cleaned DataFrame. Set to None to disable caching.
    refresh : bool
        If True, ignore any existing cache and re-fetch from USGS.

    Returns
    -------
    pandas.DataFrame
        One row per earthquake, with a tz-aware UTC ``datetime`` column.
    """
    cache_path = Path(cache_path) if cache_path else None

    if cache_path and cache_path.exists() and not refresh:
        print(f"Loading cached data from {cache_path}")
        df = pd.read_csv(cache_path, parse_dates=["datetime"])
        return df

    print(f"Fetching USGS earthquakes M>={min_magnitude} from {starttime} to {endtime}")
    records = _fetch_window(starttime, endtime, min_magnitude)
    df = pd.DataFrame.from_records(records)

    if df.empty:
        print("No events returned for this query.")
        return df

    df = _clean(df)
    print(f"Retrieved {len(df):,} earthquakes.")

    if cache_path:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(cache_path, index=False)
        print(f"Cached to {cache_path}")

    return df


def _clean(df: pd.DataFrame) -> pd.DataFrame:
    """Type conversions, derived columns, and basic quality filtering."""
    df = df.copy()
    df["datetime"] = pd.to_datetime(df["time"], unit="ms", utc=True)
    for col in ("magnitude", "longitude", "latitude", "depth_km"):
        df[col] = pd.to_numeric(df[col], errors="coerce")

    # Drop rows missing the fields every downstream step needs.
    df = df.dropna(subset=["magnitude", "longitude", "latitude"])

    # Keep tectonic earthquakes; USGS also reports quarry blasts, etc.
    if "type" in df.columns:
        df = df[df["type"].fillna("earthquake") == "earthquake"]

    df = df.sort_values("datetime").reset_index(drop=True)
    return df


def date_range_days(days_back: int) -> tuple[str, str]:
    """Convenience helper: (start, end) ISO strings for the last ``days_back`` days."""
    end = datetime.utcnow()
    start = end - timedelta(days=days_back)
    return start.strftime("%Y-%m-%d"), end.strftime("%Y-%m-%d")
