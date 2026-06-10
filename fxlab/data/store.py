"""Parquet-based local cache for OHLCV data.

Cache location: <repo_root>/data/cache/{symbol}_{timeframe}_{source}.parquet
"""

from __future__ import annotations

import logging
import re
from pathlib import Path

import pandas as pd

from fxlab.core import validate_ohlcv

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Cache root resolution
# ---------------------------------------------------------------------------

def _repo_root() -> Path:
    """Return the repository root (parent of the fxlab package directory)."""
    return Path(__file__).resolve().parent.parent.parent


def _cache_dir() -> Path:
    """Return the cache directory, creating it if needed."""
    d = _repo_root() / "data" / "cache"
    d.mkdir(parents=True, exist_ok=True)
    return d


def _cache_path(symbol: str, timeframe: str, source: str) -> Path:
    """Return the parquet path for a given (symbol, timeframe, source) triple."""
    # Sanitize components to avoid path injection
    safe = re.compile(r"[^A-Za-z0-9_\-]")
    s = safe.sub("_", symbol)
    t = safe.sub("_", timeframe)
    src = safe.sub("_", source)
    return _cache_dir() / f"{s}_{t}_{src}.parquet"


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def save(df: pd.DataFrame, symbol: str, timeframe: str, source: str) -> Path:
    """Validate and write *df* to the parquet cache.

    Returns the path where the file was written.
    """
    df = validate_ohlcv(df)
    # Strip the DatetimeIndex freq metadata before writing; parquet round-trips
    # may silently change it, causing assert_frame_equal failures.
    df = df.copy()
    df.index.freq = None
    path = _cache_path(symbol, timeframe, source)
    df.to_parquet(path, engine="pyarrow", compression="snappy")
    log.debug("Cached %d rows → %s", len(df), path)
    return path


def load(
    symbol: str,
    timeframe: str,
    source: str,
    start: str | None = None,
    end: str | None = None,
) -> pd.DataFrame | None:
    """Load a frame from the parquet cache.

    Returns *None* if no cached file exists.  Optionally filters by *start*
    and *end* (inclusive ISO-8601 date strings or ``None``).
    """
    path = _cache_path(symbol, timeframe, source)
    if not path.exists():
        return None
    df = pd.read_parquet(path, engine="pyarrow")
    if start is not None:
        df = df.loc[df.index >= pd.Timestamp(start)]
    if end is not None:
        df = df.loc[df.index <= pd.Timestamp(end)]
    if df.empty:
        return None
    return df


def merge(
    existing: pd.DataFrame,
    new: pd.DataFrame,
    symbol: str,
    timeframe: str,
    source: str,
) -> pd.DataFrame:
    """Union *existing* and *new* frames (new data wins on overlapping dates).

    Writes the merged result back to the cache and returns it.
    """
    combined = pd.concat([existing, new])
    # new wins: keep last occurrence where index duplicates
    combined = combined[~combined.index.duplicated(keep="last")]
    combined.sort_index(inplace=True)
    save(combined, symbol, timeframe, source)
    return combined


def inventory() -> pd.DataFrame:
    """Return a DataFrame describing all cached files.

    Columns: symbol, timeframe, source, rows, start, end.
    """
    rows: list[dict] = []
    for path in sorted(_cache_dir().glob("*.parquet")):
        stem = path.stem  # e.g. XAUUSD_1d_stooq
        parts = stem.rsplit("_", 2)
        if len(parts) != 3:
            log.warning("Unexpected cache filename: %s — skipping", path.name)
            continue
        sym, tf, src = parts
        try:
            df = pd.read_parquet(path, engine="pyarrow", columns=["close"])
            rows.append(
                {
                    "symbol": sym,
                    "timeframe": tf,
                    "source": src,
                    "rows": len(df),
                    "start": df.index.min(),
                    "end": df.index.max(),
                }
            )
        except Exception as exc:  # noqa: BLE001
            log.warning("Could not read %s: %s", path.name, exc)
    if not rows:
        return pd.DataFrame(
            columns=["symbol", "timeframe", "source", "rows", "start", "end"]
        )
    return pd.DataFrame(rows)
