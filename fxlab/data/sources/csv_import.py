"""CSV import — robustly parse common OHLCV export formats.

Supported formats:
* TradingView  (``time,open,high,low,close[,volume]``)
* MT4 / MT5    (``Date,Time,Open,High,Low,Close,Volume`` or with ``Tickvol``)
* Dukascopy    (various date/time column names, semicolon-delimited)
* Generic      (any CSV with recognisable column names and a date/datetime column)

The parsed frame is validated with ``core.validate_ohlcv`` and written into
the parquet cache via ``store.save``.
"""

from __future__ import annotations

import io
import logging
import re
from pathlib import Path

import pandas as pd

from fxlab.core import DataSourceError, validate_ohlcv
from fxlab.data import store

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Column name normalisation maps
# ---------------------------------------------------------------------------

_DATE_COLS = {"time", "date", "datetime", "timestamp", "gmt time", "local time", "date/time"}
_TIME_COLS = {"time"}
_OPEN_COLS = {"open", "o"}
_HIGH_COLS = {"high", "h"}
_LOW_COLS = {"low", "l"}
_CLOSE_COLS = {"close", "c", "last"}
_VOL_COLS = {"volume", "vol", "tickvol", "tick vol", "real volume"}


def _normalise_col(name: str) -> str:
    """Lowercase and strip a column name."""
    return name.strip().lower()


def _detect_delimiter(text: str) -> str:
    """Heuristically detect the CSV delimiter."""
    first_line = text.splitlines()[0] if text else ""
    for delim in (",", ";", "\t", "|"):
        if delim in first_line:
            return delim
    return ","


def _find_col(columns: list[str], candidates: set[str]) -> str | None:
    """Return the first column whose normalised name is in *candidates*."""
    for col in columns:
        if _normalise_col(col) in candidates:
            return col
    return None


def _parse_frame(text: str) -> pd.DataFrame:
    """Parse raw CSV text into a raw DataFrame, trying common delimiters."""
    delim = _detect_delimiter(text)
    df = pd.read_csv(io.StringIO(text), sep=delim, engine="python", skipinitialspace=True)
    if df.shape[1] == 1:
        # Try other delimiters
        for alt in (",", ";", "\t"):
            if alt == delim:
                continue
            df2 = pd.read_csv(io.StringIO(text), sep=alt, engine="python", skipinitialspace=True)
            if df2.shape[1] > 1:
                return df2
    return df


def _try_parse_datetime(raw: "pd.Series") -> "pd.DatetimeIndex":
    """Try multiple strategies to parse a Series as datetime.

    Works with pandas ≥ 2.0 (``infer_datetime_format`` was removed in 3.0).
    """
    # Strategy 1: standard pandas parse (handles ISO 8601, many common formats)
    try:
        parsed = pd.to_datetime(raw, utc=False)
        if parsed.dt.tz is not None:
            parsed = parsed.dt.tz_localize(None)
        return pd.DatetimeIndex(parsed)
    except Exception:
        pass

    # Strategy 2: dayfirst formats (e.g. 03.01.2022 00:00:00.000)
    try:
        parsed = pd.to_datetime(raw, dayfirst=True, utc=False)
        if parsed.dt.tz is not None:
            parsed = parsed.dt.tz_localize(None)
        return pd.DatetimeIndex(parsed)
    except Exception:
        pass

    # Strategy 3: explicit format guessing for dot-separated / slash dates
    for fmt in (
        "%d.%m.%Y %H:%M:%S.%f",
        "%d.%m.%Y %H:%M:%S",
        "%d.%m.%Y %H:%M",
        "%d.%m.%Y",
        "%Y.%m.%d %H:%M:%S",
        "%Y.%m.%d",
        "%m/%d/%Y %H:%M",
        "%m/%d/%Y",
    ):
        try:
            parsed = pd.to_datetime(raw, format=fmt, utc=False)
            if parsed.dt.tz is not None:
                parsed = parsed.dt.tz_localize(None)
            return pd.DatetimeIndex(parsed)
        except Exception:
            continue

    # Strategy 4: unix timestamp in seconds
    try:
        parsed = pd.to_datetime(raw.astype("int64"), unit="s", utc=False)
        if parsed.dt.tz is not None:
            parsed = parsed.dt.tz_localize(None)
        return pd.DatetimeIndex(parsed)
    except Exception as exc:
        raise DataSourceError(
            f"csv_import: cannot parse datetime column with values like {raw.iloc[0]!r}: {exc}"
        ) from exc


def _build_datetime_index(df: pd.DataFrame) -> pd.DatetimeIndex:
    """Construct a DatetimeIndex from the DataFrame, handling various formats.

    Looks for a single date/datetime column, or a Date + Time column pair
    (common in MT4/MT5 exports).
    """
    cols_lower = {_normalise_col(c): c for c in df.columns}

    # Check for separate "Date" AND "Time" columns (MT4/MT5 style)
    # Both must be present; a lone "time" column is treated as a full datetime
    date_col = cols_lower.get("date")
    time_col = cols_lower.get("time") if date_col is not None else None

    if date_col and time_col:
        combined = df[date_col].astype(str) + " " + df[time_col].astype(str)
        return _try_parse_datetime(combined)

    # Single date/datetime column — search in priority order
    priority = ["time", "datetime", "timestamp", "date", "gmt time", "local time", "date/time"]
    dt_col = None
    for candidate in priority:
        if candidate in cols_lower:
            dt_col = cols_lower[candidate]
            break
    if dt_col is None:
        # Fall back: any column whose name contains "date" or "time"
        for lower, orig in cols_lower.items():
            if "date" in lower or "time" in lower:
                dt_col = orig
                break
    if dt_col is None:
        raise DataSourceError("csv_import: could not identify a date/time column")

    return _try_parse_datetime(df[dt_col])


def import_csv(
    path: str | Path,
    symbol: str,
    timeframe: str = "1d",
) -> pd.DataFrame:
    """Parse *path* as a CSV file and write it into the OHLCV parquet cache.

    The function autodetects:
    * Delimiter (comma, semicolon, tab, pipe)
    * Date/time column(s) (TradingView, MT4/MT5, Dukascopy, generic)
    * Column names (case-insensitive; see ``_OPEN_COLS`` etc.)

    The parsed frame is validated with :func:`~fxlab.core.validate_ohlcv` and
    stored via :func:`~fxlab.data.store.save`.

    Parameters
    ----------
    path:
        Filesystem path to the CSV file.
    symbol:
        Canonical symbol name (e.g. ``"EURUSD"``).
    timeframe:
        Bar timeframe string (e.g. ``"1d"``).

    Returns
    -------
    pd.DataFrame
        Validated OHLCV frame.

    Raises
    ------
    DataSourceError
        If the file cannot be found, parsed, or validated.
    """
    path = Path(path)
    if not path.exists():
        raise DataSourceError(f"csv_import: file not found: {path}")

    try:
        text = path.read_text(encoding="utf-8-sig")  # handle BOM
    except UnicodeDecodeError:
        try:
            text = path.read_text(encoding="latin-1")
        except Exception as exc:
            raise DataSourceError(f"csv_import: cannot read {path}: {exc}") from exc

    try:
        raw_df = _parse_frame(text)
    except Exception as exc:
        raise DataSourceError(f"csv_import: CSV parse error for {path}: {exc}") from exc

    if raw_df.empty or raw_df.shape[1] < 2:
        raise DataSourceError(f"csv_import: empty or single-column CSV at {path}")

    # Build datetime index
    try:
        index = _build_datetime_index(raw_df)
    except DataSourceError:
        raise
    except Exception as exc:
        raise DataSourceError(f"csv_import: datetime parse failed for {path}: {exc}") from exc

    cols_lower = {_normalise_col(c): c for c in raw_df.columns}

    def _get(candidates: set[str]) -> pd.Series | None:
        for cand in candidates:
            if cand in cols_lower:
                return raw_df[cols_lower[cand]]
        return None

    open_s = _get(_OPEN_COLS)
    high_s = _get(_HIGH_COLS)
    low_s = _get(_LOW_COLS)
    close_s = _get(_CLOSE_COLS)
    vol_s = _get(_VOL_COLS)

    missing = [n for n, s in [("open", open_s), ("high", high_s), ("low", low_s), ("close", close_s)] if s is None]
    if missing:
        raise DataSourceError(f"csv_import: missing required columns {missing} in {path}")

    def _to_float_array(s: "pd.Series") -> "np.ndarray":
        """Convert Series to float array, bypassing integer index alignment."""
        return pd.to_numeric(s, errors="coerce").to_numpy()

    import numpy as np  # noqa: PLC0415

    df = pd.DataFrame(
        {
            "open": _to_float_array(open_s),
            "high": _to_float_array(high_s),
            "low": _to_float_array(low_s),
            "close": _to_float_array(close_s),
            "volume": _to_float_array(vol_s) if vol_s is not None else np.zeros(len(index)),
        },
        index=index,
    )
    df["volume"] = df["volume"].fillna(0.0)

    # Drop rows with NaN in OHLC
    df = df.dropna(subset=["open", "high", "low", "close"])

    if df.empty:
        raise DataSourceError(f"csv_import: no valid OHLC rows in {path}")

    df.sort_index(inplace=True)
    df = df[~df.index.duplicated(keep="last")]
    df.index.name = None

    try:
        validated = validate_ohlcv(df)
    except ValueError as exc:
        raise DataSourceError(f"csv_import: validation failed for {path}: {exc}") from exc

    # Write to cache under source="csv"
    store.save(validated, symbol=symbol, timeframe=timeframe, source="csv")
    log.info("csv_import: imported %d rows for %s/%s from %s", len(validated), symbol, timeframe, path)
    return validated
