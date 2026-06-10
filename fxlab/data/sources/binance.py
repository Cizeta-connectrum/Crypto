"""Binance Klines data source (crypto only).

API endpoint: https://api.binance.com/api/v3/klines
Paginates using *startTime* / *limit=1000* to cover the full requested range.
Default range: 5 years for daily; proportionally shorter for intraday.
"""

from __future__ import annotations

import logging
import time
from datetime import datetime, timedelta, timezone

import pandas as pd
import requests

from fxlab.core import DataSourceError, validate_ohlcv

log = logging.getLogger(__name__)

_BASE_URL = "https://api.binance.com/api/v3/klines"
_TIMEOUT = 30
_PAGE_SIZE = 1000

# Binance interval strings
_INTERVAL_MAP: dict[str, str] = {
    "1d": "1d",
    "4h": "4h",
    "1h": "1h",
    "30m": "30m",
    "15m": "15m",
    "5m": "5m",
    "1m": "1m",
    "1w": "1w",
}

# Approximate bar duration in seconds (for pagination)
_BAR_SECONDS: dict[str, int] = {
    "1d": 86_400,
    "4h": 14_400,
    "1h": 3_600,
    "30m": 1_800,
    "15m": 900,
    "5m": 300,
    "1m": 60,
    "1w": 604_800,
}

# Default lookback in days per timeframe
_DEFAULT_LOOKBACK_DAYS: dict[str, int] = {
    "1d": 365 * 5,
    "4h": 365,
    "1h": 180,
    "30m": 90,
    "15m": 30,
    "5m": 14,
    "1m": 3,
    "1w": 365 * 5,
}


def _ms_from_ts(ts: pd.Timestamp | datetime | str | None) -> int | None:
    """Convert a timestamp to Unix milliseconds, or return None."""
    if ts is None:
        return None
    if isinstance(ts, str):
        ts = pd.Timestamp(ts)
    if isinstance(ts, pd.Timestamp):
        return int(ts.value // 1_000_000)
    # datetime
    return int(ts.replace(tzinfo=timezone.utc).timestamp() * 1000)


def fetch(
    binance_symbol: str,
    timeframe: str = "1d",
    start: str | None = None,
    end: str | None = None,
) -> pd.DataFrame:
    """Download OHLCV klines from Binance.

    Parameters
    ----------
    binance_symbol:
        Binance pair symbol, e.g. ``"BTCUSDT"`` or ``"ETHUSDТ"``.
    timeframe:
        Bar size — one of ``"1d"``, ``"4h"``, ``"1h"``, ``"30m"``,
        ``"15m"``, ``"5m"``, ``"1m"``, ``"1w"``.
    start:
        ISO-8601 start date/datetime.  Defaults to a lookback period
        appropriate for the timeframe.
    end:
        ISO-8601 end date/datetime.  Defaults to now.

    Returns
    -------
    pd.DataFrame
        Validated OHLCV frame.

    Raises
    ------
    DataSourceError
        On any network or parsing failure.
    """
    interval = _INTERVAL_MAP.get(timeframe)
    if interval is None:
        raise DataSourceError(
            f"binance: unsupported timeframe {timeframe!r}. "
            f"Supported: {sorted(_INTERVAL_MAP)}"
        )

    # Resolve start / end timestamps
    now_ms = int(time.time() * 1000)

    if end is None:
        end_ms = now_ms
    else:
        end_ms = _ms_from_ts(end)

    if start is None:
        lookback_days = _DEFAULT_LOOKBACK_DAYS.get(timeframe, 365 * 5)
        start_ms = now_ms - lookback_days * 86_400 * 1000
    else:
        start_ms = _ms_from_ts(start)

    log.debug(
        "binance fetch: %s interval=%s start_ms=%d end_ms=%d",
        binance_symbol,
        interval,
        start_ms,
        end_ms,
    )

    all_rows: list[list] = []
    cursor_ms = start_ms

    while cursor_ms < end_ms:
        params = {
            "symbol": binance_symbol.upper(),
            "interval": interval,
            "startTime": cursor_ms,
            "endTime": end_ms,
            "limit": _PAGE_SIZE,
        }
        try:
            resp = requests.get(_BASE_URL, params=params, timeout=_TIMEOUT)
            resp.raise_for_status()
        except requests.RequestException as exc:
            raise DataSourceError(
                f"binance network error for {binance_symbol!r}: {exc}"
            ) from exc

        try:
            page = resp.json()
        except Exception as exc:
            raise DataSourceError(
                f"binance JSON parse error for {binance_symbol!r}: {exc}"
            ) from exc

        if isinstance(page, dict) and "code" in page:
            raise DataSourceError(
                f"binance API error for {binance_symbol!r}: {page}"
            )

        if not page:
            break

        all_rows.extend(page)

        # Advance cursor: open-time of last bar + bar duration
        last_open_ms = page[-1][0]
        bar_sec = _BAR_SECONDS.get(timeframe, 86_400)
        cursor_ms = last_open_ms + bar_sec * 1000

        if len(page) < _PAGE_SIZE:
            break  # last page

    if not all_rows:
        raise DataSourceError(
            f"binance returned no data for {binance_symbol!r}. "
            "Use --source synthetic or CSV import as offline fallbacks."
        )

    # Klines columns:
    # 0: Open time, 1: Open, 2: High, 3: Low, 4: Close, 5: Volume, ...
    df = pd.DataFrame(all_rows)
    df = df.iloc[:, :6]
    df.columns = ["open_time", "open", "high", "low", "close", "volume"]
    df["open_time"] = pd.to_datetime(df["open_time"], unit="ms", utc=True).dt.tz_localize(None)
    df.set_index("open_time", inplace=True)
    df.index.name = None
    df = df.astype("float64")

    df.sort_index(inplace=True)
    df = df[~df.index.duplicated(keep="last")]

    try:
        return validate_ohlcv(df)
    except ValueError as exc:
        raise DataSourceError(
            f"binance data validation failed for {binance_symbol!r}: {exc}"
        ) from exc
