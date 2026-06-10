"""Stooq data source — free daily OHLCV via CSV download.

URL pattern: https://stooq.com/q/d/l/?s={ticker}&i=d
Tickers are lowercase (e.g. xauusd, eurusd, btcusd).
Only daily (1d) timeframe is supported.
"""

from __future__ import annotations

import io
import logging

import pandas as pd
import requests

from fxlab.core import DataSourceError, validate_ohlcv

log = logging.getLogger(__name__)

_BASE_URL = "https://stooq.com/q/d/l/"
_TIMEOUT = 30  # seconds


def fetch(
    ticker: str,
    start: str | None = None,
    end: str | None = None,
) -> pd.DataFrame:
    """Download daily OHLCV data from Stooq for *ticker*.

    Parameters
    ----------
    ticker:
        Stooq ticker symbol (lowercase), e.g. ``"xauusd"``, ``"eurusd"``.
    start:
        Optional ISO-8601 start date string.
    end:
        Optional ISO-8601 end date string.

    Returns
    -------
    pd.DataFrame
        Validated OHLCV frame with a tz-naive DatetimeIndex.

    Raises
    ------
    DataSourceError
        On any network or parsing failure.
    """
    params: dict[str, str] = {"s": ticker.lower(), "i": "d"}
    if start:
        params["d1"] = pd.Timestamp(start).strftime("%Y%m%d")
    if end:
        params["d2"] = pd.Timestamp(end).strftime("%Y%m%d")

    log.debug("stooq fetch: %s params=%s", ticker, params)
    try:
        resp = requests.get(_BASE_URL, params=params, timeout=_TIMEOUT)
        resp.raise_for_status()
    except requests.RequestException as exc:
        raise DataSourceError(f"stooq network error for {ticker!r}: {exc}") from exc

    text = resp.text.strip()
    if not text or "No data" in text or len(text.splitlines()) < 2:
        raise DataSourceError(
            f"stooq returned no data for ticker {ticker!r}. "
            "Check the ticker or try --source synthetic / CSV import as offline fallbacks."
        )

    try:
        df = pd.read_csv(
            io.StringIO(text),
            parse_dates=["Date"],
            index_col="Date",
        )
    except Exception as exc:
        raise DataSourceError(f"stooq CSV parse error for {ticker!r}: {exc}") from exc

    # Normalise column names
    df.columns = [c.lower() for c in df.columns]
    rename = {}
    if "vol" in df.columns and "volume" not in df.columns:
        rename["vol"] = "volume"
    if rename:
        df.rename(columns=rename, inplace=True)

    # Ensure volume column exists (stooq sometimes omits it for FX)
    if "volume" not in df.columns:
        df["volume"] = 0.0

    df.index.name = None

    # Ensure the index is a DatetimeIndex
    if not isinstance(df.index, pd.DatetimeIndex):
        df.index = pd.to_datetime(df.index)

    # Remove timezone if present
    if df.index.tz is not None:
        df.index = df.index.tz_localize(None)

    df.sort_index(inplace=True)
    df = df[~df.index.duplicated(keep="last")]

    try:
        return validate_ohlcv(df)
    except ValueError as exc:
        raise DataSourceError(f"stooq data validation failed for {ticker!r}: {exc}") from exc
