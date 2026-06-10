"""Yahoo Finance data source via the *yfinance* package.

Ticker mappings (tried in order for each canonical symbol):
  XAUUSD  → ["GC=F", "XAUUSD=X"]
  XAGUSD  → ["SI=F", "XAGUSD=X"]
  EURUSD  → ["EURUSD=X"]
  BTCUSD  → ["BTC-USD"]
  ETHUSD  → ["ETH-USD"]
  SOLUSD  → ["SOL-USD"]
  XRPUSD  → ["XRP-USD"]
  (most FX pairs) → ["<BASE><QUOTE>=X"]

Supported timeframes: ``"1d"`` and ``"1h"``.
"""

from __future__ import annotations

import logging

import pandas as pd

from fxlab.core import DataSourceError, validate_ohlcv

log = logging.getLogger(__name__)

# Explicit ticker override lists; None means auto-construct "<SYM>=X"
_TICKER_MAP: dict[str, list[str]] = {
    "XAUUSD": ["GC=F", "XAUUSD=X"],
    "XAGUSD": ["SI=F", "XAGUSD=X"],
    "BTCUSD": ["BTC-USD", "BTCUSD=X"],
    "ETHUSD": ["ETH-USD"],
    "SOLUSD": ["SOL-USD"],
    "XRPUSD": ["XRP-USD"],
    "EURUSD": ["EURUSD=X"],
    "USDJPY": ["USDJPY=X"],
    "GBPUSD": ["GBPUSD=X"],
    "AUDUSD": ["AUDUSD=X"],
    "USDCAD": ["USDCAD=X"],
    "USDCHF": ["USDCHF=X"],
    "NZDUSD": ["NZDUSD=X"],
    "EURJPY": ["EURJPY=X"],
    "GBPJPY": ["GBPJPY=X"],
}

_INTERVAL_MAP: dict[str, str] = {
    "1w": "1wk",
    "1d": "1d",
    "4h": "1h",   # yfinance has no 4h; returned as 1h (caller may resample)
    "1h": "1h",
    "30m": "30m",
    "15m": "15m",
    "5m":  "5m",
}

# Yahoo Finance intraday data availability limits (days back from today):
#   1h / 4h : ~730 days
#   30m      : ~60 days
#   15m / 5m : ~60 days
_INTRADAY_MAX_DAYS: dict[str, int] = {
    "1h":  729,
    "30m": 59,
    "15m": 59,
    "5m":  59,
}


def _history_kwargs(
    interval: str, start: str | None, end: str | None
) -> dict[str, str]:
    """Build the date-range kwargs for ``Ticker.history``.

    * Intraday: clamp ``start`` into Yahoo's availability window so a too-early
      start does not silently return an empty frame.
    * Daily / weekly with no ``start``: use ``period="max"`` (yfinance's
      default period is only one month).
    """
    kwargs: dict[str, str] = {}
    max_days = _INTRADAY_MAX_DAYS.get(interval)
    if max_days is not None:
        limit = pd.Timestamp.utcnow().tz_localize(None).normalize() - pd.Timedelta(
            days=max_days
        )
        s = pd.Timestamp(start) if start else None
        clamped = limit if s is None or s < limit else s
        kwargs["start"] = clamped.strftime("%Y-%m-%d")
        if end:
            kwargs["end"] = end
    elif start:
        kwargs["start"] = start
        if end:
            kwargs["end"] = end
    else:
        kwargs["period"] = "max"
        if end:
            kwargs["end"] = end
    return kwargs


def _tickers_for(symbol: str) -> list[str]:
    """Return the list of yfinance tickers to try for *symbol*."""
    sym = symbol.upper()
    if sym in _TICKER_MAP:
        return _TICKER_MAP[sym]
    # Generic FX-style fallback
    return [f"{sym}=X"]


def fetch(
    symbol: str,
    timeframe: str = "1d",
    start: str | None = None,
    end: str | None = None,
) -> pd.DataFrame:
    """Download OHLCV data from Yahoo Finance for *symbol*.

    Tries each ticker in ``_TICKER_MAP[symbol]`` in order and returns the
    first successful result.

    Parameters
    ----------
    symbol:
        Canonical symbol (e.g. ``"EURUSD"``).
    timeframe:
        Bar size — ``"1d"`` or ``"1h"``.
    start:
        Optional ISO-8601 start date.
    end:
        Optional ISO-8601 end date.

    Returns
    -------
    pd.DataFrame
        Validated OHLCV frame.

    Raises
    ------
    DataSourceError
        If all tickers fail or data cannot be validated.
    """
    try:
        import yfinance as yf  # noqa: PLC0415
    except ImportError as exc:
        raise DataSourceError("yfinance package is not installed") from exc

    interval = _INTERVAL_MAP.get(timeframe, "1d")
    tickers = _tickers_for(symbol)
    last_exc: Exception | None = None

    for ticker in tickers:
        log.debug("yfinance fetch: %s (ticker=%s, interval=%s)", symbol, ticker, interval)
        try:
            tkr = yf.Ticker(ticker)
            df = tkr.history(
                interval=interval,
                auto_adjust=True,
                actions=False,
                **_history_kwargs(interval, start, end),
            )
        except Exception as exc:  # noqa: BLE001
            log.debug("yfinance error for %s: %s", ticker, exc)
            last_exc = exc
            continue

        if df is None or df.empty:
            log.debug("yfinance: empty result for %s", ticker)
            last_exc = DataSourceError(f"yfinance returned empty data for {ticker!r}")
            continue

        # Flatten possible MultiIndex columns (occurs for multi-ticker downloads)
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = ["_".join(str(c) for c in col).strip("_") for col in df.columns]

        # Normalise column names
        df.columns = [c.lower() for c in df.columns]

        # yfinance may return "stock splits", "dividends" etc. — keep only OHLCV
        col_map: dict[str, str] = {}
        for col in df.columns:
            if col.startswith("open"):
                col_map[col] = "open"
            elif col.startswith("high"):
                col_map[col] = "high"
            elif col.startswith("low"):
                col_map[col] = "low"
            elif col.startswith("close"):
                col_map[col] = "close"
            elif col.startswith("vol"):
                col_map[col] = "volume"
        df = df.rename(columns=col_map)

        # Ensure volume exists
        if "volume" not in df.columns:
            df["volume"] = 0.0

        # Remove tz info
        if isinstance(df.index, pd.DatetimeIndex) and df.index.tz is not None:
            df.index = df.index.tz_convert(None)

        df.sort_index(inplace=True)
        df = df[~df.index.duplicated(keep="last")]
        df.index.name = None

        # Drop rows where OHLC is NaN
        df = df.dropna(subset=["open", "high", "low", "close"])

        if df.empty:
            last_exc = DataSourceError(f"yfinance: all rows NaN after cleaning for {ticker!r}")
            continue

        try:
            return validate_ohlcv(df)
        except ValueError as exc:
            log.debug("yfinance validation failed for %s: %s", ticker, exc)
            last_exc = exc
            continue

    raise DataSourceError(
        f"yfinance could not fetch {symbol!r} (tried {tickers}): {last_exc}. "
        "Use --source synthetic or CSV import as offline fallbacks."
    ) from last_exc
