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
    "1d": "1d",
    "1h": "1h",
    "4h": "1h",   # yfinance has no 4 h; caller may resample
}


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
                start=start,
                end=end,
                auto_adjust=True,
                actions=False,
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
