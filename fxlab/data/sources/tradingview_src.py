"""TradingView data source via the *tvdatafeed* package.

The original StreamAlpha/tvdatafeed repository was removed from GitHub;
install the maintained fork instead:

    pip install git+https://github.com/rongardF/tvdatafeed.git

Credentials (tried in order):
  1. Explicit username/password arguments
  2. Env vars FXLAB_TV_USERNAME / FXLAB_TV_PASSWORD
  3. Anonymous (limited data, no intraday)
"""
from __future__ import annotations
import logging
import os
import pandas as pd
from fxlab.core import DataSourceError, validate_ohlcv

log = logging.getLogger(__name__)

# Map canonical symbols to (exchange, tv_symbol) tuples
_SYMBOL_MAP: dict[str, tuple[str, str]] = {
    "XAUUSD": ("OANDA", "XAUUSD"),
    "XAGUSD": ("OANDA", "XAGUSD"),
    "EURUSD": ("OANDA", "EURUSD"),
    "USDJPY": ("OANDA", "USDJPY"),
    "GBPUSD": ("OANDA", "GBPUSD"),
    "AUDUSD": ("OANDA", "AUDUSD"),
    "USDCAD": ("OANDA", "USDCAD"),
    "USDCHF": ("OANDA", "USDCHF"),
    "NZDUSD": ("OANDA", "NZDUSD"),
    "EURJPY": ("OANDA", "EURJPY"),
    "GBPJPY": ("OANDA", "GBPJPY"),
    "BTCUSD": ("COINBASE", "BTCUSD"),
    "ETHUSD": ("COINBASE", "ETHUSD"),
    "SOLUSD": ("COINBASE", "SOLUSD"),
    "XRPUSD": ("COINBASE", "XRPUSD"),
}

_INTERVAL_MAP: dict[str, object] = {}  # populated lazily after import


def _import_tvdatafeed():
    """Import the tvdatafeed package (module name varies between forks)."""
    try:
        from tvDatafeed import Interval, TvDatafeed  # noqa: PLC0415
    except ImportError:
        try:
            from tvdatafeed import Interval, TvDatafeed  # noqa: PLC0415
        except ImportError as exc:
            raise DataSourceError(
                "tvdatafeed is not installed. "
                "Run: pip install git+https://github.com/rongardF/tvdatafeed.git"
            ) from exc
    return TvDatafeed, Interval


def _interval(timeframe: str, Interval):
    """Return tvdatafeed Interval enum value for *timeframe*."""
    mapping = {
        "1d": Interval.in_daily,
        "1h": Interval.in_1_hour,
        "4h": Interval.in_4_hour,
        "15m": Interval.in_15_minute,
        "5m": Interval.in_5_minute,
    }
    iv = mapping.get(timeframe)
    if iv is None:
        raise DataSourceError(f"TradingView: unsupported timeframe {timeframe!r}")
    return iv


def fetch(
    symbol: str,
    timeframe: str = "1d",
    start: str | None = None,
    end: str | None = None,
    username: str | None = None,
    password: str | None = None,
    n_bars: int = 5000,
) -> pd.DataFrame:
    """Download OHLCV data from TradingView for *symbol*.

    Parameters
    ----------
    symbol:    Canonical symbol (e.g. ``"EURUSD"``).
    timeframe: Bar size — ``"1d"``, ``"4h"``, ``"1h"``, etc.
    start:     Optional ISO-8601 start date (used for post-download filtering only).
    end:       Optional ISO-8601 end date (used for post-download filtering only).
    username:  TradingView username (overrides env var).
    password:  TradingView password (overrides env var).
    n_bars:    Number of bars to request (default 5000).
    """
    TvDatafeed, Interval = _import_tvdatafeed()

    sym = symbol.upper()
    if sym in _SYMBOL_MAP:
        exchange, tv_sym = _SYMBOL_MAP[sym]
    else:
        # Generic fallback: try OANDA for FX, COINBASE for crypto
        exchange = "OANDA"
        tv_sym = sym

    u = username or os.environ.get("FXLAB_TV_USERNAME")
    p = password or os.environ.get("FXLAB_TV_PASSWORD")

    try:
        if u and p:
            tv = TvDatafeed(username=u, password=p)
            log.debug("TradingView: authenticated as %s", u)
        else:
            tv = TvDatafeed()
            log.debug("TradingView: anonymous session (limited data)")
    except Exception as exc:  # noqa: BLE001
        raise DataSourceError(f"TradingView login failed: {exc}") from exc

    interval = _interval(timeframe, Interval)
    log.debug("TradingView fetch: %s/%s interval=%s n_bars=%d", exchange, tv_sym, timeframe, n_bars)
    try:
        df = tv.get_hist(symbol=tv_sym, exchange=exchange, interval=interval, n_bars=n_bars)
    except Exception as exc:  # noqa: BLE001
        raise DataSourceError(f"TradingView fetch error for {sym}: {exc}") from exc

    if df is None or df.empty:
        raise DataSourceError(f"TradingView returned empty data for {sym!r}")

    # Normalize columns
    df.columns = [c.lower() for c in df.columns]
    if "volume" not in df.columns:
        df["volume"] = 0.0

    # Remove tz info
    if isinstance(df.index, pd.DatetimeIndex) and df.index.tz is not None:
        df.index = df.index.tz_convert(None)

    df.sort_index(inplace=True)
    df = df[~df.index.duplicated(keep="last")]
    df.index.name = None
    df = df.dropna(subset=["open", "high", "low", "close"])

    # Apply start/end filters
    if start:
        df = df[df.index >= pd.Timestamp(start)]
    if end:
        df = df[df.index <= pd.Timestamp(end)]

    if df.empty:
        raise DataSourceError(f"TradingView: no data after filtering for {sym!r}")

    return validate_ohlcv(df)
