"""FXLab data layer — download, cache, and serve OHLCV market data.

Public API
----------
.. code-block:: python

    from fxlab.data import load_ohlcv, download, list_cached, SYMBOLS

    df = load_ohlcv("XAUUSD", timeframe="1d", source="synthetic")
    results = download(["EURUSD", "BTCUSD"], timeframe="1d")
    inventory = list_cached()

``SYMBOLS``
-----------
A dict mapping each canonical symbol to its metadata::

    {
        "XAUUSD": {
            "asset_class": "metal",
            "stooq":   "xauusd",
            "yfinance": ["GC=F", "XAUUSD=X"],
            "binance":  None,
        },
        ...
    }
"""

from __future__ import annotations

import logging
from typing import Any

import pandas as pd

from fxlab.core import DataSourceError, validate_ohlcv
from fxlab.data import store

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Canonical symbol registry
# ---------------------------------------------------------------------------

SYMBOLS: dict[str, dict[str, Any]] = {
    # ---- Metals ----
    "XAUUSD": {
        "asset_class": "metal",
        "stooq": "xauusd",
        "yfinance": ["GC=F", "XAUUSD=X"],
        "binance": None,
    },
    "XAGUSD": {
        "asset_class": "metal",
        "stooq": "xagusd",
        "yfinance": ["SI=F", "XAGUSD=X"],
        "binance": None,
    },
    # ---- FX pairs ----
    "EURUSD": {
        "asset_class": "fx",
        "stooq": "eurusd",
        "yfinance": ["EURUSD=X"],
        "binance": None,
    },
    "USDJPY": {
        "asset_class": "fx",
        "stooq": "usdjpy",
        "yfinance": ["USDJPY=X"],
        "binance": None,
    },
    "GBPUSD": {
        "asset_class": "fx",
        "stooq": "gbpusd",
        "yfinance": ["GBPUSD=X"],
        "binance": None,
    },
    "AUDUSD": {
        "asset_class": "fx",
        "stooq": "audusd",
        "yfinance": ["AUDUSD=X"],
        "binance": None,
    },
    "USDCAD": {
        "asset_class": "fx",
        "stooq": "usdcad",
        "yfinance": ["USDCAD=X"],
        "binance": None,
    },
    "USDCHF": {
        "asset_class": "fx",
        "stooq": "usdchf",
        "yfinance": ["USDCHF=X"],
        "binance": None,
    },
    "NZDUSD": {
        "asset_class": "fx",
        "stooq": "nzdusd",
        "yfinance": ["NZDUSD=X"],
        "binance": None,
    },
    "EURJPY": {
        "asset_class": "fx",
        "stooq": "eurjpy",
        "yfinance": ["EURJPY=X"],
        "binance": None,
    },
    "GBPJPY": {
        "asset_class": "fx",
        "stooq": "gbpjpy",
        "yfinance": ["GBPJPY=X"],
        "binance": None,
    },
    # ---- Crypto ----
    "BTCUSD": {
        "asset_class": "crypto",
        "stooq": "btcusd",
        "yfinance": ["BTC-USD"],
        "binance": "BTCUSDT",
    },
    "ETHUSD": {
        "asset_class": "crypto",
        "stooq": "ethusd",
        "yfinance": ["ETH-USD"],
        "binance": "ETHUSDT",
    },
    "SOLUSD": {
        "asset_class": "crypto",
        "stooq": "solusd",
        "yfinance": ["SOL-USD"],
        "binance": "SOLUSDT",
    },
    "XRPUSD": {
        "asset_class": "crypto",
        "stooq": "xrpusd",
        "yfinance": ["XRP-USD"],
        "binance": "XRPUSDT",
    },
}

# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _asset_class(symbol: str) -> str:
    return SYMBOLS.get(symbol.upper(), {}).get("asset_class", "fx")


def _fetch_stooq(symbol: str, timeframe: str, start: str | None, end: str | None) -> pd.DataFrame:
    from fxlab.data.sources import stooq  # noqa: PLC0415
    meta = SYMBOLS.get(symbol.upper(), {})
    ticker = meta.get("stooq", symbol.lower())
    return stooq.fetch(ticker, start=start, end=end)


def _fetch_yfinance(symbol: str, timeframe: str, start: str | None, end: str | None) -> pd.DataFrame:
    from fxlab.data.sources import yfinance_src  # noqa: PLC0415
    return yfinance_src.fetch(symbol, timeframe=timeframe, start=start, end=end)


def _fetch_binance(symbol: str, timeframe: str, start: str | None, end: str | None) -> pd.DataFrame:
    from fxlab.data.sources import binance  # noqa: PLC0415
    meta = SYMBOLS.get(symbol.upper(), {})
    binance_sym = meta.get("binance")
    if not binance_sym:
        raise DataSourceError(
            f"binance source not available for non-crypto symbol {symbol!r}"
        )
    return binance.fetch(binance_sym, timeframe=timeframe, start=start, end=end)


def _fetch_synthetic(symbol: str, timeframe: str, start: str | None, end: str | None) -> pd.DataFrame:
    from fxlab.data.sources import synthetic  # noqa: PLC0415
    ac = _asset_class(symbol)
    kw: dict[str, Any] = {"asset_class": ac}
    if start:
        kw["start"] = start
    if end:
        kw["end"] = end
    return synthetic.generate(symbol, timeframe=timeframe, **kw)


def _filter_dates(df: pd.DataFrame, start: str | None, end: str | None) -> pd.DataFrame:
    if start is not None:
        df = df.loc[df.index >= pd.Timestamp(start)]
    if end is not None:
        df = df.loc[df.index <= pd.Timestamp(end)]
    return df


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def load_ohlcv(
    symbol: str,
    timeframe: str = "1d",
    start: str | None = None,
    end: str | None = None,
    source: str = "auto",
    refresh: bool = False,
) -> pd.DataFrame:
    """Load OHLCV data for *symbol*, using the cache or fetching from a source.

    Parameters
    ----------
    symbol:
        Canonical symbol, e.g. ``"XAUUSD"``, ``"EURUSD"``, ``"BTCUSD"``.
    timeframe:
        Bar size, e.g. ``"1d"``, ``"1h"``, ``"4h"``, ``"15m"``.
    start:
        Optional ISO-8601 start date (e.g. ``"2020-01-01"``).
    end:
        Optional ISO-8601 end date.
    source:
        One of ``"auto"``, ``"stooq"``, ``"yfinance"``, ``"binance"``,
        ``"synthetic"``, ``"csv"``.

        * ``"auto"`` — returns the cache when present and *refresh* is
          ``False``; otherwise tries stooq → yfinance → binance (crypto
          only) in order.
        * Any explicit value uses exactly that source (except ``"csv"``,
          which only reads the existing cache written by
          :func:`~fxlab.data.sources.csv_import.import_csv`).
    refresh:
        When ``True``, re-download even if cached data exist and merge
        new rows in.

    Returns
    -------
    pd.DataFrame
        Validated OHLCV frame.

    Raises
    ------
    DataSourceError
        When the requested data cannot be obtained from any source.
    """
    sym = symbol.upper()

    # ---- "csv" source: load from cache only ----
    if source == "csv":
        cached = store.load(sym, timeframe, "csv", start=start, end=end)
        if cached is not None:
            return _filter_dates(cached, start, end)
        raise DataSourceError(
            f"No cached CSV data for {sym}/{timeframe}. "
            "Import a file first with fxlab.data.sources.csv_import.import_csv()."
        )

    # ---- "synthetic" source ----
    if source == "synthetic":
        df = _fetch_synthetic(sym, timeframe, start, end)
        # Drop freq so the result is consistent whether served live or from cache
        df = df.copy()
        df.index.freq = None
        # Cache synthetic results so they can be inspected offline
        try:
            store.save(df, sym, timeframe, "synthetic")
        except Exception as exc:  # noqa: BLE001
            log.debug("Could not cache synthetic data: %s", exc)
        return _filter_dates(df, start, end)

    # ---- "auto" source ----
    if source == "auto":
        # Check cache first (unless refresh requested)
        if not refresh:
            cached = store.load(sym, timeframe, "stooq", start=start, end=end)
            if cached is not None and not cached.empty:
                log.debug("auto: serving %s/%s from stooq cache", sym, timeframe)
                return cached
            cached = store.load(sym, timeframe, "yfinance", start=start, end=end)
            if cached is not None and not cached.empty:
                log.debug("auto: serving %s/%s from yfinance cache", sym, timeframe)
                return cached
            if _asset_class(sym) == "crypto":
                cached = store.load(sym, timeframe, "binance", start=start, end=end)
                if cached is not None and not cached.empty:
                    log.debug("auto: serving %s/%s from binance cache", sym, timeframe)
                    return cached

        # Try live sources
        errors: list[str] = []

        # 1. stooq (daily only)
        if timeframe == "1d":
            try:
                df = _fetch_stooq(sym, timeframe, start, end)
                _merge_and_cache(df, sym, timeframe, "stooq", refresh)
                return _filter_dates(df, start, end)
            except DataSourceError as exc:
                log.debug("stooq failed for %s: %s", sym, exc)
                errors.append(f"stooq: {exc}")

        # 2. yfinance
        try:
            df = _fetch_yfinance(sym, timeframe, start, end)
            _merge_and_cache(df, sym, timeframe, "yfinance", refresh)
            return _filter_dates(df, start, end)
        except DataSourceError as exc:
            log.debug("yfinance failed for %s: %s", sym, exc)
            errors.append(f"yfinance: {exc}")

        # 3. binance (crypto only)
        if _asset_class(sym) == "crypto":
            try:
                df = _fetch_binance(sym, timeframe, start, end)
                _merge_and_cache(df, sym, timeframe, "binance", refresh)
                return _filter_dates(df, start, end)
            except DataSourceError as exc:
                log.debug("binance failed for %s: %s", sym, exc)
                errors.append(f"binance: {exc}")

        raise DataSourceError(
            f"All data sources failed for {sym}/{timeframe}.\n"
            f"Errors: {'; '.join(errors)}\n"
            "Offline fallbacks: use --source synthetic (generated data) "
            "or import a CSV file with fxlab.data.sources.csv_import.import_csv()."
        )

    # ---- "tradingview" source ----
    if source == "tradingview":
        from fxlab.data.sources import tradingview_src  # noqa: PLC0415
        df = tradingview_src.fetch(sym, timeframe=timeframe, start=start, end=end)
        _merge_and_cache(df, sym, timeframe, "tradingview", refresh)
        return _filter_dates(df, start, end)

    # ---- Explicit source ----
    fetchers = {
        "stooq": _fetch_stooq,
        "yfinance": _fetch_yfinance,
        "binance": _fetch_binance,
    }
    fetcher = fetchers.get(source)
    if fetcher is None:
        raise DataSourceError(
            f"Unknown source {source!r}. "
            f"Valid values: auto, stooq, yfinance, binance, synthetic, csv, tradingview."
        )

    try:
        df = fetcher(sym, timeframe, start, end)
    except DataSourceError:
        raise
    except Exception as exc:
        raise DataSourceError(f"{source} fetch failed for {sym}: {exc}") from exc

    _merge_and_cache(df, sym, timeframe, source, refresh)
    return _filter_dates(df, start, end)


def _merge_and_cache(
    df: pd.DataFrame,
    symbol: str,
    timeframe: str,
    source: str,
    refresh: bool,
) -> None:
    """Persist *df* to the cache, merging with any existing rows."""
    try:
        existing = store.load(symbol, timeframe, source)
        if existing is not None and refresh:
            store.merge(existing, df, symbol, timeframe, source)
        elif existing is None:
            store.save(df, symbol, timeframe, source)
        # If existing and not refresh: leave cache alone (we already served it above)
    except Exception as exc:  # noqa: BLE001
        log.warning("Could not update cache for %s/%s/%s: %s", symbol, timeframe, source, exc)


def download(
    symbols: list[str],
    timeframe: str = "1d",
    source: str = "auto",
) -> dict[str, str]:
    """Download and cache OHLCV data for a list of symbols.

    Parameters
    ----------
    symbols:
        List of canonical symbol names.
    timeframe:
        Bar size.
    source:
        Data source (see :func:`load_ohlcv`).

    Returns
    -------
    dict[str, str]
        Mapping from symbol to status string (``"ok"`` or an error message).
    """
    results: dict[str, str] = {}
    for sym in symbols:
        try:
            df = load_ohlcv(sym, timeframe=timeframe, source=source, refresh=True)
            results[sym] = f"ok ({len(df)} rows)"
        except DataSourceError as exc:
            results[sym] = f"error: {exc}"
        except Exception as exc:  # noqa: BLE001
            results[sym] = f"unexpected error: {exc}"
    return results


def list_cached() -> pd.DataFrame:
    """Return a DataFrame describing all locally cached OHLCV files.

    Columns: ``symbol``, ``timeframe``, ``source``, ``rows``, ``start``,
    ``end``.
    """
    return store.inventory()
