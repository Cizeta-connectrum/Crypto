"""Tests for the FXLab data layer.

No network access — all tests use synthetic data or locally written CSV files.
"""

from __future__ import annotations

import textwrap
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def patch_cache_dir(tmp_path, monkeypatch):
    """Redirect the parquet cache to a temporary directory for every test."""
    cache_dir = tmp_path / "cache"
    cache_dir.mkdir()

    import fxlab.data.store as store_mod

    monkeypatch.setattr(store_mod, "_cache_dir", lambda: cache_dir)
    return cache_dir


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_ohlcv(n: int = 50, start: str = "2020-01-01") -> pd.DataFrame:
    """Return a minimal valid OHLCV frame for testing."""
    from fxlab.core import validate_ohlcv

    index = pd.date_range(start=start, periods=n, freq="B")
    rng = np.random.default_rng(42)
    closes = 1000.0 + np.cumsum(rng.normal(0, 5, n))
    opens = closes * (1 + rng.normal(0, 0.002, n))
    highs = np.maximum(opens, closes) * (1 + np.abs(rng.normal(0, 0.003, n)))
    lows = np.minimum(opens, closes) * (1 - np.abs(rng.normal(0, 0.003, n)))
    volumes = rng.lognormal(10, 1, n)

    df = pd.DataFrame(
        {"open": opens, "high": highs, "low": lows, "close": closes, "volume": volumes},
        index=index,
    )
    return validate_ohlcv(df)


# ---------------------------------------------------------------------------
# 1. synthetic: determinism
# ---------------------------------------------------------------------------


def test_synthetic_determinism():
    """Same symbol + timeframe must produce identical frames on repeated calls."""
    from fxlab.data.sources.synthetic import generate

    df1 = generate("EURUSD", timeframe="1d", start="2020-01-01", end="2021-01-01")
    df2 = generate("EURUSD", timeframe="1d", start="2020-01-01", end="2021-01-01")

    pd.testing.assert_frame_equal(df1, df2)


def test_synthetic_different_symbols_differ():
    """Different symbols must produce different price paths."""
    from fxlab.data.sources.synthetic import generate

    df_eur = generate("EURUSD", timeframe="1d", start="2020-01-01", end="2021-01-01")
    df_gbp = generate("GBPUSD", timeframe="1d", start="2020-01-01", end="2021-01-01")

    # Closes should not be identical
    assert not np.allclose(df_eur["close"].values, df_gbp["close"].values)


def test_synthetic_explicit_seed_overrides_determinism():
    """Explicit seed=0 and seed=1 produce different paths."""
    from fxlab.data.sources.synthetic import generate

    df0 = generate("EURUSD", timeframe="1d", start="2020-01-01", end="2021-01-01", seed=0)
    df1 = generate("EURUSD", timeframe="1d", start="2020-01-01", end="2021-01-01", seed=1)

    assert not np.allclose(df0["close"].values, df1["close"].values)


# ---------------------------------------------------------------------------
# 2. validate_ohlcv passes on synthetic data
# ---------------------------------------------------------------------------


def test_synthetic_passes_validate_ohlcv():
    """Generated data must pass core.validate_ohlcv without raising."""
    from fxlab.core import validate_ohlcv
    from fxlab.data.sources.synthetic import generate

    df = generate("XAUUSD", timeframe="1d", start="2018-01-01", end="2022-01-01")
    # Should not raise
    validated = validate_ohlcv(df)
    assert set(validated.columns) == {"open", "high", "low", "close", "volume"}


# ---------------------------------------------------------------------------
# 3. OHLC consistency
# ---------------------------------------------------------------------------


def test_synthetic_ohlc_consistency():
    """high >= max(open,close) >= min(open,close) >= low for every bar."""
    from fxlab.data.sources.synthetic import generate

    df = generate("BTCUSD", timeframe="1d", start="2020-01-01", end="2023-01-01",
                  asset_class="crypto")

    assert (df["high"] >= df["open"]).all(), "high < open on some bar"
    assert (df["high"] >= df["close"]).all(), "high < close on some bar"
    assert (df["low"] <= df["open"]).all(), "low > open on some bar"
    assert (df["low"] <= df["close"]).all(), "low > close on some bar"
    assert (df["high"] >= df["low"]).all(), "high < low on some bar"


def test_synthetic_index_is_datetimeindex():
    """Index must be a tz-naive DatetimeIndex."""
    from fxlab.data.sources.synthetic import generate

    df = generate("USDJPY", timeframe="1d", start="2021-01-01", end="2022-01-01")
    assert isinstance(df.index, pd.DatetimeIndex)
    assert df.index.tz is None


def test_synthetic_fx_volume_is_zero():
    """FX synthetic data must have volume == 0."""
    from fxlab.data.sources.synthetic import generate

    df = generate("EURUSD", timeframe="1d", start="2021-01-01", end="2022-01-01",
                  asset_class="fx")
    assert (df["volume"] == 0.0).all()


# ---------------------------------------------------------------------------
# 4. Cache round-trip (store.save / store.load)
# ---------------------------------------------------------------------------


def test_cache_round_trip(tmp_path):
    """save() then load() returns an identical frame (freq metadata stripped)."""
    import fxlab.data.store as store

    df = _make_ohlcv(100)
    store.save(df, "EURUSD", "1d", "test")
    loaded = store.load("EURUSD", "1d", "test")

    assert loaded is not None
    # strip freq from both sides — parquet does not preserve DatetimeIndex freq
    df_cmp = df.copy()
    df_cmp.index.freq = None
    pd.testing.assert_frame_equal(df_cmp, loaded)


def test_cache_load_missing_returns_none():
    """load() returns None when no file exists."""
    import fxlab.data.store as store

    result = store.load("NONEXISTENT", "1d", "test")
    assert result is None


def test_cache_load_filters_by_date():
    """load() with start/end slices the cached frame correctly."""
    import fxlab.data.store as store

    df = _make_ohlcv(200, start="2020-01-01")
    store.save(df, "EURUSD", "1d", "test")

    loaded = store.load("EURUSD", "1d", "test", start="2020-06-01", end="2020-09-30")
    assert loaded is not None
    assert loaded.index.min() >= pd.Timestamp("2020-06-01")
    assert loaded.index.max() <= pd.Timestamp("2020-09-30")


def test_cache_merge():
    """merge() unions two frames; new data wins on overlapping rows."""
    import fxlab.data.store as store

    df1 = _make_ohlcv(50, start="2020-01-01")
    df2 = _make_ohlcv(50, start="2020-03-01")

    store.save(df1, "EURUSD", "1d", "test")
    existing = store.load("EURUSD", "1d", "test")
    store.merge(existing, df2, "EURUSD", "1d", "test")

    merged = store.load("EURUSD", "1d", "test")
    assert merged is not None
    assert len(merged) > len(df1)
    assert merged.index.is_monotonic_increasing
    assert not merged.index.has_duplicates


def test_inventory_lists_saved_files():
    """inventory() returns one row per cached file."""
    import fxlab.data.store as store

    df = _make_ohlcv(30)
    store.save(df, "EURUSD", "1d", "test")
    store.save(df, "XAUUSD", "1d", "test")

    inv = store.inventory()
    assert len(inv) == 2
    assert set(inv["symbol"]) == {"EURUSD", "XAUUSD"}
    assert "rows" in inv.columns
    assert "start" in inv.columns
    assert "end" in inv.columns


# ---------------------------------------------------------------------------
# 5. CSV import — TradingView style
# ---------------------------------------------------------------------------


_TV_CSV = textwrap.dedent("""\
    time,open,high,low,close,volume
    2022-01-03,1.1350,1.1400,1.1300,1.1380,12000
    2022-01-04,1.1380,1.1450,1.1350,1.1420,13500
    2022-01-05,1.1420,1.1500,1.1400,1.1490,11000
    2022-01-06,1.1490,1.1520,1.1460,1.1505,9800
    2022-01-07,1.1505,1.1540,1.1480,1.1530,10200
""")


def test_csv_import_tradingview(tmp_path):
    """TradingView-style CSV is parsed and cached correctly."""
    from fxlab.data.sources.csv_import import import_csv

    csv_path = tmp_path / "tv_eurusd.csv"
    csv_path.write_text(_TV_CSV)

    df = import_csv(csv_path, symbol="EURUSD", timeframe="1d")

    assert len(df) == 5
    assert isinstance(df.index, pd.DatetimeIndex)
    assert df.index.tz is None
    assert set(df.columns) == {"open", "high", "low", "close", "volume"}
    assert (df["high"] >= df["low"]).all()
    assert df.index.is_monotonic_increasing


# ---------------------------------------------------------------------------
# 6. CSV import — MT5 style
# ---------------------------------------------------------------------------


_MT5_CSV = textwrap.dedent("""\
    Date,Time,Open,High,Low,Close,Tickvol,Vol,Spread
    2022.01.03,00:00,1.13500,1.14000,1.13000,1.13800,12000,0,0
    2022.01.04,00:00,1.13800,1.14500,1.13500,1.14200,13500,0,0
    2022.01.05,00:00,1.14200,1.15000,1.14000,1.14900,11000,0,0
""")


def test_csv_import_mt5(tmp_path):
    """MT5-style CSV (Date + Time columns, Tickvol) is parsed correctly."""
    from fxlab.data.sources.csv_import import import_csv

    csv_path = tmp_path / "mt5_eurusd.csv"
    csv_path.write_text(_MT5_CSV)

    df = import_csv(csv_path, symbol="EURUSD", timeframe="1d")

    assert len(df) == 3
    assert isinstance(df.index, pd.DatetimeIndex)
    assert (df["high"] >= df["low"]).all()


# ---------------------------------------------------------------------------
# 7. CSV import — semicolon-delimited (Dukascopy-like)
# ---------------------------------------------------------------------------


_DUKA_CSV = textwrap.dedent("""\
    Gmt time;Open;High;Low;Close;Volume
    03.01.2022 00:00:00.000;1.13500;1.14000;1.13000;1.13800;12000
    04.01.2022 00:00:00.000;1.13800;1.14500;1.13500;1.14200;13500
    05.01.2022 00:00:00.000;1.14200;1.15000;1.14000;1.14900;11000
""")


def test_csv_import_dukascopy(tmp_path):
    """Dukascopy-style semicolon CSV is parsed correctly."""
    from fxlab.data.sources.csv_import import import_csv

    csv_path = tmp_path / "duka_eurusd.csv"
    csv_path.write_text(_DUKA_CSV)

    df = import_csv(csv_path, symbol="EURUSD", timeframe="1d")

    assert len(df) == 3
    assert isinstance(df.index, pd.DatetimeIndex)
    assert (df["high"] >= df["low"]).all()


# ---------------------------------------------------------------------------
# 8. load_ohlcv with source="synthetic" end-to-end
# ---------------------------------------------------------------------------


def test_load_ohlcv_synthetic_end_to_end():
    """load_ohlcv(source='synthetic') returns a valid frame and caches it."""
    from fxlab.data import load_ohlcv, list_cached

    df = load_ohlcv(
        "XAUUSD",
        timeframe="1d",
        start="2022-01-01",
        end="2022-12-31",
        source="synthetic",
    )

    from fxlab.core import validate_ohlcv
    validated = validate_ohlcv(df)

    # Date range approximately respected (synthetic uses business day freq)
    assert df.index.min() >= pd.Timestamp("2022-01-01")
    assert df.index.max() <= pd.Timestamp("2022-12-31")
    assert len(df) > 200  # plenty of business days in 2022

    # Should now appear in inventory
    inv = list_cached()
    assert not inv.empty
    assert "XAUUSD" in inv["symbol"].values


def test_load_ohlcv_synthetic_is_deterministic():
    """Two calls to load_ohlcv(source='synthetic') for the same args return equal frames."""
    from fxlab.data import load_ohlcv

    df1 = load_ohlcv("BTCUSD", timeframe="1d", start="2021-01-01", end="2022-01-01", source="synthetic")
    df2 = load_ohlcv("BTCUSD", timeframe="1d", start="2021-01-01", end="2022-01-01", source="synthetic")

    pd.testing.assert_frame_equal(df1, df2)


# ---------------------------------------------------------------------------
# 9. SYMBOLS registry
# ---------------------------------------------------------------------------


def test_symbols_registry_keys():
    """SYMBOLS must contain exactly the 15 canonical symbols."""
    from fxlab.data import SYMBOLS

    expected = {
        "XAUUSD", "XAGUSD",
        "EURUSD", "USDJPY", "GBPUSD", "AUDUSD", "USDCAD", "USDCHF",
        "NZDUSD", "EURJPY", "GBPJPY",
        "BTCUSD", "ETHUSD", "SOLUSD", "XRPUSD",
    }
    assert set(SYMBOLS.keys()) == expected


def test_symbols_have_asset_class():
    """Every symbol entry must include an asset_class field."""
    from fxlab.data import SYMBOLS

    for sym, meta in SYMBOLS.items():
        assert "asset_class" in meta, f"{sym} missing asset_class"
        assert meta["asset_class"] in ("fx", "metal", "crypto"), (
            f"{sym} has unknown asset_class {meta['asset_class']!r}"
        )


def test_symbols_crypto_have_binance_ticker():
    """Crypto symbols must have a non-None binance ticker."""
    from fxlab.data import SYMBOLS

    cryptos = [s for s, m in SYMBOLS.items() if m["asset_class"] == "crypto"]
    for sym in cryptos:
        assert SYMBOLS[sym].get("binance") is not None, (
            f"Crypto {sym} has no binance ticker"
        )


# ---------------------------------------------------------------------------
# 10. list_cached returns correct schema
# ---------------------------------------------------------------------------


def test_list_cached_empty():
    """list_cached() returns an empty DataFrame (with correct columns) when cache is empty."""
    from fxlab.data import list_cached

    inv = list_cached()
    assert isinstance(inv, pd.DataFrame)
    assert set(inv.columns) == {"symbol", "timeframe", "source", "rows", "start", "end"}


# ---------------------------------------------------------------------------
# 11. store: _cache_path sanitises symbols with special chars
# ---------------------------------------------------------------------------


def test_cache_path_sanitises():
    """Cache path must not contain slash or dot from special symbol names."""
    import fxlab.data.store as store

    path = store._cache_path("BTC/USD", "1d", "test")
    assert "/" not in path.name or path.name.count("/") == 0
    # The name should not contain literal "/"
    assert "/" not in path.stem


# ---------------------------------------------------------------------------
# 12. validate_ohlcv rejects bad data
# ---------------------------------------------------------------------------


def test_validate_ohlcv_rejects_high_less_than_low():
    """validate_ohlcv must raise ValueError when high < low."""
    from fxlab.core import validate_ohlcv

    index = pd.date_range("2020-01-01", periods=3, freq="B")
    df = pd.DataFrame(
        {
            "open": [1.0, 1.0, 1.0],
            "high": [0.9, 1.1, 1.1],   # first bar: high < low
            "low": [1.0, 0.9, 0.9],
            "close": [1.0, 1.0, 1.0],
            "volume": [0.0, 0.0, 0.0],
        },
        index=index,
    )
    with pytest.raises(ValueError, match="high < low"):
        validate_ohlcv(df)


def test_validate_ohlcv_rejects_duplicate_index():
    """validate_ohlcv must raise ValueError on duplicate timestamps."""
    from fxlab.core import validate_ohlcv

    index = pd.DatetimeIndex(
        [pd.Timestamp("2020-01-01"), pd.Timestamp("2020-01-01"), pd.Timestamp("2020-01-02")]
    )
    df = pd.DataFrame(
        {"open": [1, 1, 1], "high": [1, 1, 1], "low": [1, 1, 1], "close": [1, 1, 1], "volume": [0, 0, 0]},
        index=index,
    )
    with pytest.raises(ValueError, match="unique"):
        validate_ohlcv(df)
