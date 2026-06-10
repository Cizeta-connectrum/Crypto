# FXLab — FX/Gold/Crypto Strategy Verification Platform: Design

## Goal

A self-contained platform that:

1. Downloads OHLCV chart data for gold, major FX pairs, and crypto from free
   sources, caching it locally (Parquet) so any verification can be re-run
   offline at any time.
2. Ships exactly **300 concrete trading strategies**, generated from ~30
   parameterized strategy families.
3. Backtests any strategy on any cached symbol/timeframe with realistic costs,
   optional ATR stop-loss/take-profit, and a full metrics suite.
4. Exposes everything through a CLI (batch sweeps, leaderboards) and a
   Streamlit dashboard (interactive charts, comparisons).

## Repository layout

```
pyproject.toml
configs/strategies.yaml      # declarative definition that expands to 300 strategies
docs/DESIGN.md
fxlab/
  __init__.py
  __main__.py                # python -m fxlab → cli.main()
  core.py                    # shared types & contracts (OWNED BY ARCHITECT — do not change signatures)
  indicators.py              # technical indicators (pure functions on OHLCV df / Series)
  cli.py
  data/
    __init__.py              # load_ohlcv(), download(), list_cached(), SYMBOLS
    store.py                 # parquet cache
    sources/
      __init__.py
      synthetic.py           # regime-switching GBM generator (always available)
      stooq.py               # https://stooq.com/q/d/l/?s=<sym>&i=d  (free daily CSV)
      yfinance_src.py        # yfinance wrapper (gold futures GC=F, EURUSD=X, BTC-USD)
      binance.py             # https://api.binance.com/api/v3/klines (crypto, intraday ok)
      csv_import.py          # ingest user-provided CSV files (TradingView/MT5/Dukascopy)
  engine/
    __init__.py              # run_backtest()
    backtest.py
    metrics.py
  strategies/
    __init__.py              # build_all(), get(id), families()
    registry.py              # yaml → 300 Strategy instances
    families/                # one module per indicator group
app/streamlit_app.py
tests/
data/cache/                  # gitignored parquet cache
data/import/                 # user-dropped CSVs
results/                     # gitignored sweep outputs
```

## Core contracts (`fxlab/core.py` — frozen)

* **OHLCV DataFrame**: `pd.DatetimeIndex` (tz-naive UTC, ascending, unique),
  float columns `open, high, low, close, volume` (volume may be 0 for FX).
  `validate_ohlcv(df)` raises `ValueError` on violations; all module
  boundaries assume validated frames.
* **`Strategy`** (ABC): attributes `id: str` (unique, e.g. `sma_cross_10_50`),
  `family: str`, `params: dict`; method `generate(df) -> pd.Series` returning
  the **target position in [-1, 1]** aligned to `df.index`, computed using
  information up to and including each bar's close (the engine applies the
  next-bar shift — strategies must NOT shift themselves, and must not look
  ahead).
* **`BacktestResult`** (dataclass): `equity: pd.Series` (start 1.0),
  `returns: pd.Series`, `position: pd.Series` (effective, after shift),
  `trades: pd.DataFrame` (`entry_time, exit_time, direction, bars, ret`),
  `metrics: dict`.

## Data layer (`fxlab.data`)

```python
load_ohlcv(symbol, timeframe="1d", start=None, end=None,
           source="auto", refresh=False) -> pd.DataFrame
download(symbols, timeframe="1d", source="auto") -> dict[str, str]   # symbol → status
list_cached() -> pd.DataFrame                                        # symbol, timeframe, source, rows, start, end
```

* Canonical symbols (`fxlab.data.SYMBOLS`): `XAUUSD, XAGUSD, EURUSD, USDJPY,
  GBPUSD, AUDUSD, USDCAD, USDCHF, NZDUSD, EURJPY, GBPJPY, BTCUSD, ETHUSD,
  SOLUSD, XRPUSD` with per-source ticker mappings (e.g. XAUUSD →
  stooq `xauusd`, yfinance `GC=F` fallback `XAUUSD=X`, BTCUSD → binance
  `BTCUSDT` / yfinance `BTC-USD`).
* `source="auto"` tries, in order: cache → stooq → yfinance → binance (crypto
  only) → raises with a clear message suggesting `synthetic` or CSV import.
  `source="synthetic"` always works (seeded by symbol name → deterministic).
* Cache: `data/cache/{symbol}_{timeframe}_{source}.parquet`; `refresh=True`
  re-downloads and merges.
* Network failures must degrade gracefully (typed `DataSourceError`), never
  crash a sweep.

## Engine (`fxlab.engine`)

```python
run_backtest(df, position, *, cost_bps=1.0, sl_atr=None, tp_atr=None,
             atr_period=14) -> BacktestResult
```

* Execution model: target position from bar *t* is held over bar *t+1*
  (`pos = position.shift(1)`); returns are close-to-close log-less simple
  returns: `ret_t = pos_t * (close_t/close_{t-1} - 1) - cost_t`.
* Costs: `cost_bps` (one-way, of notional) charged on `|pos_t - pos_{t-1}|`.
* SL/TP (optional): event loop; entry price = close at position change; exit
  intra-bar when `low/high` crosses `entry ± k*ATR(atr_period)`; after a stop,
  remain flat until the strategy's target position *changes value*.
* Vectorized fast path when `sl_atr` and `tp_atr` are both None.
* Metrics (annualization factor from median index spacing; 252 for daily FX,
  365 for crypto daily, 252*24 hourly etc. — infer, allow override):
  `total_return, cagr, volatility, sharpe, sortino, max_drawdown, calmar,
  win_rate, profit_factor, n_trades, avg_trade_ret, exposure, time_in_market`.

## Strategies — 300 total

`configs/strategies.yaml` declares families; `registry.build_all()` expands
parameter grids and **must yield exactly 300 unique ids** (enforced by test).
Families (~30, ×~10 variants each) spanning: trend (SMA/EMA/Hull/KAMA cross,
MACD, ADX-filtered, MA ribbon, Ichimoku, SuperTrend, PSAR), breakout
(Donchian, Bollinger, Keltner, volatility/ATR, 52-bar high), mean-reversion
(RSI(2..14), Bollinger fade, z-score, Williams %R, CCI, stochastic),
momentum (ROC, dual-timeframe, TRIX, Aroon), pattern (engulfing, Heikin-Ashi
trend), seasonality (day-of-week, turn-of-month), and combined filters
(trend+pullback). Long/short symmetric unless the family is inherently
directional.

## CLI

```
python -m fxlab download  --symbols EURUSD XAUUSD BTCUSD [--source auto] [--timeframe 1d]
python -m fxlab data                                      # show cache inventory
python -m fxlab list      [--family trend]                # list strategies
python -m fxlab run       --strategy sma_cross_10_50 --symbol EURUSD [--cost-bps 1] [--sl-atr 2] [--tp-atr 4] [--plot out.html]
python -m fxlab sweep     --symbols EURUSD XAUUSD --strategies all [--top 20] [--out results/sweep.csv] [--jobs N]
python -m fxlab report    --in results/sweep.csv [--metric sharpe] [--top 20]
```

`sweep` uses `concurrent.futures.ProcessPoolExecutor`, writes one row per
(strategy, symbol) with all metrics, prints a leaderboard.

## Streamlit app (`app/streamlit_app.py`)

Sidebar: symbol, timeframe, source, date range, costs, SL/TP. Tabs:

1. **Data** — cache inventory, download buttons, candlestick preview.
2. **Backtest** — pick one strategy; candlestick + position overlay, equity
   vs buy&hold, drawdown, metrics table, trades table.
3. **Sweep** — multi-select families/symbols, run, sortable leaderboard,
   sharpe-vs-maxDD scatter, per-family box plot.
4. **Compare** — overlay equity curves of selected strategies.

Run: `streamlit run app/streamlit_app.py`

## Testing

* `tests/test_engine.py` — hand-computed equity/cost cases, SL/TP behaviour,
  no-lookahead (shift) check, metrics math.
* `tests/test_strategies.py` — exactly 300 unique ids; every strategy runs on
  synthetic data, output in [-1,1], aligned index, **no lookahead** (prefix
  invariance: generate on df[:n] equals generate on df restricted to first n
  rows for the last bar).
* `tests/test_data.py` — synthetic determinism, cache round-trip, csv import.
* `tests/test_cli.py` — smoke: list/run/sweep on synthetic.

Conventions: Python ≥3.11, type hints, no global state, pure functions where
possible, `ruff`-clean style, docstrings on public APIs. Dependencies limited
to: pandas, numpy, pyyaml, pyarrow, requests, yfinance, streamlit, plotly,
pytest.
