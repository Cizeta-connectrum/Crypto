# FXLab — FX / Gold / Crypto Strategy Verification Platform

## 日本語概要

FXLabは、外国為替・貴金属・暗号資産の取引戦略を検証するためのPythonバックテストプラットフォームです。
約30の戦略ファミリーから合計300の具体的な戦略を生成し、CLIとStreamlitダッシュボードの両方で操作できます。
データはParquet形式でローカルキャッシュされ、オフライン環境でも再実行可能です。
ネットワーク接続がない環境でも `--source synthetic` で決定論的な合成データを使って全機能をテストできます。

---

## What is FXLab?

FXLab is a self-contained Python platform for verifying FX, precious metals, and cryptocurrency trading strategies. It:

- Downloads OHLCV chart data from multiple free sources, caching it locally in Parquet so any backtest can be re-run offline.
- Ships **exactly 300 concrete trading strategies** generated from ~30 parameterized strategy families.
- Backtests any strategy on any cached symbol/timeframe with realistic transaction costs, optional ATR-based stop-loss / take-profit, and a full metrics suite.
- Exposes everything through a **CLI** (batch sweeps, leaderboards) and a **Streamlit dashboard** (interactive charts, comparisons).

---

## Installation

**Requirements:** Python ≥ 3.11

```bash
# Clone the repository
git clone <repo-url>
cd Crypto

# Install in editable mode (includes all dependencies)
pip install -e .

# For development (includes pytest)
pip install -e ".[dev]"
```

---

## Quickstart

### 1. Download data

```bash
# Download daily EURUSD and XAUUSD from the best available free source
python -m fxlab download --symbols EURUSD XAUUSD

# Always-available offline alternative
python -m fxlab download --symbols EURUSD XAUUSD --source synthetic

# Specify a source explicitly
python -m fxlab download --symbols BTCUSD ETHUSD --source binance --timeframe 1h
```

### 2. View cache inventory

```bash
python -m fxlab data
```

### 3. List strategies

```bash
# List all 300 strategies
python -m fxlab list

# Filter by family
python -m fxlab list --family sma_cross
```

### 4. Run a single backtest

```bash
# Basic run with default settings
python -m fxlab run --strategy sma_cross_10_50 --symbol EURUSD --source synthetic

# With costs, stop-loss, take-profit and a chart
python -m fxlab run \
    --strategy sma_cross_10_50 \
    --symbol EURUSD \
    --source synthetic \
    --cost-bps 2.0 \
    --sl-atr 2.0 \
    --tp-atr 4.0 \
    --plot results/my_backtest.html
```

### 5. Run a sweep

```bash
# Sweep all strategies across two symbols, 4 workers
python -m fxlab sweep \
    --symbols EURUSD XAUUSD \
    --strategies all \
    --source synthetic \
    --cost-bps 1.0 \
    --top 20 \
    --metric sharpe \
    --out results/sweep.csv \
    --jobs 4

# Only a specific family
python -m fxlab sweep \
    --symbols EURUSD \
    --strategies family:rsi \
    --source synthetic \
    --out results/rsi_sweep.csv
```

### 6. View leaderboard from a sweep

```bash
# Full leaderboard
python -m fxlab report --in results/sweep.csv --metric sharpe --top 20

# Filter by family or symbol
python -m fxlab report --in results/sweep.csv --family rsi
python -m fxlab report --in results/sweep.csv --symbol XAUUSD
```

### 7. Launch the Streamlit dashboard

```bash
streamlit run app/streamlit_app.py
```

Then open `http://localhost:8501` in your browser.

---

## Data Sources

FXLab supports five data sources, tried in order when `source=auto`:

| Source      | Coverage                              | Notes |
|-------------|---------------------------------------|-------|
| **stooq**   | FX, metals — daily only               | Free CSV download; `https://stooq.com` |
| **yfinance**| FX (EURUSD=X), metals (GC=F), crypto  | May be blocked on some networks |
| **binance** | Crypto only; daily and intraday       | `https://api.binance.com/api/v3/klines` |
| **csv**     | User-supplied CSV files               | Place in `data/import/`; TradingView / MT5 / Dukascopy format |
| **synthetic**| All symbols; any timeframe           | Always available; seeded by symbol name (deterministic) |

> **Note on network access:** Finance data hosts (stooq, Yahoo Finance, Binance) may be blocked in certain corporate or restricted network environments. In those cases use `--source synthetic` — the synthetic generator uses a regime-switching GBM model and always produces realistic OHLCV data.

### Synthetic data

The synthetic generator creates regime-switching geometric Brownian motion:
- **Noise regime** — zero drift, asset-class volatility.
- **Trend regime** — positive or negative drift, slightly elevated volatility.

Seeds are derived deterministically from the symbol + timeframe string, so `EURUSD_1d` always produces the same data.

---

## Strategy Library (300 strategies)

Strategies are declared in `configs/strategies.yaml` and expanded to exactly 300 instances by `fxlab.strategies.build_all()`. Families span:

| Category        | Families (examples) |
|-----------------|---------------------|
| Trend-following | SMA cross, EMA cross, Hull MA, KAMA, MACD, ADX filter, MA ribbon, Ichimoku, SuperTrend, PSAR |
| Breakout        | Donchian channel, Bollinger breakout, Keltner breakout, ATR expansion, 52-bar high |
| Mean-reversion  | RSI (periods 2–14), Bollinger fade, z-score, Williams %R, CCI, Stochastic |
| Momentum        | ROC, dual-timeframe, TRIX, Aroon |
| Pattern         | Engulfing candle, Heikin-Ashi trend |
| Seasonality     | Day-of-week, turn-of-month |
| Combined        | Trend + pullback filters |

Each strategy exposes:
- `id` — unique string (e.g. `sma_cross_10_50`)
- `family` — family name
- `params` — parameter dict

---

## Engine Semantics

The backtest engine (`fxlab.engine.run_backtest`) follows these rules:

- **Next-bar execution**: a signal generated at bar *t* is filled at the open of bar *t+1* (implemented as a one-bar shift of the position series).
- **Returns**: close-to-close simple returns net of costs: `ret_t = pos_t × (close_t / close_{t−1} − 1) − cost_t`.
- **Costs**: `cost_bps` (basis points, one-way) charged on `|pos_t − pos_{t−1}|`.
- **ATR SL/TP** (optional): entry price = close at position change; intra-bar exit when `low/high` crosses `entry ± k × ATR(period)`; after a stop the strategy stays flat until the target position changes.
- **Vectorized fast path** when neither SL nor TP is specified.

---

## Metrics Glossary

| Metric           | Description |
|------------------|-------------|
| `total_return`   | Cumulative net return over the full period |
| `cagr`           | Compound Annual Growth Rate |
| `volatility`     | Annualized standard deviation of per-bar returns |
| `sharpe`         | Sharpe ratio (risk-free rate = 0) |
| `sortino`        | Sortino ratio (downside deviation denominator) |
| `max_drawdown`   | Maximum peak-to-trough decline (negative fraction, e.g. −0.25) |
| `calmar`         | CAGR / \|max drawdown\| |
| `win_rate`       | Fraction of trades with positive return |
| `profit_factor`  | Gross profit / gross loss (`+inf` when no losing trades) |
| `n_trades`       | Total number of round-trip trades |
| `avg_trade_ret`  | Average per-trade simple return |
| `exposure`       | Average absolute position (0–1) |
| `time_in_market` | Fraction of bars with non-zero position |

---

## Running Tests

```bash
pytest tests/
```

The test suite uses `--source synthetic` throughout so no network access is required. Core test files:

- `tests/test_cli.py` — CLI smoke tests (list/run/sweep/report)
- `tests/test_engine.py` — hand-computed equity/cost/SL-TP cases
- `tests/test_strategies.py` — 300 unique ids, no-lookahead check
- `tests/test_data.py` — synthetic determinism, cache round-trip

---

## Disclaimer

FXLab is a **research and educational tool** only. Nothing in this software constitutes investment advice, a recommendation to trade, or a guarantee of future performance. Past performance of any strategy, including on synthetic data, is not indicative of future results. Always conduct your own due diligence before deploying any trading strategy with real capital.

---

## License

See `LICENSE` file (if present). All data fetched from third-party APIs is subject to their respective terms of service.
