---
title: FXLab
emoji: 📈
colorFrom: blue
colorTo: green
sdk: docker
app_port: 7860
pinned: false
---

# FXLab — FX / Gold / Crypto Strategy Verification Platform

## 日本語概要

FXLabは、外国為替・貴金属・暗号資産の取引戦略を検証するためのPythonバックテストプラットフォームです。
約30の戦略ファミリーから合計300の具体的な戦略を生成し、CLIとStreamlitダッシュボードの両方で操作できます。
データはParquet形式でローカルキャッシュされ、オフライン環境でも再実行可能です。
ネットワーク接続がない環境でも `--source synthetic` で決定論的な合成データを使って全機能をテストできます。
Streamlit Community CloudやHugging Face Spacesで無料ホスティングが可能で、Google Sheetsエクスポートにより永続的なバックテスト履歴を管理できます。

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

## Free hosting / 無料デプロイ

### Google Cloud — Cloud Shell (no install, instant trial)

Open [https://shell.cloud.google.com](https://shell.cloud.google.com) (any
Google account; Python is preinstalled) and run:

```bash
git clone <repo-url> && cd Crypto
git checkout <branch>
pip3 install --user -e .
python3 -m fxlab download --symbols XAUUSD EURUSD BTCUSD
python3 -m streamlit run app/streamlit_app.py --server.port 8080
```

Then click **Web Preview → Preview on port 8080** (top-right of the Cloud
Shell window). Note: Cloud Shell sessions are ephemeral (the VM is recycled
after ~20 minutes of inactivity; your home directory persists, the running
app does not).

### Google Cloud — Cloud Run (permanent URL)

A `Dockerfile` is included. From Cloud Shell or any machine with `gcloud`:

```bash
gcloud run deploy fxlab \
  --source . \
  --region asia-northeast1 \
  --memory 1Gi \
  --allow-unauthenticated
```

Cloud Build builds the image and prints a permanent
`https://fxlab-….run.app` URL. Notes:

* `--allow-unauthenticated` makes the URL public; omit it to keep the
  service private (access then requires `gcloud` identity tokens or IAP).
* Storage is ephemeral — the parquet cache resets when instances scale to
  zero. Re-download from the Data tab, or rely on Google Sheets export for
  permanent history.
* Google Sheets export works without a key file on Cloud Run: the module
  falls back to Application Default Credentials. Enable the Sheets + Drive
  APIs in the project and share the target spreadsheet with the service
  account the Cloud Run revision runs as (shown in the service details;
  defaults to `<project-number>-compute@developer.gserviceaccount.com`).
* Costs: Cloud Run has a generous free tier; a low-traffic instance that
  scales to zero typically stays within it.

### Streamlit Community Cloud

1. Push this repository to GitHub.
2. Go to [https://share.streamlit.io](https://share.streamlit.io) and sign in with your GitHub account.
3. Click **New app**, pick your repository and branch, and set the main file to `app/streamlit_app.py`.
4. Click **Deploy**.

**Important notes for the free tier:**

- The app **sleeps when idle** (roughly 7 days of inactivity).  Waking it up takes ~30 seconds on the first visit.
- Storage is **ephemeral** — the Parquet cache in `data/cache/` resets on every restart.  Use `--source synthetic` for zero-network demos, or re-download data after each wake-up.
- Use the **Google Sheets export** (see below) to keep a permanent history of sweep results across restarts.

To add Google service-account credentials, go to your app's **Settings → Secrets** on Streamlit Cloud and paste the JSON key fields as a TOML table:

```toml
[gcp_service_account]
type = "service_account"
project_id = "your-project-id"
private_key_id = "..."
private_key = "-----BEGIN RSA PRIVATE KEY-----\n..."
client_email = "your-sa@your-project.iam.gserviceaccount.com"
# … all other fields from the downloaded JSON key
```

### Alternative: Hugging Face Spaces

Create a new Space using the **Streamlit SDK** template, push the repo, and set `app/streamlit_app.py` as the entry point.  Hugging Face Spaces also has a free tier with similar ephemeral-storage caveats.

---

## Google Sheets export

FXLab can append sweep results to a Google Sheets spreadsheet for permanent history tracking.

### GCP setup (one-time)

1. Open [Google Cloud Console](https://console.cloud.google.com/) and create (or select) a project.
2. Enable **Google Sheets API** and **Google Drive API** for the project.
3. Navigate to **IAM & Admin → Service Accounts** and create a new service account.
4. Under the service account, go to **Keys → Add Key → Create new key (JSON)** and download the JSON key file.

### Credential configuration

**Local / CLI**

```bash
export FXLAB_GSHEET_CREDENTIALS=/path/to/service-account-key.json
```

Or pass the path inline:

```bash
python -m fxlab sweep --symbols EURUSD XAUUSD --source synthetic \
    --gsheet "FXLab Results"
# Uses FXLAB_GSHEET_CREDENTIALS (or GOOGLE_APPLICATION_CREDENTIALS) env var
```

**Streamlit Community Cloud**

Paste the full JSON key as a `[gcp_service_account]` TOML block in **App Settings → Secrets** (see above).

### Usage

```bash
# Sweep and export results in one step
python -m fxlab sweep \
    --symbols EURUSD XAUUSD BTCUSD \
    --strategies all \
    --source synthetic \
    --out results/sweep.csv \
    --gsheet "FXLab Results"

# Re-export a previously saved CSV
python -m fxlab export \
    --in results/sweep.csv \
    --gsheet "FXLab Results" \
    --timeframe 1d \
    --source synthetic
```

The **"FXLab Results"** spreadsheet will contain:

- A **"runs"** worksheet — one index row per export (run ID, timestamp, meta, best strategy / Sharpe).
- A **per-run worksheet** (named by run ID e.g. `20240610-142301`) — the full results table.

### Sharing

If the spreadsheet is **created by the service account** it lives in the SA's Drive.  For easier access it is usually better to **create the spreadsheet yourself**, then share it with the SA's `client_email` (Editor role); use the spreadsheet URL or title as the `--gsheet` value.  Either approach works — the SA can also create sheets from scratch.

---

## Disclaimer

FXLab is a **research and educational tool** only. Nothing in this software constitutes investment advice, a recommendation to trade, or a guarantee of future performance. Past performance of any strategy, including on synthetic data, is not indicative of future results. Always conduct your own due diligence before deploying any trading strategy with real capital.

---

## License

See `LICENSE` file (if present). All data fetched from third-party APIs is subject to their respective terms of service.
