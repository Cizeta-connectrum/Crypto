"""FXLab Streamlit Dashboard.

Tabs:
  Data      -- cache inventory, download/refresh, candlestick preview.
  Backtest  -- single strategy; metrics, chart, trades table.
  Sweep     -- multi-family / multi-symbol sweep with progress, scatter/box.
  Compare   -- overlay equity curves of selected strategies.

Run:
    streamlit run app/streamlit_app.py
"""

from __future__ import annotations

import math
import pathlib
import re
import sys
from typing import Any

import pandas as pd
import streamlit as st

# Make the repo root importable so `fxlab` works without `pip install -e .`
# (needed on Streamlit Community Cloud / Hugging Face Spaces).
_REPO_ROOT = str(pathlib.Path(__file__).resolve().parent.parent)
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

# ---------------------------------------------------------------------------
# TradingView credential helpers
# ---------------------------------------------------------------------------

_SECRETS_PATH = pathlib.Path(__file__).parent.parent / ".streamlit" / "secrets.toml"


def _load_tv_credentials() -> tuple[str, str]:
    """Return (username, password) from saved secrets, or ("", "")."""
    # 1. Streamlit secrets (works on Streamlit Cloud and locally)
    try:
        tv = st.secrets.get("tradingview", {})
        u = tv.get("username", "")
        p = tv.get("password", "")
        if u:
            return str(u), str(p)
    except Exception:  # noqa: BLE001
        pass
    # 2. Environment variables (Hugging Face Spaces secrets, Docker, etc.)
    import os
    u = os.environ.get("FXLAB_TV_USERNAME", "")
    p = os.environ.get("FXLAB_TV_PASSWORD", "")
    if u:
        return u, p
    # 3. Read secrets.toml directly (before Streamlit has loaded it this run)
    try:
        text = _SECRETS_PATH.read_text(encoding="utf-8")
        u_m = re.search(r'^\s*username\s*=\s*"([^"]*)"', text, re.MULTILINE)
        p_m = re.search(r'^\s*password\s*=\s*"([^"]*)"', text, re.MULTILINE)
        if u_m:
            return u_m.group(1), (p_m.group(1) if p_m else "")
    except Exception:  # noqa: BLE001
        pass
    return "", ""


def _save_tv_credentials(username: str, password: str) -> None:
    """Persist TradingView credentials into .streamlit/secrets.toml."""
    _SECRETS_PATH.parent.mkdir(parents=True, exist_ok=True)
    text = _SECRETS_PATH.read_text(encoding="utf-8") if _SECRETS_PATH.exists() else ""
    # Remove existing [tradingview] block
    text = re.sub(
        r"\[tradingview\][^\[]*",
        "",
        text,
        flags=re.DOTALL,
    ).rstrip()
    # Append updated block
    block = f'\n\n[tradingview]\nusername = "{username}"\npassword = "{password}"\n'
    _SECRETS_PATH.write_text(text + block, encoding="utf-8")

# ---------------------------------------------------------------------------
# Page config (must be first Streamlit call)
# ---------------------------------------------------------------------------

st.set_page_config(
    page_title="FXLab",
    page_icon=":chart_with_upwards_trend:",
    layout="wide",
)

# ---------------------------------------------------------------------------
# Safe imports (siblings may not exist in partial environments)
# ---------------------------------------------------------------------------


def _try_import(module_path: str) -> Any:
    import importlib
    try:
        return importlib.import_module(module_path)
    except Exception as exc:  # noqa: BLE001
        return exc


_data_mod = _try_import("fxlab.data")
_engine_mod = _try_import("fxlab.engine")
_strats_mod = _try_import("fxlab.strategies")
_gsheets_mod = _try_import("fxlab.export.gsheets")

# ---------------------------------------------------------------------------
# SYMBOLS list (fallback if data module missing)
# ---------------------------------------------------------------------------

_FALLBACK_SYMBOLS = [
    "XAUUSD", "XAGUSD", "EURUSD", "USDJPY", "GBPUSD", "AUDUSD",
    "USDCAD", "USDCHF", "NZDUSD", "EURJPY", "GBPJPY",
    "BTCUSD", "ETHUSD", "SOLUSD", "XRPUSD",
]


def _get_symbols() -> list[str]:
    if isinstance(_data_mod, Exception):
        return _FALLBACK_SYMBOLS
    syms = getattr(_data_mod, "SYMBOLS", None)
    if syms is None:
        return _FALLBACK_SYMBOLS
    if isinstance(syms, dict):
        return list(syms.keys())
    return list(syms)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _safe_float(val: Any) -> float | None:
    try:
        f = float(val)
        return f
    except (TypeError, ValueError):
        return None


def _fmt_metric(val: Any) -> str:
    f = _safe_float(val)
    if f is None:
        return "N/A"
    if math.isnan(f):
        return "NaN"
    if math.isinf(f):
        return ">999" if f > 0 else "<-999"
    return f"{f:.4f}"


def _display_metrics_grid(metrics: dict) -> None:
    """Render metrics as a grid of st.metric widgets."""
    key_order = [
        ("sharpe", "Sharpe"),
        ("total_return", "Total Return"),
        ("max_drawdown", "Max Drawdown"),
        ("win_rate", "Win Rate"),
        ("n_trades", "# Trades"),
        ("profit_factor", "Profit Factor"),
        ("cagr", "CAGR"),
        ("volatility", "Volatility"),
        ("sortino", "Sortino"),
        ("calmar", "Calmar"),
        ("exposure", "Avg Exposure"),
        ("time_in_market", "Time in Market"),
    ]
    cols = st.columns(6)
    for i, (key, label) in enumerate(key_order):
        val = metrics.get(key, float("nan"))
        display = _fmt_metric(val)
        if key == "n_trades":
            f = _safe_float(val)
            display = str(int(f)) if f is not None and not math.isnan(f) else "N/A"
        cols[i % 6].metric(label, display)


@st.cache_data(show_spinner=False)
def _load_ohlcv(
    symbol: str,
    timeframe: str,
    source: str,
    start: str | None,
    end: str | None,
) -> pd.DataFrame:
    if isinstance(_data_mod, Exception):
        raise ImportError(f"fxlab.data not available: {_data_mod}")
    return _data_mod.load_ohlcv(
        symbol,
        timeframe=timeframe,
        start=start or None,
        end=end or None,
        source=source,
    )


def _candlestick_fig(df: pd.DataFrame, title: str = ""):  # noqa: ANN201
    import plotly.graph_objects as go
    fig = go.Figure(data=[go.Candlestick(
        x=df.index,
        open=df["open"],
        high=df["high"],
        low=df["low"],
        close=df["close"],
        name="OHLCV",
    )])
    fig.update_layout(
        title=title,
        xaxis_rangeslider_visible=False,
        height=400,
        template="plotly_white",
    )
    return fig


def _backtest_fig(df: pd.DataFrame, result: Any, symbol: str, strategy_id: str):  # noqa: ANN201
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    pos = result.position
    eq = result.equity
    bh = df["close"] / df["close"].iloc[0]
    running_max = eq.cummax()
    drawdown = eq / running_max - 1.0

    shapes = []
    if len(pos) > 0:
        idx = pos.index
        prev_val = float(pos.iloc[0])
        seg_start = idx[0]
        for i in range(1, len(pos)):
            val = float(pos.iloc[i])
            changed = (val != prev_val) or (i == len(pos) - 1)
            if changed:
                if prev_val != 0:
                    colour = "rgba(0,180,0,0.12)" if prev_val > 0 else "rgba(220,0,0,0.12)"
                    shapes.append(dict(
                        type="rect",
                        xref="x",
                        yref="paper",
                        x0=str(seg_start),
                        x1=str(idx[i]),
                        y0=0, y1=1,
                        fillcolor=colour,
                        line_width=0,
                        layer="below",
                    ))
                seg_start = idx[i]
                prev_val = val

    fig = make_subplots(
        rows=3, cols=1,
        shared_xaxes=True,
        row_heights=[0.5, 0.3, 0.2],
        subplot_titles=[
            f"{symbol} candlestick ({strategy_id})",
            "Equity vs Buy&Hold",
            "Drawdown",
        ],
        vertical_spacing=0.04,
    )
    fig.add_trace(go.Candlestick(
        x=df.index,
        open=df["open"],
        high=df["high"],
        low=df["low"],
        close=df["close"],
        name="OHLCV",
        showlegend=False,
    ), row=1, col=1)
    fig.add_trace(go.Scatter(
        x=eq.index, y=eq, name="Strategy", line=dict(color="steelblue"),
    ), row=2, col=1)
    fig.add_trace(go.Scatter(
        x=bh.index, y=bh, name="Buy&Hold", line=dict(color="orange", dash="dash"),
    ), row=2, col=1)
    fig.add_trace(go.Scatter(
        x=drawdown.index, y=drawdown * 100,
        name="Drawdown %", fill="tozeroy", line=dict(color="crimson"),
    ), row=3, col=1)

    fig.update_layout(
        shapes=shapes,
        height=900,
        xaxis_rangeslider_visible=False,
        template="plotly_white",
    )
    return fig


# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------

st.title("FXLab — Strategy Verification Platform")

symbols = _get_symbols()

with st.sidebar:
    st.header("Settings")
    symbol = st.selectbox("Symbol", symbols, index=symbols.index("EURUSD") if "EURUSD" in symbols else 0)
    timeframe = st.selectbox("Timeframe", ["1d", "4h", "1h", "15m", "5m", "1w"], index=0)
    source = st.selectbox(
        "Source", ["auto", "synthetic", "stooq", "yfinance", "binance", "csv", "tradingview"], index=0
    )

    if source == "tradingview":
        st.subheader("TradingView Credentials")
        _saved_u, _saved_p = _load_tv_credentials()
        tv_username = st.text_input("TV Username", value=_saved_u, key="tv_username")
        tv_password = st.text_input("TV Password", value=_saved_p, type="password", key="tv_password")
        if st.button("Save credentials", key="btn_save_tv"):
            try:
                _save_tv_credentials(tv_username, tv_password)
                st.success("Saved!")
            except Exception as _e:  # noqa: BLE001
                st.error(f"Could not save: {_e}")
    else:
        tv_username = ""
        tv_password = ""

    st.subheader("Date range")
    start_date = st.date_input("Start", value=pd.Timestamp("2020-01-01").date())
    end_date = st.date_input("End", value=pd.Timestamp.today().date())

    st.subheader("Costs & Risk")
    cost_bps = st.number_input("Cost (bps)", min_value=0.0, value=1.0, step=0.5)
    sl_atr = st.number_input("SL ATR mult (0=off)", min_value=0.0, value=0.0, step=0.5)
    tp_atr = st.number_input("TP ATR mult (0=off)", min_value=0.0, value=0.0, step=0.5)

start_str = str(start_date) if start_date else None
end_str = str(end_date) if end_date else None
sl_atr_val = float(sl_atr) if sl_atr > 0 else None
tp_atr_val = float(tp_atr) if tp_atr > 0 else None

# ---------------------------------------------------------------------------
# Tabs
# ---------------------------------------------------------------------------

tab_ranking, tab_data, tab_backtest, tab_sweep, tab_compare = st.tabs(
    ["🏆 ランキング", "Data", "Backtest", "Sweep", "Compare"]
)

# ===================================================================
# Tab: Ranking (one-click: run all 300 strategies, rank the best)
# ===================================================================

_RANKING_HISTORY_PATH = pathlib.Path(_REPO_ROOT) / "results" / "ranking_history.csv"
_RANKING_HISTORY_COLS = [
    "run_at", "symbol", "timeframe", "source", "start", "end",
    "cost_bps", "sl_atr", "tp_atr",
    "strategy", "family", "sharpe", "total_return", "cagr", "max_drawdown",
    "win_rate", "profit_factor", "n_trades", "sortino", "calmar",
]
_RANKING_METRIC_COLS = [
    "sharpe", "total_return", "cagr", "max_drawdown", "win_rate",
    "profit_factor", "n_trades", "sortino", "calmar",
]


def _ranking_sort_key(v: Any) -> float:
    """Numeric sort key: NaN/unparseable to -inf so they rank last."""
    f = _safe_float(v)
    if f is None or math.isnan(f):
        return float("-inf")
    return f


def _format_ranking_df(df: pd.DataFrame) -> pd.DataFrame:
    """Return a display copy: percents with 2 decimals, ratios with 3, inf as ∞."""
    disp = df.copy()

    def _pct(v: Any) -> str:
        f = _safe_float(v)
        if f is None or math.isnan(f):
            return "N/A"
        if math.isinf(f):
            return "∞" if f > 0 else "-∞"
        return f"{f * 100:.2f}%"

    def _ratio(v: Any) -> str:
        f = _safe_float(v)
        if f is None or math.isnan(f):
            return "N/A"
        if math.isinf(f):
            return "∞" if f > 0 else "-∞"
        return f"{f:.3f}"

    def _pf(v: Any) -> str:
        f = _safe_float(v)
        if f is None or math.isnan(f):
            return "N/A"
        if math.isinf(f):
            return "∞"
        return f"{f:.2f}"

    def _int(v: Any) -> str:
        f = _safe_float(v)
        if f is None or math.isnan(f):
            return "N/A"
        return str(int(f))

    for col in ("total_return", "max_drawdown", "win_rate", "cagr"):
        if col in disp.columns:
            disp[col] = disp[col].apply(_pct)
    for col in ("sharpe", "sortino", "calmar"):
        if col in disp.columns:
            disp[col] = disp[col].apply(_ratio)
    if "profit_factor" in disp.columns:
        disp["profit_factor"] = disp["profit_factor"].apply(_pf)
    if "n_trades" in disp.columns:
        disp["n_trades"] = disp["n_trades"].apply(_int)
    return disp


def _prepare_ranking_history_rows(top_df: pd.DataFrame, meta: dict) -> pd.DataFrame:
    """Attach run metadata columns to the top rows of a ranking run."""
    rows = top_df.copy()
    for key, val in meta.items():
        rows[key] = val
    return rows[[c for c in _RANKING_HISTORY_COLS if c in rows.columns]]


def _append_ranking_history(rows: pd.DataFrame) -> None:
    """Append prepared ranking rows to results/ranking_history.csv (bounded)."""
    _RANKING_HISTORY_PATH.parent.mkdir(parents=True, exist_ok=True)
    if _RANKING_HISTORY_PATH.exists():
        try:
            hist = pd.read_csv(_RANKING_HISTORY_PATH)
            hist = pd.concat([hist, rows], ignore_index=True)
        except Exception:  # noqa: BLE001
            hist = rows
    else:
        hist = rows
    if len(hist) > 5000:
        hist = hist.tail(5000).reset_index(drop=True)
    hist.to_csv(_RANKING_HISTORY_PATH, index=False)


def _gsheets_credentials() -> dict | None:
    """Explicit credentials from st.secrets[gcp_service_account], if present."""
    try:
        if "gcp_service_account" in st.secrets:
            return dict(st.secrets["gcp_service_account"])
    except Exception:  # noqa: BLE001
        pass  # st.secrets not available; fall back to env-based resolution
    return None


def _gsheets_configured() -> bool:
    """True if Google Sheets credentials are configured (secrets or env)."""
    if _gsheets_credentials() is not None:
        return True
    import os
    return bool(
        os.environ.get("FXLAB_GSHEET_CREDENTIALS")
        or os.environ.get("GOOGLE_APPLICATION_CREDENTIALS")
    )


@st.cache_data(ttl=60, show_spinner=False)
def _read_ranking_history_gsheets() -> pd.DataFrame:
    """Read the all-time ranking history from Google Sheets (cached 60s)."""
    if isinstance(_gsheets_mod, Exception):
        raise RuntimeError(f"fxlab.export.gsheets is not available: {_gsheets_mod}")
    return _gsheets_mod.read_ranking_history(credentials=_gsheets_credentials())


def _append_ranking_history_gsheets(rows: pd.DataFrame) -> None:
    """Append prepared ranking rows to the Google Sheets history worksheet."""
    if isinstance(_gsheets_mod, Exception):
        raise RuntimeError(f"fxlab.export.gsheets is not available: {_gsheets_mod}")
    _gsheets_mod.append_ranking_history(rows, credentials=_gsheets_credentials())


with tab_ranking:
    st.header("🏆 戦略ランキング")
    st.markdown("通貨を選んでボタンを押すだけ。全300戦略を一括検証して、成績の良い順に表示します。")

    if isinstance(_strats_mod, Exception) or isinstance(_engine_mod, Exception) or isinstance(_data_mod, Exception):
        missing = []
        if isinstance(_strats_mod, Exception):
            missing.append("fxlab.strategies")
        if isinstance(_engine_mod, Exception):
            missing.append("fxlab.engine")
        if isinstance(_data_mod, Exception):
            missing.append("fxlab.data")
        st.error(f"必要なモジュールが読み込めません: {', '.join(missing)}")
    else:
        _risk_note = ""
        if sl_atr_val:
            _risk_note += f" | SL: ATR×{sl_atr_val:g}"
        if tp_atr_val:
            _risk_note += f" | TP: ATR×{tp_atr_val:g}"
        st.caption(
            f"対象: {symbol} / {timeframe} / {source} | "
            f"期間: {start_str or '最初'}〜{end_str or '最新'} | "
            f"コスト: {cost_bps}bps{_risk_note}"
            "（サイドバーで変更できます）"
        )

        ranking_metric = st.selectbox(
            "ランキング指標",
            ["sharpe", "total_return", "profit_factor", "win_rate", "calmar", "max_drawdown", "sortino"],
            index=0,
            key="ranking_metric",
            help="sharpe = リスクあたりの収益（おすすめ）",
        )

        run_ranking = st.button(
            "🚀 全300戦略を実行",
            key="btn_run_ranking",
            type="primary",
            use_container_width=True,
        )

        if run_ranking:
            import time as _time

            try:
                all_strategies_rank = _strats_mod.build_all()
            except Exception as exc:  # noqa: BLE001
                all_strategies_rank = []
                st.error(f"戦略リストを構築できませんでした: {exc}")

            df_rank = None
            if all_strategies_rank:
                try:
                    df_rank = _load_ohlcv(symbol, timeframe, source, start_str, end_str)
                except Exception as exc:  # noqa: BLE001
                    st.error(f"データを取得できませんでした: {exc}")

            if df_rank is not None:
                _t0 = _time.time()
                total_rank = len(all_strategies_rank)
                rank_progress = st.progress(0.0, text="検証を開始しています…")
                rank_rows: list[dict] = []
                rank_errors = 0

                for i, s in enumerate(all_strategies_rank, start=1):
                    try:
                        pos = s.generate(df_rank)
                        bt_kw_r: dict[str, Any] = {"cost_bps": float(cost_bps)}
                        if sl_atr_val:
                            bt_kw_r["sl_atr"] = sl_atr_val
                        if tp_atr_val:
                            bt_kw_r["tp_atr"] = tp_atr_val
                        res = _engine_mod.run_backtest(df_rank, pos, **bt_kw_r)
                        row_r: dict[str, Any] = {"strategy": s.id, "family": s.family}
                        for mkey in _RANKING_METRIC_COLS:
                            row_r[mkey] = res.metrics.get(mkey, float("nan"))
                        rank_rows.append(row_r)
                    except Exception:  # noqa: BLE001
                        rank_errors += 1
                    rank_progress.progress(
                        min(i / total_rank, 1.0),
                        text=f"{i}/{total_rank} — {s.id}",
                    )

                rank_progress.progress(1.0, text="完了！")
                elapsed = _time.time() - _t0

                if rank_rows:
                    rank_df = pd.DataFrame(rank_rows)
                    st.session_state["ranking_results"] = rank_df
                    st.session_state["ranking_meta"] = {
                        "symbol": symbol,
                        "timeframe": timeframe,
                        "source": source,
                        "start": start_str or "",
                        "end": end_str or "",
                        "cost_bps": float(cost_bps),
                        "sl_atr": sl_atr_val or "",
                        "tp_atr": tp_atr_val or "",
                    }

                    # Persist the top 50 of this run to the all-time history.
                    sorted_for_hist = rank_df.iloc[
                        rank_df[ranking_metric].apply(_ranking_sort_key).argsort()[::-1].values
                    ]
                    hist_meta = dict(st.session_state["ranking_meta"])
                    hist_meta["run_at"] = pd.Timestamp.now().isoformat(timespec="seconds")
                    hist_rows = _prepare_ranking_history_rows(
                        sorted_for_hist.head(50), hist_meta
                    )
                    try:
                        _append_ranking_history(hist_rows)
                    except Exception as exc:  # noqa: BLE001
                        st.warning(f"履歴の保存に失敗しました: {exc}")

                    # Also sync to Google Sheets (permanent storage) when
                    # credentials are configured; never block the rest.
                    if _gsheets_configured():
                        try:
                            _append_ranking_history_gsheets(hist_rows)
                            _read_ranking_history_gsheets.clear()
                            st.caption("✅ Google Sheets に保存しました")
                        except Exception as exc:  # noqa: BLE001
                            st.warning(f"Google Sheets への保存に失敗しました: {exc}")

                    best_row = sorted_for_hist.iloc[0]
                    st.success(
                        f"完了！ {len(rank_rows)}戦略を{elapsed:.1f}秒で検証しました。"
                        f" 1位: **{best_row['strategy']}**"
                        f"（{ranking_metric} = {_fmt_metric(best_row[ranking_metric])}）"
                    )
                    if rank_errors:
                        st.warning(f"{rank_errors}件の戦略でエラーが発生し、スキップしました。")
                else:
                    st.error("有効な結果が得られませんでした。データや期間の設定を確認してください。")

        # --- Current-run ranking (re-rendered from session state) ---
        if "ranking_results" in st.session_state:
            rank_df = st.session_state["ranking_results"]
            meta_r = st.session_state.get("ranking_meta", {})
            if meta_r:
                st.caption(
                    f"検証結果: {meta_r.get('symbol', '')} / {meta_r.get('timeframe', '')}"
                    f" / {meta_r.get('source', '')} | "
                    f"期間: {meta_r.get('start') or '最初'}〜{meta_r.get('end') or '最新'}"
                )

            sort_col_r = ranking_metric if ranking_metric in rank_df.columns else "sharpe"
            sorted_rank = rank_df.iloc[
                rank_df[sort_col_r].apply(_ranking_sort_key).argsort()[::-1].values
            ].reset_index(drop=True)
            sorted_rank.insert(0, "順位", range(1, len(sorted_rank) + 1))

            st.subheader(f"📊 今回のランキング（{sort_col_r}順）")
            st.dataframe(
                _format_ranking_df(sorted_rank),
                use_container_width=True,
                hide_index=True,
            )

        # --- All-time leaderboard ---
        st.subheader("🏛 歴代ランキング（通算ベスト50）")

        hist_df: pd.DataFrame | None = None
        hist_source: str = ""

        # 1. Google Sheets (when configured)
        if _gsheets_configured():
            try:
                hist_df = _read_ranking_history_gsheets()
                hist_source = "Google Sheets"
            except Exception as exc:  # noqa: BLE001
                hist_df = None
                st.warning(f"Google Sheets から履歴を読み込めませんでした: {exc}")

        # 2. Fallback: local CSV (Sheets not configured, failed, or empty)
        if (hist_df is None or hist_df.empty) and _RANKING_HISTORY_PATH.exists():
            try:
                hist_df = pd.read_csv(_RANKING_HISTORY_PATH)
                hist_source = (
                    "ローカルCSV（フォールバック）"
                    if _gsheets_configured()
                    else "ローカルCSV（Sheets未設定）"
                )
            except Exception as exc:  # noqa: BLE001
                hist_df = None
                st.warning(f"履歴ファイルを読み込めませんでした: {exc}")

        if hist_df is not None and not hist_df.empty:
            st.caption(f"データソース: {hist_source}")
            hist_metric = ranking_metric if ranking_metric in hist_df.columns else "sharpe"
            hist_df = hist_df.copy()
            hist_df["_sort"] = hist_df[hist_metric].apply(_ranking_sort_key)
            # Best record per (strategy, symbol, timeframe)
            dedup_keys = [k for k in ("strategy", "symbol", "timeframe") if k in hist_df.columns]
            best_hist = (
                hist_df.sort_values("_sort", ascending=False)
                .drop_duplicates(subset=dedup_keys, keep="first")
                .head(50)
                .drop(columns=["_sort"])
                .reset_index(drop=True)
            )
            best_hist.insert(0, "順位", range(1, len(best_hist) + 1))
            show_cols = ["順位", "strategy", "family", "symbol", "timeframe", "run_at"] + [
                c for c in _RANKING_METRIC_COLS if c in best_hist.columns
            ]
            show_cols = [c for c in show_cols if c in best_hist.columns]
            st.dataframe(
                _format_ranking_df(best_hist[show_cols]),
                use_container_width=True,
                hide_index=True,
            )
        else:
            st.info("まだ履歴がありません。上のボタンで検証を実行すると記録されます。")

        if _gsheets_configured():
            st.caption(
                "※ 履歴は Google Sheets に恒久的に保存されます"
                "（実行のたびにトップ50が自動追記されます）。"
            )
        else:
            st.caption(
                "※ Hugging Face Spaces ではストレージが一時的なため、Space の再起動で履歴はリセットされます。"
                "恒久的に記録したい場合は Google Sheets のサービスアカウントを設定するか、"
                "Sweep タブの Google Sheets エクスポートをご利用ください。"
            )

# ===================================================================
# Tab: Data
# ===================================================================

with tab_data:
    st.header("Data Cache Inventory")

    if isinstance(_data_mod, Exception):
        st.error(f"fxlab.data not available: {_data_mod}")
    else:
        col_refresh, _ = st.columns([1, 4])
        with col_refresh:
            refresh = st.button("Download / Refresh", key="btn_download")

        if refresh:
            with st.spinner(f"Downloading {symbol} ({source})…"):
                try:
                    import os as _os
                    _tv_env_set = False
                    if source == "tradingview" and tv_username and tv_password:
                        _os.environ["FXLAB_TV_USERNAME"] = tv_username
                        _os.environ["FXLAB_TV_PASSWORD"] = tv_password
                        _tv_env_set = True
                    statuses = _data_mod.download([symbol], timeframe=timeframe, source=source)
                    if _tv_env_set:
                        _os.environ.pop("FXLAB_TV_USERNAME", None)
                        _os.environ.pop("FXLAB_TV_PASSWORD", None)
                        # Auto-persist working credentials so they are
                        # pre-filled on the next app start.
                        try:
                            _save_tv_credentials(tv_username, tv_password)
                        except Exception:  # noqa: BLE001
                            pass
                    for sym, status in statuses.items():
                        if "ok" in status.lower() or "success" in status.lower():
                            st.success(f"{sym}: {status}")
                        else:
                            st.warning(f"{sym}: {status}")
                    _load_ohlcv.clear()
                except Exception as exc:  # noqa: BLE001
                    st.error(f"Download failed: {exc}")

        try:
            inv = _data_mod.list_cached()
            if inv.empty:
                st.info("Cache is empty. Use 'Download / Refresh' to fetch data.")
            else:
                st.dataframe(inv, use_container_width=True)
        except Exception as exc:  # noqa: BLE001
            st.error(f"Could not read cache inventory: {exc}")

    st.subheader("Import CSV (TradingView / MT4 / MT5 / Dukascopy)")
    if not isinstance(_data_mod, Exception):
        up_col1, up_col2 = st.columns([1, 1])
        with up_col1:
            upload_symbol = st.text_input(
                "Symbol for imported data", value=symbol, key="csv_symbol"
            ).strip().upper()
        with up_col2:
            upload_tf = st.selectbox(
                "Timeframe of the CSV", ["1d", "1h", "4h", "15m"], key="csv_tf"
            )
        uploaded = st.file_uploader(
            "Drop an exported chart CSV here (delimiter and column names are auto-detected)",
            type=["csv", "txt"],
            key="csv_upload",
        )
        if uploaded is not None and st.button("Import into cache", key="btn_import_csv"):
            import tempfile
            from fxlab.data.sources import csv_import as _csv_import

            try:
                with tempfile.NamedTemporaryFile(
                    suffix=".csv", delete=False
                ) as tmp:
                    tmp.write(uploaded.getvalue())
                    tmp_path = tmp.name
                imported = _csv_import.import_csv(
                    tmp_path, symbol=upload_symbol, timeframe=upload_tf
                )
                _load_ohlcv.clear()
                st.success(
                    f"Imported {len(imported):,} bars for {upload_symbol} "
                    f"({upload_tf}): {imported.index[0].date()} → "
                    f"{imported.index[-1].date()}. "
                    f"Select source 'csv' in the sidebar to use it."
                )
            except Exception as exc:  # noqa: BLE001
                st.error(f"CSV import failed: {exc}")

    st.subheader(f"Candlestick preview — {symbol}")
    try:
        df_preview = _load_ohlcv(symbol, timeframe, source, start_str, end_str)
        st.plotly_chart(_candlestick_fig(df_preview, title=f"{symbol} {timeframe}"),
                        use_container_width=True)
        st.caption(f"{len(df_preview):,} bars  |  {df_preview.index[0].date()} → {df_preview.index[-1].date()}")
    except Exception as exc:  # noqa: BLE001
        st.error(f"Could not load data for {symbol}: {exc}")

# ===================================================================
# Tab: Backtest
# ===================================================================

with tab_backtest:
    st.header("Single Strategy Backtest")

    if isinstance(_strats_mod, Exception):
        st.error(f"fxlab.strategies not available: {_strats_mod}")
    elif isinstance(_engine_mod, Exception):
        st.error(f"fxlab.engine not available: {_engine_mod}")
    else:
        # Build strategy list grouped by family
        try:
            all_strategies = _strats_mod.build_all()
        except Exception as exc:  # noqa: BLE001
            all_strategies = []
            st.error(f"Could not build strategies: {exc}")

        if all_strategies:
            # Group by family for selectbox display
            family_map: dict[str, list] = {}
            for s in all_strategies:
                family_map.setdefault(s.family, []).append(s)

            # Flat list with family labels
            options: list[str] = []
            for fam in sorted(family_map):
                for s in family_map[fam]:
                    options.append(s.id)

            selected_strategy_id = st.selectbox(
                "Strategy",
                options,
                format_func=lambda sid: sid,
            )

            run_bt = st.button("Run Backtest", key="btn_run_bt")

            if run_bt:
                with st.spinner("Running backtest…"):
                    try:
                        df_bt = _load_ohlcv(symbol, timeframe, source, start_str, end_str)
                        strategy_obj = _strats_mod.get(selected_strategy_id)
                        position = strategy_obj.generate(df_bt)

                        bt_kwargs: dict[str, Any] = {"cost_bps": float(cost_bps)}
                        if sl_atr_val:
                            bt_kwargs["sl_atr"] = sl_atr_val
                        if tp_atr_val:
                            bt_kwargs["tp_atr"] = tp_atr_val

                        result = _engine_mod.run_backtest(df_bt, position, **bt_kwargs)

                        st.subheader("Metrics")
                        _display_metrics_grid(result.metrics)

                        st.subheader("Chart")
                        fig_bt = _backtest_fig(df_bt, result, symbol, selected_strategy_id)
                        st.plotly_chart(fig_bt, use_container_width=True)

                        st.subheader("Trades")
                        trades_df = result.trades
                        if trades_df is not None and len(trades_df) > 0:
                            st.dataframe(trades_df, use_container_width=True)
                        else:
                            st.info("No trades in this period.")

                    except Exception as exc:  # noqa: BLE001
                        st.error(f"Backtest failed: {exc}")

# ===================================================================
# Tab: Sweep
# ===================================================================

with tab_sweep:
    st.header("Strategy Sweep")

    if isinstance(_strats_mod, Exception) or isinstance(_engine_mod, Exception) or isinstance(_data_mod, Exception):
        missing = []
        if isinstance(_strats_mod, Exception):
            missing.append("fxlab.strategies")
        if isinstance(_engine_mod, Exception):
            missing.append("fxlab.engine")
        if isinstance(_data_mod, Exception):
            missing.append("fxlab.data")
        st.error(f"Required modules not available: {', '.join(missing)}")
    else:
        try:
            all_strategies_sweep = _strats_mod.build_all()
        except Exception as exc:  # noqa: BLE001
            all_strategies_sweep = []
            st.error(f"Could not build strategies: {exc}")

        if all_strategies_sweep:
            all_families = sorted({s.family for s in all_strategies_sweep})
            sweep_symbols = st.multiselect("Symbols", _get_symbols(), default=[symbol])
            sweep_families = st.multiselect("Strategy families", all_families, default=all_families[:3] if len(all_families) >= 3 else all_families)
            sweep_metric = st.selectbox("Ranking metric", ["sharpe", "total_return", "cagr", "sortino", "calmar"], key="sweep_metric")

            run_sweep = st.button("Run Sweep", key="btn_run_sweep")

            if run_sweep:
                if not sweep_symbols:
                    st.warning("Select at least one symbol.")
                elif not sweep_families:
                    st.warning("Select at least one strategy family.")
                else:
                    sel_strategies = [s for s in all_strategies_sweep if s.family in sweep_families]
                    total_jobs = len(sel_strategies) * len(sweep_symbols)
                    progress_bar = st.progress(0.0, text="Starting sweep…")
                    sweep_results: list[dict] = []
                    job_count = 0

                    for sym in sweep_symbols:
                        try:
                            df_sw = _load_ohlcv(sym, timeframe, source, start_str, end_str)
                        except Exception as exc:  # noqa: BLE001
                            st.warning(f"Could not load {sym}: {exc}")
                            job_count += len(sel_strategies)
                            progress_bar.progress(min(job_count / total_jobs, 1.0))
                            continue

                        for s in sel_strategies:
                            try:
                                pos = s.generate(df_sw)
                                bt_kw: dict[str, Any] = {"cost_bps": float(cost_bps)}
                                if sl_atr_val:
                                    bt_kw["sl_atr"] = sl_atr_val
                                if tp_atr_val:
                                    bt_kw["tp_atr"] = tp_atr_val
                                res = _engine_mod.run_backtest(df_sw, pos, **bt_kw)
                                row: dict[str, Any] = {
                                    "strategy_id": s.id,
                                    "family": s.family,
                                    "symbol": sym,
                                }
                                row.update(res.metrics)
                                sweep_results.append(row)
                            except Exception as exc:  # noqa: BLE001
                                sweep_results.append({
                                    "strategy_id": s.id,
                                    "family": s.family,
                                    "symbol": sym,
                                    "error": str(exc),
                                })
                            job_count += 1
                            progress_bar.progress(
                                min(job_count / total_jobs, 1.0),
                                text=f"{job_count}/{total_jobs} — {s.id} on {sym}",
                            )

                    progress_bar.progress(1.0, text="Done!")

                    if sweep_results:
                        results_df = pd.DataFrame(sweep_results)
                        st.session_state["sweep_results"] = results_df

            # Display results if available
            if "sweep_results" in st.session_state:
                results_df = st.session_state["sweep_results"]

                st.subheader("Results")

                # Sort by chosen metric
                def _sort_val(v: Any) -> float:
                    try:
                        f = float(v)
                        return f if math.isfinite(f) else float("-inf")
                    except (TypeError, ValueError):
                        return float("-inf")

                sort_col = sweep_metric if sweep_metric in results_df.columns else results_df.columns[0]
                if sort_col in results_df.columns:
                    sorted_df = results_df.iloc[
                        results_df[sort_col].apply(_sort_val).argsort()[::-1].values
                    ]
                else:
                    sorted_df = results_df

                st.dataframe(sorted_df, use_container_width=True)

                # Download button
                csv_bytes = sorted_df.to_csv(index=False).encode("utf-8")
                st.download_button(
                    "Download CSV",
                    data=csv_bytes,
                    file_name="sweep_results.csv",
                    mime="text/csv",
                )

                # Scatter: sharpe vs max_drawdown
                if "sharpe" in results_df.columns and "max_drawdown" in results_df.columns:
                    try:
                        import plotly.express as px
                        scatter_df = results_df.copy()
                        scatter_df["sharpe_num"] = pd.to_numeric(scatter_df["sharpe"], errors="coerce")
                        scatter_df["mdd_num"] = pd.to_numeric(scatter_df["max_drawdown"], errors="coerce")
                        scatter_df = scatter_df.dropna(subset=["sharpe_num", "mdd_num"])
                        scatter_df = scatter_df[scatter_df["sharpe_num"].apply(math.isfinite) & scatter_df["mdd_num"].apply(math.isfinite)]

                        fig_scatter = px.scatter(
                            scatter_df,
                            x="mdd_num",
                            y="sharpe_num",
                            color="family",
                            hover_data=["strategy_id", "symbol"],
                            labels={"mdd_num": "Max Drawdown", "sharpe_num": "Sharpe"},
                            title="Sharpe vs Max Drawdown",
                        )
                        st.plotly_chart(fig_scatter, use_container_width=True)
                    except Exception as exc:  # noqa: BLE001
                        st.warning(f"Could not render scatter plot: {exc}")

                # Box plot: sharpe per family
                if "sharpe" in results_df.columns and "family" in results_df.columns:
                    try:
                        import plotly.express as px
                        box_df = results_df.copy()
                        box_df["sharpe_num"] = pd.to_numeric(box_df["sharpe"], errors="coerce")
                        box_df = box_df.dropna(subset=["sharpe_num"])
                        box_df = box_df[box_df["sharpe_num"].apply(math.isfinite)]

                        fig_box = px.box(
                            box_df,
                            x="family",
                            y="sharpe_num",
                            title="Sharpe Distribution by Family",
                            labels={"sharpe_num": "Sharpe", "family": "Family"},
                        )
                        st.plotly_chart(fig_box, use_container_width=True)
                    except Exception as exc:  # noqa: BLE001
                        st.warning(f"Could not render box plot: {exc}")

                # Google Sheets export
                st.subheader("Export to Google Sheets")
                gsheet_target = st.text_input(
                    "Google Sheets (URL / ID / title)",
                    value="FXLab Results",
                    key="sweep_gsheet_target",
                )
                if st.button("Export to Google Sheets", key="btn_gsheet_export"):
                    _creds: dict | None = None
                    try:
                        if "gcp_service_account" in st.secrets:
                            _creds = dict(st.secrets["gcp_service_account"])
                    except Exception:  # noqa: BLE001
                        pass  # st.secrets not available; fall back to env

                    try:
                        from fxlab.export.gsheets import export_sweep as _export_sweep  # type: ignore[import]
                        _run_meta = {
                            "symbols": " ".join(sweep_symbols) if sweep_symbols else "",
                            "timeframe": timeframe,
                            "source": source,
                            "start": start_str or "",
                            "end": end_str or "",
                            "cost_bps": float(cost_bps),
                            "sl_atr": sl_atr_val or "",
                            "tp_atr": tp_atr_val or "",
                        }
                        with st.spinner("Exporting to Google Sheets…"):
                            _sheet_url = _export_sweep(
                                results_df,
                                _run_meta,
                                gsheet_target,
                                credentials=_creds,
                            )
                        st.success(
                            f"Exported successfully. [Open spreadsheet]({_sheet_url})"
                        )
                    except Exception as _exc:  # noqa: BLE001
                        st.error(f"Google Sheets export failed: {_exc}")

# ===================================================================
# Tab: Compare
# ===================================================================

with tab_compare:
    st.header("Strategy Comparison")

    if isinstance(_strats_mod, Exception) or isinstance(_engine_mod, Exception) or isinstance(_data_mod, Exception):
        missing = []
        if isinstance(_strats_mod, Exception):
            missing.append("fxlab.strategies")
        if isinstance(_engine_mod, Exception):
            missing.append("fxlab.engine")
        if isinstance(_data_mod, Exception):
            missing.append("fxlab.data")
        st.error(f"Required modules not available: {', '.join(missing)}")
    else:
        try:
            all_strategies_cmp = _strats_mod.build_all()
        except Exception as exc:  # noqa: BLE001
            all_strategies_cmp = []
            st.error(f"Could not build strategies: {exc}")

        if all_strategies_cmp:
            all_ids = [s.id for s in all_strategies_cmp]
            compare_ids = st.multiselect(
                "Select strategies to compare",
                all_ids,
                default=all_ids[:3] if len(all_ids) >= 3 else all_ids,
                key="compare_ids",
            )

            run_compare = st.button("Run Comparison", key="btn_compare")

            if run_compare and compare_ids:
                with st.spinner("Running comparisons…"):
                    import plotly.graph_objects as go

                    strat_map = {s.id: s for s in all_strategies_cmp}
                    fig_cmp = go.Figure()
                    metrics_rows: list[dict] = []

                    try:
                        df_cmp = _load_ohlcv(symbol, timeframe, source, start_str, end_str)
                        bh = df_cmp["close"] / df_cmp["close"].iloc[0]
                        fig_cmp.add_trace(go.Scatter(
                            x=bh.index, y=bh, name="Buy&Hold",
                            line=dict(dash="dash", color="gray"),
                        ))
                    except Exception as exc:  # noqa: BLE001
                        st.error(f"Could not load data: {exc}")
                        df_cmp = None

                    if df_cmp is not None:
                        for sid in compare_ids:
                            strat = strat_map.get(sid)
                            if strat is None:
                                continue
                            try:
                                pos = strat.generate(df_cmp)
                                bt_kw2: dict[str, Any] = {"cost_bps": float(cost_bps)}
                                if sl_atr_val:
                                    bt_kw2["sl_atr"] = sl_atr_val
                                if tp_atr_val:
                                    bt_kw2["tp_atr"] = tp_atr_val
                                res = _engine_mod.run_backtest(df_cmp, pos, **bt_kw2)
                                fig_cmp.add_trace(go.Scatter(
                                    x=res.equity.index,
                                    y=res.equity,
                                    name=sid,
                                ))
                                row_m: dict[str, Any] = {"strategy_id": sid}
                                row_m.update(res.metrics)
                                metrics_rows.append(row_m)
                            except Exception as exc:  # noqa: BLE001
                                st.warning(f"{sid} failed: {exc}")

                        fig_cmp.update_layout(
                            title=f"Equity Curves — {symbol} [{timeframe}]",
                            yaxis_title="Equity (start=1.0)",
                            height=500,
                            template="plotly_white",
                        )
                        st.plotly_chart(fig_cmp, use_container_width=True)

                        if metrics_rows:
                            st.subheader("Metrics Comparison")
                            metrics_cmp_df = pd.DataFrame(metrics_rows).set_index("strategy_id")
                            # Replace inf with string for display
                            for col in metrics_cmp_df.select_dtypes(include="number").columns:
                                metrics_cmp_df[col] = metrics_cmp_df[col].apply(
                                    lambda v: ">999" if isinstance(v, float) and math.isinf(v) and v > 0
                                    else ("<-999" if isinstance(v, float) and math.isinf(v) and v < 0
                                          else v)
                                )
                            st.dataframe(metrics_cmp_df, use_container_width=True)
