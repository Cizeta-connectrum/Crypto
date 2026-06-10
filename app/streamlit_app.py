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
from typing import Any

import pandas as pd
import streamlit as st

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
    timeframe = st.selectbox("Timeframe", ["1d", "1h"], index=0)
    source = st.selectbox(
        "Source", ["auto", "synthetic", "stooq", "yfinance", "binance", "csv"], index=0
    )

    st.subheader("Date range")
    start_date = st.date_input("Start", value=pd.Timestamp("2020-01-01").date())
    end_date = st.date_input("End", value=pd.Timestamp("2024-12-31").date())

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

tab_data, tab_backtest, tab_sweep, tab_compare = st.tabs(
    ["Data", "Backtest", "Sweep", "Compare"]
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
                    statuses = _data_mod.download([symbol], timeframe=timeframe, source=source)
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
