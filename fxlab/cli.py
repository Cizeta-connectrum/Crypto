"""FXLab command-line interface.

Subcommands
-----------
download  -- Download OHLCV data for one or more symbols.
data      -- Show the local cache inventory.
list      -- List strategy ids (with family and params).
run       -- Run a single backtest and display metrics.
sweep     -- Run all (strategy, symbol) combinations with ProcessPoolExecutor.
report    -- Read a sweep CSV and print a leaderboard / per-family summary.
export    -- Re-export a previously saved sweep CSV to Google Sheets.

Entry point: ``main(argv=None) -> int``.
"""

from __future__ import annotations

import argparse
import csv
import importlib
import math
import os
import sys
import textwrap
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _safe_fmt(val: Any, decimals: int = 4) -> str:
    """Format a numeric value safely, handling inf/nan."""
    if val is None:
        return "N/A"
    try:
        f = float(val)
    except (TypeError, ValueError):
        return str(val)
    if math.isnan(f):
        return "NaN"
    if math.isinf(f):
        return ">999" if f > 0 else "<-999"
    return f"{f:.{decimals}f}"


def _table(rows: list[list[str]], headers: list[str]) -> str:
    """Build a plain padded table string."""
    all_rows = [headers] + rows
    widths = [max(len(str(cell)) for cell in col) for col in zip(*all_rows)]
    sep = "  "
    lines = []
    for i, row in enumerate(all_rows):
        line = sep.join(str(cell).ljust(widths[j]) for j, cell in enumerate(row))
        lines.append(line)
        if i == 0:
            lines.append(sep.join("-" * w for w in widths))
    return "\n".join(lines)


def _err(msg: str) -> int:
    """Print error to stderr and return exit code 1."""
    print(f"ERROR: {msg}", file=sys.stderr)
    return 1


# ---------------------------------------------------------------------------
# Subcommand: download
# ---------------------------------------------------------------------------


def _cmd_download(args: argparse.Namespace) -> int:
    try:
        from fxlab.data import download  # type: ignore[import]
    except ImportError as exc:
        return _err(f"fxlab.data not available: {exc}")

    statuses: dict[str, str] = {}
    for sym in args.symbols:
        try:
            result = download([sym], timeframe=args.timeframe, source=args.source)
            statuses.update(result)
        except Exception as exc:  # noqa: BLE001
            statuses[sym] = f"FAILED: {exc}"

    rows = [[sym, status] for sym, status in statuses.items()]
    print(_table(rows, ["symbol", "status"]))
    return 0


# ---------------------------------------------------------------------------
# Subcommand: data
# ---------------------------------------------------------------------------


def _cmd_data(args: argparse.Namespace) -> int:
    try:
        from fxlab.data import list_cached  # type: ignore[import]
    except ImportError as exc:
        return _err(f"fxlab.data not available: {exc}")

    try:
        df = list_cached()
    except Exception as exc:  # noqa: BLE001
        return _err(f"Could not list cache: {exc}")

    if df.empty:
        print("Cache is empty.")
        return 0

    rows = []
    for _, row in df.iterrows():
        rows.append([
            str(row.get("symbol", "")),
            str(row.get("timeframe", "")),
            str(row.get("source", "")),
            str(row.get("rows", "")),
            str(row.get("start", ""))[:10],
            str(row.get("end", ""))[:10],
        ])
    print(_table(rows, ["symbol", "timeframe", "source", "rows", "start", "end"]))
    return 0


# ---------------------------------------------------------------------------
# Subcommand: list
# ---------------------------------------------------------------------------


def _cmd_list(args: argparse.Namespace) -> int:
    try:
        from fxlab.strategies import build_all  # type: ignore[import]
    except ImportError as exc:
        return _err(f"fxlab.strategies not available: {exc}")

    try:
        strategies = build_all()
    except Exception as exc:  # noqa: BLE001
        return _err(f"Could not build strategies: {exc}")

    if args.family:
        strategies = [s for s in strategies if s.family == args.family]

    rows = []
    for s in strategies:
        params_str = ", ".join(f"{k}={v}" for k, v in s.params.items())
        rows.append([s.id, s.family, params_str])

    print(_table(rows, ["id", "family", "params"]))
    print(f"\nTotal: {len(strategies)}")
    return 0


# ---------------------------------------------------------------------------
# Subcommand: run
# ---------------------------------------------------------------------------


def _build_plotly_html(df, result, symbol: str, strategy_id: str) -> str:  # noqa: ANN001
    """Build a Plotly HTML with candlestick, equity vs buy-and-hold, drawdown."""
    try:
        import numpy as np
        import plotly.graph_objects as go
        from plotly.subplots import make_subplots
    except ImportError:
        return "<html><body>plotly not available</body></html>"

    pos = result.position
    eq = result.equity

    # ---- buy&hold ----
    bh_returns = df["close"] / df["close"].iloc[0]

    # ---- drawdown ----
    running_max = eq.cummax()
    drawdown = eq / running_max - 1.0

    # ---- position shading colours ----
    # Build shape list from position series (long=green, short=red)
    shapes = []
    if len(pos) > 0:
        idx = pos.index
        prev_val = pos.iloc[0]
        seg_start = idx[0]
        for i in range(1, len(pos)):
            val = pos.iloc[i]
            if val != prev_val or i == len(pos) - 1:
                if prev_val != 0:
                    colour = "rgba(0,200,0,0.12)" if prev_val > 0 else "rgba(200,0,0,0.12)"
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
        subplot_titles=[f"{symbol} ({strategy_id})", "Equity vs Buy&Hold", "Drawdown"],
        vertical_spacing=0.04,
    )

    # Row 1: candlestick
    fig.add_trace(go.Candlestick(
        x=df.index,
        open=df["open"],
        high=df["high"],
        low=df["low"],
        close=df["close"],
        name="OHLCV",
        showlegend=False,
    ), row=1, col=1)

    # Row 2: equity curves
    fig.add_trace(go.Scatter(
        x=eq.index, y=eq, name="Strategy", line=dict(color="blue"),
    ), row=2, col=1)
    fig.add_trace(go.Scatter(
        x=bh_returns.index, y=bh_returns, name="Buy&Hold",
        line=dict(color="orange", dash="dash"),
    ), row=2, col=1)

    # Row 3: drawdown
    fig.add_trace(go.Scatter(
        x=drawdown.index, y=drawdown, name="Drawdown",
        fill="tozeroy", line=dict(color="red"),
    ), row=3, col=1)

    fig.update_layout(
        shapes=shapes,
        height=900,
        xaxis_rangeslider_visible=False,
        template="plotly_white",
    )

    return fig.to_html(full_html=True, include_plotlyjs="cdn")


def _cmd_run(args: argparse.Namespace) -> int:
    try:
        from fxlab.data import load_ohlcv  # type: ignore[import]
    except ImportError as exc:
        return _err(f"fxlab.data not available: {exc}")
    try:
        from fxlab.strategies import get as get_strategy  # type: ignore[import]
    except ImportError as exc:
        return _err(f"fxlab.strategies not available: {exc}")
    try:
        from fxlab.engine import run_backtest  # type: ignore[import]
    except ImportError as exc:
        return _err(f"fxlab.engine not available: {exc}")

    # Load data
    try:
        df = load_ohlcv(
            args.symbol,
            timeframe=args.timeframe,
            start=args.start,
            end=args.end,
            source=args.source,
        )
    except Exception as exc:  # noqa: BLE001
        return _err(f"Could not load data for {args.symbol}: {exc}")

    # Load strategy
    try:
        strategy = get_strategy(args.strategy)
    except Exception as exc:  # noqa: BLE001
        return _err(f"Could not find strategy {args.strategy!r}: {exc}")

    # Generate signal
    try:
        position = strategy.generate(df)
    except Exception as exc:  # noqa: BLE001
        return _err(f"Strategy generate() failed: {exc}")

    # Run backtest
    kwargs: dict[str, Any] = {"cost_bps": args.cost_bps}
    if args.sl_atr is not None:
        kwargs["sl_atr"] = args.sl_atr
    if args.tp_atr is not None:
        kwargs["tp_atr"] = args.tp_atr

    try:
        result = run_backtest(df, position, **kwargs)
    except Exception as exc:  # noqa: BLE001
        return _err(f"Backtest failed: {exc}")

    # Print metrics
    m = result.metrics
    metric_names = [
        "total_return", "cagr", "volatility", "sharpe", "sortino",
        "max_drawdown", "calmar", "win_rate", "profit_factor",
        "n_trades", "avg_trade_ret", "exposure", "time_in_market",
    ]
    rows = []
    for key in metric_names:
        val = m.get(key, float("nan"))
        if key == "n_trades":
            rows.append([key, str(int(val)) if not math.isnan(val) else "NaN"])
        else:
            rows.append([key, _safe_fmt(val)])

    print(f"\nBacktest: {args.strategy} on {args.symbol} [{args.timeframe}]")
    print(f"Bars: {len(df)}  |  Cost: {args.cost_bps} bps")
    if args.sl_atr:
        print(f"SL-ATR: {args.sl_atr}")
    if args.tp_atr:
        print(f"TP-ATR: {args.tp_atr}")
    print()
    print(_table(rows, ["metric", "value"]))

    # Optional plot
    if args.plot:
        html = _build_plotly_html(df, result, args.symbol, args.strategy)
        out_path = Path(args.plot)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(html, encoding="utf-8")
        print(f"\nPlot saved → {out_path}")

    return 0


# ---------------------------------------------------------------------------
# Sweep worker (must be importable at module level for pickling)
# ---------------------------------------------------------------------------


def _sweep_worker(
    strategy_id: str,
    strategy_family: str,
    strategy_params: dict,
    symbol: str,
    df_records: list,  # serialized df passed as list of dicts
    df_index: list,    # index as ISO strings
    cost_bps: float,
    sl_atr: float | None = None,
    tp_atr: float | None = None,
) -> dict:
    """Worker function for ProcessPoolExecutor sweep."""
    import pandas as pd

    try:
        from fxlab.strategies import get as get_strategy  # type: ignore[import]
        from fxlab.engine import run_backtest  # type: ignore[import]

        df = pd.DataFrame.from_records(df_records, index=pd.DatetimeIndex(df_index))
        for col in ("open", "high", "low", "close", "volume"):
            if col in df.columns:
                df[col] = df[col].astype("float64")

        strategy = get_strategy(strategy_id)
        position = strategy.generate(df)
        bt_kwargs: dict = {"cost_bps": cost_bps}
        if sl_atr is not None:
            bt_kwargs["sl_atr"] = sl_atr
        if tp_atr is not None:
            bt_kwargs["tp_atr"] = tp_atr
        result = run_backtest(df, position, **bt_kwargs)

        row = {
            "strategy_id": strategy_id,
            "family": strategy_family,
            "symbol": symbol,
        }
        row.update(result.metrics)
        return row
    except Exception as exc:  # noqa: BLE001
        return {
            "strategy_id": strategy_id,
            "family": strategy_family,
            "symbol": symbol,
            "error": str(exc),
        }


def _df_to_serializable(df) -> tuple[list, list]:  # noqa: ANN001
    """Convert a DataFrame to JSON-serializable records for inter-process transfer."""
    records = df.reset_index(drop=True).to_dict("records")
    index = [str(ts) for ts in df.index]
    return records, index


# ---------------------------------------------------------------------------
# Subcommand: sweep
# ---------------------------------------------------------------------------


def _resolve_strategy_filter(filter_str: str, all_strategies: list) -> list:
    """Resolve --strategies argument to a list of strategy objects."""
    if filter_str == "all":
        return all_strategies

    # family:NAME
    if filter_str.startswith("family:"):
        family = filter_str[len("family:"):]
        return [s for s in all_strategies if s.family == family]

    # comma-separated ids
    ids = {s.strip() for s in filter_str.split(",")}
    result = [s for s in all_strategies if s.id in ids]
    missing = ids - {s.id for s in result}
    if missing:
        print(f"Warning: unknown strategy ids: {', '.join(sorted(missing))}", file=sys.stderr)
    return result


def _cmd_sweep(args: argparse.Namespace) -> int:
    try:
        from fxlab.data import load_ohlcv  # type: ignore[import]
        from fxlab.strategies import build_all  # type: ignore[import]
    except ImportError as exc:
        return _err(f"Required module not available: {exc}")

    # Build strategies
    try:
        all_strategies = build_all()
    except Exception as exc:  # noqa: BLE001
        return _err(f"Could not build strategies: {exc}")

    strategies = _resolve_strategy_filter(args.strategies, all_strategies)
    if not strategies:
        return _err("No strategies matched the given filter.")

    symbols = args.symbols

    # Load all symbol data in parent process (once per symbol)
    symbol_data: dict[str, tuple[list, list]] = {}
    for sym in symbols:
        try:
            df = load_ohlcv(sym, timeframe=args.timeframe, source=args.source,
                            start=args.start, end=args.end)
            symbol_data[sym] = _df_to_serializable(df)
        except Exception as exc:  # noqa: BLE001
            print(f"Warning: could not load {sym}: {exc}", file=sys.stderr)

    if not symbol_data:
        return _err("No symbol data could be loaded.")

    # Build work items
    work_items = []
    for sym, (records, index) in symbol_data.items():
        for s in strategies:
            work_items.append((
                s.id, s.family, s.params,
                sym, records, index,
                args.cost_bps, args.sl_atr, args.tp_atr,
            ))

    total = len(work_items)
    print(f"Running {total} backtest(s) with {args.jobs} worker(s)...")

    results: list[dict] = []
    completed = 0

    with ProcessPoolExecutor(max_workers=args.jobs if args.jobs > 0 else None) as executor:
        futures = {
            executor.submit(_sweep_worker, *item): item
            for item in work_items
        }
        for fut in as_completed(futures):
            completed += 1
            print(f"\r  {completed}/{total}", end="", flush=True)
            try:
                row = fut.result()
                results.append(row)
            except Exception as exc:  # noqa: BLE001
                item = futures[fut]
                results.append({
                    "strategy_id": item[0],
                    "family": item[1],
                    "symbol": item[3],
                    "error": str(exc),
                })
    print()  # newline after progress

    if not results:
        return _err("No results produced.")

    # Save CSV
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    all_keys: list[str] = []
    seen: set[str] = set()
    for row in results:
        for k in row:
            if k not in seen:
                all_keys.append(k)
                seen.add(k)

    with out_path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=all_keys, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(results)

    print(f"Results saved → {out_path}")

    # Optional Google Sheets export
    if getattr(args, "gsheet", None):
        try:
            import pandas as _pd  # noqa: PLC0415
            import fxlab.export.gsheets as _gsheets_mod  # type: ignore[import]

            results_df = _pd.DataFrame(results)
            run_meta = {
                "symbols": " ".join(args.symbols),
                "timeframe": args.timeframe,
                "source": args.source,
                "start": args.start or "",
                "end": args.end or "",
                "cost_bps": args.cost_bps,
                "sl_atr": args.sl_atr or "",
                "tp_atr": args.tp_atr or "",
            }
            sheet_url = _gsheets_mod.export_sweep(results_df, run_meta, args.gsheet)
            print(f"Exported to Google Sheets → {sheet_url}")
        except Exception as exc:  # noqa: BLE001
            print(
                f"Warning: Google Sheets export failed: {exc} "
                "(check that the sheet is shared with the service-account email)",
                file=sys.stderr,
            )

    # Print leaderboard
    _print_leaderboard(results, metric=args.metric, top=args.top)
    return 0


# ---------------------------------------------------------------------------
# Subcommand: report
# ---------------------------------------------------------------------------


def _print_leaderboard(
    rows: list[dict],
    metric: str = "sharpe",
    top: int = 20,
    family_filter: str | None = None,
    symbol_filter: str | None = None,
) -> None:
    """Print a sorted leaderboard + per-family aggregate."""
    if family_filter:
        rows = [r for r in rows if r.get("family") == family_filter]
    if symbol_filter:
        rows = [r for r in rows if r.get("symbol") == symbol_filter]

    if not rows:
        print("No rows to display.")
        return

    def _sort_key(r: dict) -> float:
        val = r.get(metric, float("nan"))
        try:
            f = float(val)
        except (TypeError, ValueError):
            return float("-inf")
        if math.isnan(f) or math.isinf(f):
            return float("-inf")
        return f

    sorted_rows = sorted(rows, key=_sort_key, reverse=True)
    display_rows = sorted_rows[:top]

    table_rows = []
    for i, r in enumerate(display_rows, 1):
        val = r.get(metric, float("nan"))
        table_rows.append([
            str(i),
            str(r.get("strategy_id", "")),
            str(r.get("family", "")),
            str(r.get("symbol", "")),
            _safe_fmt(val),
        ])

    print(f"\nLeaderboard (top {len(display_rows)} by {metric})")
    print(_table(table_rows, ["#", "strategy_id", "family", "symbol", metric]))

    # Per-family aggregate (median metric)
    family_vals: dict[str, list[float]] = {}
    for r in rows:
        fam = str(r.get("family", ""))
        val = r.get(metric, float("nan"))
        try:
            f = float(val)
        except (TypeError, ValueError):
            continue
        if not math.isnan(f) and not math.isinf(f):
            family_vals.setdefault(fam, []).append(f)

    if family_vals:
        import statistics
        fam_summary = [
            (fam, statistics.median(vals), len(vals))
            for fam, vals in family_vals.items()
        ]
        fam_summary.sort(key=lambda x: x[1], reverse=True)
        fam_rows = [
            [fam, _safe_fmt(med), str(n)]
            for fam, med, n in fam_summary
        ]
        print(f"\nPer-family median {metric}")
        print(_table(fam_rows, ["family", f"median_{metric}", "n"]))


def _cmd_report(args: argparse.Namespace) -> int:
    in_path = Path(args.input)
    if not in_path.exists():
        return _err(f"File not found: {in_path}")

    try:
        with in_path.open("r", encoding="utf-8") as fh:
            reader = csv.DictReader(fh)
            rows = list(reader)
    except Exception as exc:  # noqa: BLE001
        return _err(f"Could not read {in_path}: {exc}")

    if not rows:
        print("CSV is empty.")
        return 0

    _print_leaderboard(
        rows,
        metric=args.metric,
        top=args.top,
        family_filter=getattr(args, "family", None),
        symbol_filter=getattr(args, "symbol", None),
    )
    return 0


# ---------------------------------------------------------------------------
# Subcommand: export
# ---------------------------------------------------------------------------


def _cmd_export(args: argparse.Namespace) -> int:
    """Re-export a previously saved sweep CSV to Google Sheets."""
    in_path = Path(args.input)
    if not in_path.exists():
        return _err(f"File not found: {in_path}")

    try:
        import pandas as _pd  # noqa: PLC0415
        results_df = _pd.read_csv(in_path)
    except Exception as exc:  # noqa: BLE001
        return _err(f"Could not read {in_path}: {exc}")

    if results_df.empty:
        print("CSV is empty; nothing to export.")
        return 0

    run_meta = {
        "symbols": getattr(args, "symbols", "") or "",
        "timeframe": getattr(args, "timeframe", "") or "",
        "source": getattr(args, "source", "") or "",
        "start": "",
        "end": "",
        "cost_bps": "",
        "sl_atr": "",
        "tp_atr": "",
    }

    try:
        import fxlab.export.gsheets as _gsheets_mod  # type: ignore[import]
        sheet_url = _gsheets_mod.export_sweep(results_df, run_meta, args.gsheet)
        print(f"Exported to Google Sheets → {sheet_url}")
    except Exception as exc:  # noqa: BLE001
        print(
            f"Warning: Google Sheets export failed: {exc} "
            "(check that the sheet is shared with the service-account email)",
            file=sys.stderr,
        )

    return 0


# ---------------------------------------------------------------------------
# Argument parser
# ---------------------------------------------------------------------------


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="fxlab",
        description="FXLab — FX/Gold/Crypto strategy verification platform",
    )
    sub = parser.add_subparsers(dest="command", metavar="<command>")

    # ---- download ----
    p_dl = sub.add_parser("download", help="Download OHLCV data for symbols")
    p_dl.add_argument("--symbols", nargs="+", required=True, metavar="SYM",
                      help="Symbols to download (e.g. EURUSD XAUUSD)")
    p_dl.add_argument("--timeframe", default="1d", metavar="TF",
                      help="Bar timeframe (default: 1d)")
    p_dl.add_argument("--source", default="auto",
                      choices=["auto", "stooq", "yfinance", "binance", "synthetic"],
                      help="Data source (default: auto)")

    # ---- data ----
    sub.add_parser("data", help="Show cache inventory")

    # ---- list ----
    p_list = sub.add_parser("list", help="List available strategies")
    p_list.add_argument("--family", default=None, metavar="F",
                        help="Filter by strategy family")

    # ---- run ----
    p_run = sub.add_parser("run", help="Run a backtest")
    p_run.add_argument("--strategy", required=True, metavar="ID",
                       help="Strategy id (e.g. sma_cross_10_50)")
    p_run.add_argument("--symbol", required=True, metavar="SYM",
                       help="Symbol (e.g. EURUSD)")
    p_run.add_argument("--timeframe", default="1d", metavar="TF",
                       help="Bar timeframe (default: 1d)")
    p_run.add_argument("--source", default="auto",
                       choices=["auto", "stooq", "yfinance", "binance", "synthetic"],
                       help="Data source (default: auto)")
    p_run.add_argument("--start", default=None, metavar="YYYY-MM-DD",
                       help="Start date")
    p_run.add_argument("--end", default=None, metavar="YYYY-MM-DD",
                       help="End date")
    p_run.add_argument("--cost-bps", type=float, default=1.0, metavar="BPS",
                       help="One-way transaction cost in bps (default: 1.0)")
    p_run.add_argument("--sl-atr", type=float, default=None, metavar="X",
                       help="Stop-loss ATR multiplier")
    p_run.add_argument("--tp-atr", type=float, default=None, metavar="X",
                       help="Take-profit ATR multiplier")
    p_run.add_argument("--plot", default=None, metavar="OUT.HTML",
                       help="Write Plotly HTML chart to this path")

    # ---- sweep ----
    p_sweep = sub.add_parser("sweep", help="Run a strategy/symbol sweep")
    p_sweep.add_argument("--symbols", nargs="+", required=True, metavar="SYM",
                         help="Symbols to sweep")
    p_sweep.add_argument("--strategies", default="all", metavar="FILTER",
                         help="all | id1,id2 | family:NAME (default: all)")
    p_sweep.add_argument("--timeframe", default="1d", metavar="TF",
                         help="Bar timeframe (default: 1d)")
    p_sweep.add_argument("--source", default="auto",
                         choices=["auto", "stooq", "yfinance", "binance", "synthetic"],
                         help="Data source (default: auto)")
    p_sweep.add_argument("--cost-bps", type=float, default=1.0, metavar="BPS",
                         help="One-way transaction cost in bps (default: 1.0)")
    p_sweep.add_argument("--start", default=None, metavar="YYYY-MM-DD",
                         help="Start date filter")
    p_sweep.add_argument("--end", default=None, metavar="YYYY-MM-DD",
                         help="End date filter")
    p_sweep.add_argument("--sl-atr", type=float, default=None, metavar="X",
                         help="ATR stop-loss multiple applied to every backtest")
    p_sweep.add_argument("--tp-atr", type=float, default=None, metavar="X",
                         help="ATR take-profit multiple applied to every backtest")
    p_sweep.add_argument("--top", type=int, default=20, metavar="N",
                         help="Number of top results to display (default: 20)")
    p_sweep.add_argument("--metric", default="sharpe", metavar="M",
                         help="Metric for ranking (default: sharpe)")
    p_sweep.add_argument("--out", default="results/sweep.csv", metavar="PATH",
                         help="Output CSV path (default: results/sweep.csv)")
    p_sweep.add_argument("--jobs", type=int, default=1, metavar="N",
                         help="Worker processes (default: 1; 0=cpu_count)")
    p_sweep.add_argument("--gsheet", default=None, metavar="SPREADSHEET",
                         help="Export results to Google Sheets (URL, ID, or title)")

    # ---- report ----
    p_report = sub.add_parser("report", help="Print leaderboard from a sweep CSV")
    p_report.add_argument("--in", dest="input", required=True, metavar="PATH",
                          help="Input sweep CSV path")
    p_report.add_argument("--metric", default="sharpe", metavar="M",
                          help="Metric for ranking (default: sharpe)")
    p_report.add_argument("--top", type=int, default=20, metavar="N",
                          help="Number of top results to display (default: 20)")
    p_report.add_argument("--family", default=None, metavar="F",
                          help="Filter by family")
    p_report.add_argument("--symbol", default=None, metavar="S",
                          help="Filter by symbol")

    # ---- export ----
    p_export = sub.add_parser(
        "export",
        help="Re-export a saved sweep CSV to Google Sheets",
    )
    p_export.add_argument("--in", dest="input", required=True, metavar="PATH",
                          help="Input sweep CSV path")
    p_export.add_argument("--gsheet", required=True, metavar="SPREADSHEET",
                          help="Target Google Sheets spreadsheet (URL, ID, or title)")
    p_export.add_argument("--timeframe", default="", metavar="TF",
                          help="Timeframe label for run_meta (optional)")
    p_export.add_argument("--source", default="", metavar="S",
                          help="Source label for run_meta (optional)")

    return parser


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

_COMMANDS = {
    "download": _cmd_download,
    "data": _cmd_data,
    "list": _cmd_list,
    "run": _cmd_run,
    "sweep": _cmd_sweep,
    "report": _cmd_report,
    "export": _cmd_export,
}


def main(argv: list[str] | None = None) -> int:
    """CLI entry point. Returns exit code (0 = success, 1 = error)."""
    parser = _build_parser()
    args = parser.parse_args(argv)

    if args.command is None:
        parser.print_help()
        return 1

    handler = _COMMANDS.get(args.command)
    if handler is None:
        return _err(f"Unknown command: {args.command}")

    try:
        return handler(args)
    except KeyboardInterrupt:
        print("\nInterrupted.", file=sys.stderr)
        return 1
