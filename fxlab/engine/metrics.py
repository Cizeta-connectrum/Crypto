"""Performance metrics for backtest results.

All functions are pure and operate on the per-bar return / equity series and the
trades table produced by :func:`fxlab.engine.backtest.run_backtest`. Annualization
relies on a ``periods_per_year`` factor, which can be inferred from the index
spacing via :func:`infer_periods_per_year`.

Notes on ``profit_factor``: it is gross profit divided by gross loss. When there
are no losing trades (and at least one winning trade) this is mathematically
``+inf``; we return ``float('inf')`` as-is. When there are no trades at all (or
no wins and no losses) we return ``np.nan``. Consumers (CLI/UI) should treat inf
robustly (e.g. display as ">999" or sort it last/first explicitly).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

SECONDS_PER_YEAR = 365.0 * 24.0 * 3600.0


def infer_periods_per_year(index: pd.Index) -> float:
    """Estimate the number of bars per year from an index's calendar coverage.

    Uses the *observed* calendar span rather than the nominal bar spacing, so a
    business-day daily index (which skips weekends) yields ~250-260 and a fully
    populated hourly index yields close to ``24*365``. Concretely::

        bars_per_year = (n_bars - 1) / total_seconds_spanned * SECONDS_PER_YEAR

    For fewer than 2 timestamps (or a zero/degenerate span) we fall back to a
    sensible default of 252.0.
    """
    if index is None or len(index) < 2:
        return 252.0
    idx = pd.DatetimeIndex(index)
    total_seconds = (idx[-1] - idx[0]).total_seconds()
    if total_seconds <= 0:
        return 252.0
    return float(len(idx) - 1) / total_seconds * SECONDS_PER_YEAR


def _annualization(periods_per_year: float | None, index: pd.Index | None) -> float:
    if periods_per_year is not None and periods_per_year > 0:
        return float(periods_per_year)
    return infer_periods_per_year(index) if index is not None else 252.0


def max_drawdown(equity: pd.Series) -> float:
    """Maximum drawdown of an equity curve as a non-positive fraction.

    Returns 0.0 for an empty/flat curve, otherwise the most negative value of
    ``equity / running_max - 1`` (e.g. -0.25 for a 25% peak-to-trough decline).
    """
    eq = np.asarray(equity, dtype="float64")
    if eq.size == 0:
        return 0.0
    running_max = np.maximum.accumulate(eq)
    # Guard against zero/negative running max (degenerate); avoid div-by-zero.
    with np.errstate(divide="ignore", invalid="ignore"):
        dd = eq / running_max - 1.0
    dd = dd[np.isfinite(dd)]
    if dd.size == 0:
        return 0.0
    return float(min(dd.min(), 0.0))


def compute_metrics(
    returns: pd.Series,
    equity: pd.Series,
    position: pd.Series,
    trades: pd.DataFrame,
    periods_per_year: float | None = None,
) -> dict[str, float]:
    """Compute the full metrics suite for a backtest.

    Parameters
    ----------
    returns:
        Per-bar simple returns net of costs.
    equity:
        Compounded equity curve (starts at 1.0).
    position:
        Effective (post-shift) position actually held each bar.
    trades:
        Trades table with at least a ``ret`` and ``direction`` column.
    periods_per_year:
        Annualization factor; if ``None`` it is inferred from the returns index.

    Returns
    -------
    dict
        total_return, cagr, volatility, sharpe, sortino, max_drawdown, calmar,
        win_rate, profit_factor, n_trades, avg_trade_ret, exposure,
        time_in_market.
    """
    ret = pd.Series(returns, dtype="float64").dropna()
    eq = pd.Series(equity, dtype="float64").dropna()
    ppy = _annualization(periods_per_year, ret.index if len(ret) else None)

    n = int(len(ret))
    ret_vals = ret.to_numpy()

    # Total return / CAGR.
    if len(eq) == 0:
        total_return = 0.0
    else:
        total_return = float(eq.iloc[-1] / eq.iloc[0] - 1.0)

    if len(eq) >= 2 and eq.iloc[0] > 0 and eq.iloc[-1] > 0:
        years = n / ppy if ppy > 0 else np.nan
        if years and years > 0:
            cagr = float((eq.iloc[-1] / eq.iloc[0]) ** (1.0 / years) - 1.0)
        else:
            cagr = np.nan
    else:
        cagr = np.nan

    # Volatility (annualized std of per-bar returns).
    if n >= 2:
        std = float(np.std(ret_vals, ddof=1))
        volatility = std * np.sqrt(ppy)
    else:
        std = 0.0
        volatility = 0.0

    mean_ret = float(np.mean(ret_vals)) if n >= 1 else 0.0

    # Sharpe (rf = 0). If volatility is exactly zero, undefined -> nan, unless
    # there are no nonzero returns at all (-> 0.0).
    if n >= 2 and std > 0:
        sharpe = mean_ret / std * np.sqrt(ppy)
    elif n >= 2 and mean_ret == 0.0:
        sharpe = 0.0
    elif n >= 2:
        # Zero volatility but nonzero mean -> infinite risk-adjusted return.
        sharpe = np.inf if mean_ret > 0 else -np.inf
    else:
        sharpe = np.nan

    # Sortino (downside deviation, rf = 0).
    if n >= 2:
        downside = ret_vals[ret_vals < 0]
        if downside.size > 0:
            # Downside deviation uses the full sample size in the denominator.
            dd_std = float(np.sqrt(np.sum(downside ** 2) / n))
        else:
            dd_std = 0.0
        if dd_std > 0:
            sortino = mean_ret / dd_std * np.sqrt(ppy)
        elif mean_ret == 0.0:
            sortino = 0.0
        else:
            sortino = np.inf if mean_ret > 0 else -np.inf
    else:
        sortino = np.nan

    mdd = max_drawdown(eq)

    # Calmar = CAGR / |max drawdown|.
    if np.isfinite(cagr) and mdd < 0:
        calmar = float(cagr / abs(mdd))
    else:
        calmar = np.nan

    # Trade-based metrics.
    if trades is not None and len(trades) > 0 and "ret" in trades.columns:
        trade_ret = pd.Series(trades["ret"], dtype="float64").to_numpy()
        n_trades = int(trade_ret.size)
        wins = trade_ret[trade_ret > 0]
        losses = trade_ret[trade_ret < 0]
        win_rate = float(wins.size / n_trades) if n_trades > 0 else np.nan
        gross_profit = float(wins.sum())
        gross_loss = float(-losses.sum())
        if gross_loss > 0:
            profit_factor = gross_profit / gross_loss
        elif gross_profit > 0:
            profit_factor = np.inf
        else:
            profit_factor = np.nan
        avg_trade_ret = float(trade_ret.mean())
    else:
        n_trades = 0
        win_rate = np.nan
        profit_factor = np.nan
        avg_trade_ret = np.nan

    # Exposure / time in market.
    pos = pd.Series(position, dtype="float64").dropna()
    if len(pos) > 0:
        abs_pos = pos.abs().to_numpy()
        exposure = float(abs_pos.mean())
        time_in_market = float(np.mean(abs_pos > 0))
    else:
        exposure = 0.0
        time_in_market = 0.0

    return {
        "total_return": total_return,
        "cagr": cagr,
        "volatility": volatility,
        "sharpe": sharpe,
        "sortino": sortino,
        "max_drawdown": mdd,
        "calmar": calmar,
        "win_rate": win_rate,
        "profit_factor": profit_factor,
        "n_trades": n_trades,
        "avg_trade_ret": avg_trade_ret,
        "exposure": exposure,
        "time_in_market": time_in_market,
    }
