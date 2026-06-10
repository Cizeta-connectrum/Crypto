"""The FXLab backtest engine.

Two execution paths share a single public entry point :func:`run_backtest`:

* **Vectorized fast path** (``sl_atr`` and ``tp_atr`` both ``None``): pure
  pandas/numpy, close-to-close simple returns with the canonical next-bar shift.
* **Event-loop path** (any ATR stop/target supplied): an explicit per-bar loop
  that detects intrabar stop-loss / take-profit fills.

Both paths return a :class:`fxlab.core.BacktestResult`.

Conventions
-----------
* The strategy's *target* position is held over the *next* bar: the effective
  position is ``target.shift(1).fillna(0)``. Strategies never shift themselves.
* Per-bar return (no SL/TP)::

      ret_t = pos_t * (close_t / close_{t-1} - 1) - cost_t
      cost_t = (cost_bps / 1e4) * |pos_t - pos_{t-1}|

  The first bar has no prior close, so its price return is 0; only an entry cost
  (if ``pos_0 != 0``) is charged.
* Equity is ``(1 + ret).cumprod()`` and therefore starts at ``1 + ret_0``.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .. import core
from ..indicators import atr as atr_indicator
from .metrics import compute_metrics, infer_periods_per_year

__all__ = ["run_backtest"]


def run_backtest(
    df: pd.DataFrame,
    position: pd.Series,
    *,
    cost_bps: float = 1.0,
    sl_atr: float | None = None,
    tp_atr: float | None = None,
    atr_period: int = 14,
    periods_per_year: float | None = None,
) -> core.BacktestResult:
    """Backtest a target position series against an OHLCV frame.

    Parameters
    ----------
    df:
        A validated OHLCV frame (see :func:`fxlab.core.validate_ohlcv`).
    position:
        The strategy's target position in ``[-1, 1]`` aligned to ``df.index``.
        It is cleaned via :func:`fxlab.core.clean_position` before use.
    cost_bps:
        One-way transaction cost in basis points of notional, charged on every
        change in effective position: ``cost = cost_bps/1e4 * |Δpos|``.
    sl_atr, tp_atr:
        Optional ATR-multiple stop-loss / take-profit distances. If either is
        given the event-loop path is used.
    atr_period:
        Lookback for :func:`fxlab.indicators.atr` used to size SL/TP levels.
    periods_per_year:
        Annualization factor for metrics; inferred from the index when ``None``.

    Returns
    -------
    fxlab.core.BacktestResult
    """
    target = core.clean_position(position, df.index)
    close = df["close"].astype("float64")

    if len(df) == 0:
        empty_ret = pd.Series(dtype="float64", index=df.index)
        empty_eq = pd.Series(dtype="float64", index=df.index)
        trades = _empty_trades()
        ppy = periods_per_year if periods_per_year is not None else 252.0
        metrics = compute_metrics(empty_ret, empty_eq, target, trades, ppy)
        return core.BacktestResult(empty_eq, empty_ret, target, trades, metrics)

    if sl_atr is None and tp_atr is None:
        returns, eff_pos = _vectorized_path(close, target, cost_bps)
    else:
        atr_series = atr_indicator(df, atr_period)
        returns, eff_pos = _event_loop_path(
            df, target, cost_bps, sl_atr, tp_atr, atr_series
        )

    equity = (1.0 + returns).cumprod()
    trades = _build_trades(eff_pos, returns, df.index)

    ppy = (
        periods_per_year
        if periods_per_year is not None
        else infer_periods_per_year(df.index)
    )
    metrics = compute_metrics(returns, equity, eff_pos, trades, ppy)

    return core.BacktestResult(
        equity=equity,
        returns=returns,
        position=eff_pos,
        trades=trades,
        metrics=metrics,
    )


# ---------------------------------------------------------------------------
# Vectorized fast path
# ---------------------------------------------------------------------------
def _vectorized_path(
    close: pd.Series, target: pd.Series, cost_bps: float
) -> tuple[pd.Series, pd.Series]:
    """Close-to-close vectorized returns; no SL/TP. Returns (returns, eff_pos)."""
    pos = target.shift(1).fillna(0.0)
    price_ret = close.pct_change().fillna(0.0)
    cost = (cost_bps / 1e4) * pos.diff().abs().fillna(pos.abs())
    returns = pos * price_ret - cost
    returns = returns.astype("float64")
    returns.name = None
    return returns, pos.astype("float64")


# ---------------------------------------------------------------------------
# Event-loop path (SL/TP)
# ---------------------------------------------------------------------------
def _event_loop_path(
    df: pd.DataFrame,
    target: pd.Series,
    cost_bps: float,
    sl_atr: float | None,
    tp_atr: float | None,
    atr_series: pd.Series,
) -> tuple[pd.Series, pd.Series]:
    """Per-bar event loop with intrabar SL/TP fills.

    Fill model (documented and consistent):

    * The *desired* position for bar ``t`` is ``target[t-1]`` (the usual
      next-bar shift). All position changes — entries, exits and sign flips —
      execute at the boundary between bars, i.e. at ``close[t-1]``, exactly as
      in the vectorized path. A new trade's entry price and ATR are therefore
      taken from bar ``t-1`` and the trade is first exposed to SL/TP on bar
      ``t``. With protective levels that can never trigger, this path
      reproduces the vectorized path bar for bar.
    * For a held long trade entered at ``entry`` with ``a = ATR`` at entry:
      ``stop = entry - sl_atr*a``, ``tp = entry + tp_atr*a`` (mirrored for
      shorts). On each held bar we check, in this order (stop has priority — the
      conservative assumption when both are touched in one bar):

        - Stop: long if ``low <= stop`` (short if ``high >= stop``). If the bar
          *gapped beyond* the level (``open <= stop`` long / ``open >= stop``
          short) the fill is at ``open``; otherwise at ``stop``.
        - Take-profit: long if ``high >= tp`` (short if ``low <= tp``). Gap fill
          at ``open`` if ``open >= tp`` long / ``open <= tp`` short, else at
          ``tp``.

    * The bar's return splits into the part earned up to the fill and zero
      thereafter (flat for the remainder of the bar)::

          ret_bar = pos * (fill_price / prev_ref - 1) - cost

      where ``prev_ref`` is the previous bar's close (or, on the entry bar, the
      entry price) and ``cost`` covers the position change(s) on that bar.
    * After a stop/tp exit the position is forced flat and stays flat until the
      *target* series changes value from what it was at entry (a genuinely new
      signal), preventing immediate re-entry into the same exited trade.
    """
    n = len(df)
    o = df["open"].to_numpy(dtype="float64")
    h = df["high"].to_numpy(dtype="float64")
    low = df["low"].to_numpy(dtype="float64")
    c = df["close"].to_numpy(dtype="float64")
    tgt = target.to_numpy(dtype="float64")
    a = atr_series.to_numpy(dtype="float64")

    sl = float(sl_atr) if sl_atr is not None else None
    tp = float(tp_atr) if tp_atr is not None else None
    cost_rate = cost_bps / 1e4

    returns = np.zeros(n, dtype="float64")
    eff_pos = np.zeros(n, dtype="float64")  # position held over each bar

    held = 0.0          # current effective position
    stop_level = np.nan
    tp_level = np.nan
    # Target value at the time the current (or last) trade was opened. While
    # flat-after-stop we wait for tgt to differ from this to allow re-entry.
    blocked_target = None  # target value we are "locked out" against

    for t in range(n):
        # Desired position for this bar comes from the previous bar's target.
        desired = tgt[t - 1] if t >= 1 else 0.0

        # Flat-after-stop lockout: suppress the unchanged entry signal; lift
        # the lockout as soon as the target takes any other value.
        if held == 0.0 and blocked_target is not None:
            if desired == blocked_target:
                desired = 0.0
            else:
                blocked_target = None

        # Boundary execution at close[t-1]: bring the position to `desired`
        # (entry, exit or sign flip), charging cost on the change.
        cost = 0.0
        if desired != held:
            cost = cost_rate * abs(desired - held)
            if desired != 0.0:
                held, _, stop_level, tp_level, blocked_target = _open(
                    desired, c[t - 1], a[t - 1], sl, tp, desired
                )
            else:
                held = 0.0
                stop_level = tp_level = np.nan
                blocked_target = None

        if held == 0.0:
            returns[t] = -cost
            eff_pos[t] = 0.0
            continue

        # Held bar: check intrabar SL/TP against the previous close reference.
        prev_ref = c[t - 1]
        fill_price = None
        if held > 0:
            # Long: stop first (conservative), then tp.
            if not np.isnan(stop_level) and low[t] <= stop_level:
                fill_price = o[t] if o[t] <= stop_level else stop_level
            elif not np.isnan(tp_level) and h[t] >= tp_level:
                fill_price = o[t] if o[t] >= tp_level else tp_level
        else:
            # Short: stop first, then tp.
            if not np.isnan(stop_level) and h[t] >= stop_level:
                fill_price = o[t] if o[t] >= stop_level else stop_level
            elif not np.isnan(tp_level) and low[t] <= tp_level:
                fill_price = o[t] if o[t] <= tp_level else tp_level

        if fill_price is not None:
            # Exit intrabar; flat for the remainder of the bar. blocked_target
            # (set at entry) is retained to lock out immediate re-entry.
            exit_cost = cost_rate * abs(held)
            returns[t] = held * (fill_price / prev_ref - 1.0) - cost - exit_cost
            eff_pos[t] = held
            held = 0.0
            stop_level = tp_level = np.nan
            continue

        returns[t] = held * (c[t] / prev_ref - 1.0) - cost
        eff_pos[t] = held

    ret_series = pd.Series(returns, index=df.index, dtype="float64")
    pos_series = pd.Series(eff_pos, index=df.index, dtype="float64")
    return ret_series, pos_series


def _open(
    desired: float,
    entry_price: float,
    atr_at_entry: float,
    sl: float | None,
    tp: float | None,
    entry_target: float,
) -> tuple[float, float, float, float, float]:
    """Open a trade; return (held, entry_price, stop_level, tp_level, blocked)."""
    if np.isnan(atr_at_entry):
        # No ATR yet -> cannot size SL/TP; hold without protective levels.
        stop_level = np.nan
        tp_level = np.nan
    else:
        if desired > 0:
            stop_level = entry_price - sl * atr_at_entry if sl is not None else np.nan
            tp_level = entry_price + tp * atr_at_entry if tp is not None else np.nan
        else:
            stop_level = entry_price + sl * atr_at_entry if sl is not None else np.nan
            tp_level = entry_price - tp * atr_at_entry if tp is not None else np.nan
    return desired, entry_price, stop_level, tp_level, entry_target


# ---------------------------------------------------------------------------
# Trades table
# ---------------------------------------------------------------------------
def _empty_trades() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "entry_time": pd.Series([], dtype="datetime64[ns]"),
            "exit_time": pd.Series([], dtype="datetime64[ns]"),
            "direction": pd.Series([], dtype="int64"),
            "bars": pd.Series([], dtype="int64"),
            "ret": pd.Series([], dtype="float64"),
        }
    )


def _build_trades(
    eff_pos: pd.Series, returns: pd.Series, index: pd.Index
) -> pd.DataFrame:
    """Derive the trades table from the effective position and per-bar returns.

    A trade spans the consecutive run of bars on which a single sign of the
    effective position is held. ``ret`` is the compounded net return over those
    bars (product of ``1+ret`` minus 1), which therefore embeds entry/exit costs.
    An open trade at the end is closed at the last bar. ``bars`` counts held bars.
    """
    pos = eff_pos.to_numpy(dtype="float64")
    ret = returns.to_numpy(dtype="float64")
    n = len(pos)
    rows = []

    t = 0
    while t < n:
        if pos[t] == 0.0:
            t += 1
            continue
        direction = 1 if pos[t] > 0 else -1
        start = t
        # Extend while same sign held.
        while t < n and (1 if pos[t] > 0 else (-1 if pos[t] < 0 else 0)) == direction:
            t += 1
        end = t - 1  # inclusive last held bar
        held_bars = end - start + 1
        # The closing transaction cost is charged on the bar where the position
        # leaves this trade. If that next bar is flat (pos == 0) it is otherwise
        # unattributed, so fold its (cost-only) return into this trade's return
        # to keep the trades table compounding to the equity curve exactly. A
        # direct sign flip instead attaches that bar (entry of the next trade)
        # to the next trade, so we do not absorb it here.
        last = end
        if t < n and pos[t] == 0.0:
            last = t  # include the exit-cost-only flat bar
        comp = float(np.prod(1.0 + ret[start : last + 1]) - 1.0)
        rows.append(
            {
                "entry_time": index[start],
                "exit_time": index[end],
                "direction": direction,
                "bars": int(held_bars),
                "ret": comp,
            }
        )

    if not rows:
        return _empty_trades()
    out = pd.DataFrame(rows)
    out["direction"] = out["direction"].astype("int64")
    out["bars"] = out["bars"].astype("int64")
    out["ret"] = out["ret"].astype("float64")
    return out
