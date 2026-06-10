"""Tests for the FXLab backtest engine (vectorized & SL/TP paths) and metrics."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from fxlab import core
from fxlab.engine import (
    compute_metrics,
    infer_periods_per_year,
    max_drawdown,
    run_backtest,
)


def _make_df(closes, opens=None, highs=None, lows=None, index=None):
    closes = np.asarray(closes, dtype="float64")
    n = len(closes)
    if index is None:
        index = pd.date_range("2020-01-01", periods=n, freq="D")
    opens = np.asarray(opens, dtype="float64") if opens is not None else closes.copy()
    highs = (
        np.asarray(highs, dtype="float64")
        if highs is not None
        else np.maximum(opens, closes)
    )
    lows = (
        np.asarray(lows, dtype="float64")
        if lows is not None
        else np.minimum(opens, closes)
    )
    df = pd.DataFrame(
        {
            "open": opens,
            "high": highs,
            "low": lows,
            "close": closes,
            "volume": np.ones(n),
        },
        index=index,
    )
    return core.validate_ohlcv(df)


# ---------------------------------------------------------------------------
# Hand-computed vectorized case
# ---------------------------------------------------------------------------
def test_hand_computed_equity_and_returns():
    closes = [100.0, 110.0, 99.0, 99.0, 108.9]
    df = _make_df(closes)
    # Target position by bar (the *signal* known at that bar's close):
    # bar0 -> hold long over bar1, bar1 -> flat over bar2, etc.
    target = pd.Series([1.0, 0.0, 1.0, 1.0, 0.0], index=df.index)
    cost_bps = 10.0  # 0.001 per unit turnover
    res = run_backtest(df, target, cost_bps=cost_bps)

    pos = target.shift(1).fillna(0.0).to_numpy()  # [0,1,0,1,1]
    c = np.asarray(closes)
    price_ret = np.concatenate([[0.0], c[1:] / c[:-1] - 1.0])
    turnover = np.abs(np.diff(np.concatenate([[0.0], pos])))
    cost = (cost_bps / 1e4) * turnover
    expected_ret = pos * price_ret - cost
    expected_eq = np.cumprod(1.0 + expected_ret)

    np.testing.assert_allclose(res.returns.to_numpy(), expected_ret, atol=1e-12)
    np.testing.assert_allclose(res.equity.to_numpy(), expected_eq, atol=1e-12)


# ---------------------------------------------------------------------------
# Buy & hold
# ---------------------------------------------------------------------------
def test_buy_and_hold_matches_price_ratio_minus_entry_cost():
    closes = [100.0, 105.0, 103.0, 120.0, 119.0]
    df = _make_df(closes)
    target = pd.Series(1.0, index=df.index)
    cost_bps = 5.0
    res = run_backtest(df, target, cost_bps=cost_bps)

    # Effective position is shifted: flat on bar0, enter at bar1 (pos 0->1).
    # A single entry cost (additive within that bar's return) is charged; there
    # is no exit (open trade at the end). Gross captures the full close ratio.
    c = np.asarray(closes)
    price_ret = np.concatenate([[0.0], c[1:] / c[:-1] - 1.0])
    entry_cost = cost_bps / 1e4
    expected_ret = price_ret.copy()
    expected_ret[1] -= entry_cost  # entry on bar1
    expected_final = float(np.prod(1.0 + expected_ret))
    assert res.equity.iloc[-1] == pytest.approx(expected_final, abs=1e-12)
    # Sanity: with zero cost this is exactly the close ratio.
    res0 = run_backtest(df, target, cost_bps=0.0)
    assert res0.equity.iloc[-1] == pytest.approx(closes[-1] / closes[0], abs=1e-12)


# ---------------------------------------------------------------------------
# No lookahead (shift) check
# ---------------------------------------------------------------------------
def test_no_lookahead_shift():
    # Big up-move happens between bar2 and bar3 (close 100 -> 200).
    closes = [100.0, 101.0, 100.0, 200.0, 201.0]
    df = _make_df(closes)
    # Position +1 ONLY on the bar with the biggest up-move (bar3, the 200 bar).
    target = pd.Series([0.0, 0.0, 0.0, 1.0, 0.0], index=df.index)
    res = run_backtest(df, target, cost_bps=0.0)

    # The +1 at bar3 is held over bar4, so it captures 201/200-1, NOT the 100->200 jump.
    assert res.returns.iloc[3] == pytest.approx(0.0, abs=1e-12)
    assert res.returns.iloc[4] == pytest.approx(201.0 / 200.0 - 1.0, abs=1e-12)
    # It must NOT have captured the +100% move.
    assert res.equity.iloc[-1] < 1.5


# ---------------------------------------------------------------------------
# Costs on flipping
# ---------------------------------------------------------------------------
def test_flip_cost_is_two_cost_bps():
    closes = [100.0, 100.0, 100.0, 100.0, 100.0]  # constant -> isolate cost
    df = _make_df(closes)
    target = pd.Series([1.0, -1.0, 1.0, -1.0, 1.0], index=df.index)
    cost_bps = 7.0
    res = run_backtest(df, target, cost_bps=cost_bps)

    # Effective pos = [0, 1, -1, 1, -1]. Flip turnover from bar1->bar2 is |−1−1|=2.
    per_flip = 2.0 * cost_bps / 1e4
    # bar2,3,4 are flips; price return is zero, so ret == -per_flip there.
    assert res.returns.iloc[2] == pytest.approx(-per_flip, abs=1e-12)
    assert res.returns.iloc[3] == pytest.approx(-per_flip, abs=1e-12)
    assert res.returns.iloc[4] == pytest.approx(-per_flip, abs=1e-12)
    # bar1: entry 0->1 turnover 1 -> one-way cost.
    assert res.returns.iloc[1] == pytest.approx(-cost_bps / 1e4, abs=1e-12)


# ---------------------------------------------------------------------------
# SL/TP fills
# ---------------------------------------------------------------------------
def test_stop_loss_fill_at_level():
    # Enter long at close=100 on bar0; ATR fixed so stop=95.
    # bar2 pierces with low=90, open=98 -> exit at the stop level 95.
    n = 20
    closes = [100.0] * n
    opens = [100.0] * n
    highs = [100.0] * n
    lows = [100.0] * n
    # Make ATR well-defined and equal to 1.0 by a steady 1-wide range early.
    for i in range(n):
        highs[i] = closes[i] + 0.5
        lows[i] = closes[i] - 0.5
    # The piercing bar: index 16 (after ATR warmup of 14).
    pierce = 16
    opens[pierce] = 98.0
    highs[pierce] = 98.0
    lows[pierce] = 90.0
    closes[pierce] = 90.0
    df = _make_df(closes, opens=opens, highs=highs, lows=lows)

    # Go long starting at bar 14 (ATR defined ~1.0 there); entry at bar14 close=100,
    # held from bar15. stop = 100 - 5*1 = 95.
    target = pd.Series(0.0, index=df.index)
    target.iloc[14:] = 1.0
    res = run_backtest(df, target, cost_bps=0.0, sl_atr=5.0, atr_period=14)

    # prev close before pierce bar is 100; fill at the stop level 95.
    fill_ret = (95.0 / 100.0) - 1.0  # long * (95/100 - 1)
    assert res.returns.iloc[pierce] == pytest.approx(fill_ret, abs=1e-9)
    # After the stop, flat until target changes (it never does) -> stays flat.
    assert (res.position.iloc[pierce + 1 :] == 0.0).all()


def test_stop_loss_gap_fills_at_open():
    n = 20
    closes = [100.0] * n
    opens = [100.0] * n
    highs = [100.0] * n
    lows = [100.0] * n
    for i in range(n):
        highs[i] = closes[i] + 0.5
        lows[i] = closes[i] - 0.5
    pierce = 16
    opens[pierce] = 92.0  # gaps below stop=95
    highs[pierce] = 92.0
    lows[pierce] = 90.0
    closes[pierce] = 91.0
    df = _make_df(closes, opens=opens, highs=highs, lows=lows)

    target = pd.Series(0.0, index=df.index)
    target.iloc[14:] = 1.0
    res = run_backtest(df, target, cost_bps=0.0, sl_atr=5.0, atr_period=14)
    fill_ret = (92.0 / 100.0) - 1.0  # exit at the gapped open
    assert res.returns.iloc[pierce] == pytest.approx(fill_ret, abs=1e-9)


def test_take_profit_fill_at_level():
    n = 20
    closes = [100.0] * n
    opens = [100.0] * n
    highs = [100.0] * n
    lows = [100.0] * n
    for i in range(n):
        highs[i] = closes[i] + 0.5
        lows[i] = closes[i] - 0.5
    hit = 16
    opens[hit] = 102.0
    highs[hit] = 110.0  # touches tp=105
    lows[hit] = 102.0
    closes[hit] = 108.0
    df = _make_df(closes, opens=opens, highs=highs, lows=lows)

    target = pd.Series(0.0, index=df.index)
    target.iloc[14:] = 1.0
    # tp_atr=5 with ATR~1 -> tp=105. No stop.
    res = run_backtest(df, target, cost_bps=0.0, tp_atr=5.0, atr_period=14)
    fill_ret = (105.0 / 100.0) - 1.0
    assert res.returns.iloc[hit] == pytest.approx(fill_ret, abs=1e-9)
    assert (res.position.iloc[hit + 1 :] == 0.0).all()


def test_reentry_only_after_target_changes():
    n = 25
    closes = [100.0] * n
    opens = [100.0] * n
    highs = [100.0] * n
    lows = [100.0] * n
    for i in range(n):
        highs[i] = closes[i] + 0.5
        lows[i] = closes[i] - 0.5
    pierce = 16
    opens[pierce] = 98.0
    highs[pierce] = 98.0
    lows[pierce] = 90.0
    closes[pierce] = 96.0
    df = _make_df(closes, opens=opens, highs=highs, lows=lows)

    target = pd.Series(0.0, index=df.index)
    target.iloc[14:] = 1.0
    # Drop to flat then back to long after the stop, to allow re-entry.
    target.iloc[pierce + 2] = 0.0
    target.iloc[pierce + 3 :] = 1.0
    res = run_backtest(df, target, cost_bps=0.0, sl_atr=5.0, atr_period=14)

    # Immediately after stop and before target changes -> flat.
    assert res.position.iloc[pierce + 1] == 0.0
    # After target goes to 0 then back to 1, re-entry occurs (held again).
    assert (res.position.iloc[pierce + 5 :] == 1.0).any()


# ---------------------------------------------------------------------------
# Trades table consistency
# ---------------------------------------------------------------------------
def test_trades_compound_to_equity():
    rng = np.random.default_rng(0)
    n = 200
    rets = rng.normal(0, 0.01, n)
    closes = 100.0 * np.cumprod(1.0 + rets)
    df = _make_df(closes)
    target = pd.Series(rng.choice([-1.0, 0.0, 1.0], size=n), index=df.index)
    res = run_backtest(df, target, cost_bps=2.0)

    # Product of (1+trade_ret) over all trades should match the equity built
    # from only in-trade bars; since flat bars contribute zero return, the full
    # equity equals the product over trades.
    if len(res.trades) > 0:
        prod_trades = np.prod(1.0 + res.trades["ret"].to_numpy())
        assert prod_trades == pytest.approx(res.equity.iloc[-1], rel=1e-9)

    # Directions correct: every trade direction matches the sign of the held pos.
    for _, tr in res.trades.iterrows():
        seg = res.position.loc[tr["entry_time"] : tr["exit_time"]]
        assert np.sign(seg.iloc[0]) == tr["direction"]


def test_trades_compound_to_equity_sltp_path():
    rng = np.random.default_rng(3)
    n = 300
    rets = rng.normal(0, 0.015, n)
    closes = 100.0 * np.cumprod(1.0 + rets)
    highs = closes * (1.0 + np.abs(rng.normal(0, 0.01, n)))
    lows = closes * (1.0 - np.abs(rng.normal(0, 0.01, n)))
    opens = closes * (1.0 + rng.normal(0, 0.005, n))
    highs = np.maximum.reduce([highs, opens, closes])
    lows = np.minimum.reduce([lows, opens, closes])
    df = _make_df(closes, opens=opens, highs=highs, lows=lows)
    target = pd.Series(rng.choice([-1.0, 0.0, 1.0], size=n), index=df.index)
    res = run_backtest(df, target, cost_bps=1.0, sl_atr=2.0, tp_atr=3.0, atr_period=14)
    if len(res.trades) > 0:
        prod_trades = np.prod(1.0 + res.trades["ret"].to_numpy())
        assert prod_trades == pytest.approx(res.equity.iloc[-1], rel=1e-9)


def test_open_trade_closed_at_last_bar():
    closes = [100.0, 110.0, 121.0]
    df = _make_df(closes)
    target = pd.Series([1.0, 1.0, 1.0], index=df.index)
    res = run_backtest(df, target, cost_bps=0.0)
    assert len(res.trades) == 1
    assert res.trades["exit_time"].iloc[0] == df.index[-1]
    assert res.trades["direction"].iloc[0] == 1


# ---------------------------------------------------------------------------
# Metrics math
# ---------------------------------------------------------------------------
def test_max_drawdown_known_path():
    eq = pd.Series([1.0, 1.2, 0.9, 1.0, 0.6])
    # Peak 1.2, trough 0.6 -> dd = 0.6/1.2 - 1 = -0.5.
    assert max_drawdown(eq) == pytest.approx(-0.5, abs=1e-12)


def test_sharpe_constant_returns_inf_safe():
    # Constant positive returns -> zero std -> sharpe should be inf (not crash).
    ret = pd.Series([0.001] * 50, index=pd.date_range("2020", periods=50, freq="D"))
    eq = (1 + ret).cumprod()
    pos = pd.Series(1.0, index=ret.index)
    m = compute_metrics(ret, eq, pos, pd.DataFrame({"ret": [0.05], "direction": [1]}), 252)
    assert np.isinf(m["sharpe"]) or m["sharpe"] > 1e6


def test_infer_periods_per_year_business_daily():
    idx = pd.bdate_range("2018-01-01", "2022-12-31")
    ppy = infer_periods_per_year(idx)
    assert 250 <= ppy <= 262


def test_infer_periods_per_year_hourly():
    idx = pd.date_range("2021-01-01", periods=24 * 365, freq="h")
    ppy = infer_periods_per_year(idx)
    target = 24 * 365
    assert abs(ppy - target) / target < 0.10


def test_infer_periods_per_year_crypto_daily():
    idx = pd.date_range("2019-01-01", periods=730, freq="D")  # includes weekends
    ppy = infer_periods_per_year(idx)
    assert 360 <= ppy <= 366


# ---------------------------------------------------------------------------
# Degenerate inputs
# ---------------------------------------------------------------------------
def test_all_flat_position():
    df = _make_df([100.0, 101.0, 102.0, 103.0])
    target = pd.Series(0.0, index=df.index)
    res = run_backtest(df, target, cost_bps=1.0)
    assert (res.returns == 0.0).all()
    assert res.equity.iloc[-1] == pytest.approx(1.0, abs=1e-12)
    assert len(res.trades) == 0
    assert res.metrics["total_return"] == pytest.approx(0.0, abs=1e-12)


def test_constant_prices():
    df = _make_df([100.0] * 10)
    target = pd.Series(1.0, index=df.index)
    res = run_backtest(df, target, cost_bps=1.0)
    # Only the single entry cost; no price moves.
    assert res.equity.iloc[-1] == pytest.approx(1.0 - 1.0 / 1e4, abs=1e-12)


def test_very_short_frame():
    df = _make_df([100.0, 101.0])
    target = pd.Series([1.0, 1.0], index=df.index)
    res = run_backtest(df, target, cost_bps=1.0)
    assert len(res.returns) == 2
    # Should not crash and metrics dict has expected keys.
    for key in ["sharpe", "max_drawdown", "n_trades", "exposure"]:
        assert key in res.metrics


def test_single_bar_frame():
    df = _make_df([100.0])
    target = pd.Series([1.0], index=df.index)
    res = run_backtest(df, target, cost_bps=1.0)
    assert len(res.returns) == 1
    assert "sharpe" in res.metrics


def test_short_frame_sltp_path():
    df = _make_df([100.0, 101.0])
    target = pd.Series([1.0, 1.0], index=df.index)
    # ATR undefined for such a short frame -> must not crash, just hold.
    res = run_backtest(df, target, cost_bps=1.0, sl_atr=2.0, tp_atr=4.0, atr_period=14)
    assert len(res.returns) == 2


def test_event_loop_equals_vectorized_when_levels_unreachable():
    """With SL/TP levels too far to ever trigger, the event-loop path must
    reproduce the vectorized path bar for bar (same boundary execution)."""
    rng = np.random.default_rng(7)
    n = 400
    close = 100.0 * np.exp(np.cumsum(rng.normal(0, 0.01, n)))
    df = _make_df(list(close))
    # A target that enters, exits, and flips sign repeatedly.
    raw = np.sign(np.sin(np.arange(n) / 7.0))
    raw[::13] = 0.0
    target = pd.Series(raw, index=df.index)

    fast = run_backtest(df, target, cost_bps=2.5)
    slow = run_backtest(df, target, cost_bps=2.5, sl_atr=1e9, tp_atr=1e9)

    pd.testing.assert_series_equal(fast.returns, slow.returns, atol=1e-12, rtol=0)
    pd.testing.assert_series_equal(fast.position, slow.position, atol=1e-12, rtol=0)
    assert fast.equity.iloc[-1] == pytest.approx(slow.equity.iloc[-1], abs=1e-12)
