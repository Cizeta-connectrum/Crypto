"""Price-action families: engulfing, Heikin-Ashi trend, inside-bar breakout."""

from __future__ import annotations

from typing import Any, Callable

import numpy as np
import pandas as pd

from ... import indicators as ind

GenerateFn = Callable[[pd.DataFrame], pd.Series]


def make_engulfing_pattern(params: dict[str, Any]) -> GenerateFn:
    """Engulfing reversal filtered by trend: trade engulfing candles with the MA."""
    ma_period = int(params["ma_period"])
    hold = int(params.get("hold", 3))

    def gen(df: pd.DataFrame) -> pd.Series:
        o, c = df["open"], df["close"]
        po, pc = o.shift(1), c.shift(1)
        bull = (c > o) & (pc < po) & (c >= po) & (o <= pc)
        bear = (c < o) & (pc > po) & (c <= po) & (o >= pc)
        ma = ind.sma(c, ma_period)
        long_sig = bull & (c > ma)
        short_sig = bear & (c < ma)
        events = pd.Series(np.nan, index=df.index)
        events = events.mask(long_sig, 1.0)
        events = events.mask(short_sig, -1.0)
        state = events.ffill()
        # decay to flat after `hold` bars with no new signal
        bars_since = _bars_since(long_sig | short_sig)
        state = state.where(bars_since <= hold, 0.0)
        return state

    return gen


def _bars_since(event: pd.Series) -> pd.Series:
    """Number of bars since the last True in ``event`` (NaN until first event)."""
    idx = np.arange(len(event))
    last = pd.Series(np.where(event.to_numpy(), idx, np.nan),
                     index=event.index).ffill()
    return pd.Series(idx, index=event.index) - last


def make_heikin_ashi_trend(params: dict[str, Any]) -> GenerateFn:
    """Heikin-Ashi trend: long on N consecutive HA-up bars, short on HA-down."""
    streak = int(params["streak"])

    def gen(df: pd.DataFrame) -> pd.Series:
        ha = ind.heikin_ashi(df)
        up = ha["close"] > ha["open"]
        down = ha["close"] < ha["open"]
        up_streak = up.rolling(streak, min_periods=streak).sum() == streak
        down_streak = down.rolling(streak, min_periods=streak).sum() == streak
        events = pd.Series(np.nan, index=df.index)
        events = events.mask(up_streak, 1.0)
        events = events.mask(down_streak, -1.0)
        return events.ffill()

    return gen


def make_inside_bar_breakout(params: dict[str, Any]) -> GenerateFn:
    """Inside-bar breakout: after an inside bar, go with the breakout direction."""
    hold = int(params["hold"])

    def gen(df: pd.DataFrame) -> pd.Series:
        h, low = df["high"], df["low"]
        inside = (h < h.shift(1)) & (low > low.shift(1))
        # mother bar = the bar before the inside bar
        mother_high = h.shift(1).where(inside)
        mother_low = low.shift(1).where(inside)
        mh = mother_high.ffill()
        ml = mother_low.ffill()
        bars_since_inside = _bars_since(inside)
        active = bars_since_inside <= hold
        long_brk = active & (df["close"] > mh)
        short_brk = active & (df["close"] < ml)
        events = pd.Series(np.nan, index=df.index)
        events = events.mask(long_brk, 1.0)
        events = events.mask(short_brk, -1.0)
        state = events.ffill()
        return state.where(bars_since_inside <= (hold + 2), 0.0)

    return gen


FACTORIES: dict[str, object] = {
    "engulfing_pattern": make_engulfing_pattern,
    "heikin_ashi_trend": make_heikin_ashi_trend,
    "inside_bar_breakout": make_inside_bar_breakout,
}
