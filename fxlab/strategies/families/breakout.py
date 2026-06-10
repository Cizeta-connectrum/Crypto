"""Breakout / channel families: Donchian, Bollinger, Keltner, ATR, N-bar high/low."""

from __future__ import annotations

from typing import Any, Callable

import numpy as np
import pandas as pd

from ... import indicators as ind

GenerateFn = Callable[[pd.DataFrame], pd.Series]


def _breakout_state(
    long_trig: pd.Series, short_trig: pd.Series
) -> pd.Series:
    """Position state machine via ffill of +1/-1 entry events."""
    raw = pd.Series(
        np.where(long_trig, 1.0, np.where(short_trig, -1.0, np.nan)),
        index=long_trig.index,
    )
    return raw.ffill()


def make_donchian_breakout(params: dict[str, Any]) -> GenerateFn:
    """Donchian breakout (turtle-style): enter on N-bar high/low break.

    Optional exit channel (``exit`` < entry) flattens before the opposite
    signal; without it the system is always in the market.
    """
    entry = int(params["entry"])
    exit_n = int(params.get("exit", 0))

    def gen(df: pd.DataFrame) -> pd.Series:
        _, upper, lower = ind.donchian(df, entry)
        long_in = df["close"] > upper
        short_in = df["close"] < lower
        if exit_n and exit_n > 0:
            ex_low = df["low"].rolling(exit_n, min_periods=exit_n).min().shift(1)
            ex_high = df["high"].rolling(exit_n, min_periods=exit_n).max().shift(1)
            long_exit = df["close"] < ex_low
            short_exit = df["close"] > ex_high
            events = pd.Series(np.nan, index=df.index)
            events = events.mask(long_in, 1.0)
            events = events.mask(short_in, -1.0)
            events = events.mask(long_exit & ~short_in, 0.0)
            events = events.mask(short_exit & ~long_in, 0.0)
            return events.ffill()
        return _breakout_state(long_in, short_in)

    return gen


def make_bollinger_breakout(params: dict[str, Any]) -> GenerateFn:
    """Bollinger breakout: long above upper band, short below lower band."""
    period, num_std = int(params["period"]), float(params["num_std"])

    def gen(df: pd.DataFrame) -> pd.Series:
        _, upper, lower = ind.bollinger(df["close"], period, num_std)
        return _breakout_state(df["close"] > upper, df["close"] < lower)

    return gen


def make_keltner_breakout(params: dict[str, Any]) -> GenerateFn:
    """Keltner breakout: long above upper channel, short below lower channel."""
    period, mult = int(params["period"]), float(params["mult"])

    def gen(df: pd.DataFrame) -> pd.Series:
        _, upper, lower = ind.keltner(df, period, mult, atr_period=period)
        return _breakout_state(df["close"] > upper, df["close"] < lower)

    return gen


def make_atr_volatility_breakout(params: dict[str, Any]) -> GenerateFn:
    """ATR volatility breakout: close beyond prev close +/- k*ATR."""
    period, k = int(params["period"]), float(params["k"])

    def gen(df: pd.DataFrame) -> pd.Series:
        rng = ind.atr(df, period).shift(1)
        prev_close = df["close"].shift(1)
        long_trig = df["close"] > prev_close + k * rng
        short_trig = df["close"] < prev_close - k * rng
        return _breakout_state(long_trig, short_trig)

    return gen


def make_rolling_high_low(params: dict[str, Any]) -> GenerateFn:
    """N-bar high/low momentum: long at new N-bar close high, short at low."""
    period = int(params["period"])

    def gen(df: pd.DataFrame) -> pd.Series:
        hh = df["close"].rolling(period, min_periods=period).max().shift(1)
        ll = df["close"].rolling(period, min_periods=period).min().shift(1)
        return _breakout_state(df["close"] >= hh, df["close"] <= ll)

    return gen


FACTORIES: dict[str, object] = {
    "donchian_breakout": make_donchian_breakout,
    "bollinger_breakout": make_bollinger_breakout,
    "keltner_breakout": make_keltner_breakout,
    "atr_volatility_breakout": make_atr_volatility_breakout,
    "rolling_high_low": make_rolling_high_low,
}
