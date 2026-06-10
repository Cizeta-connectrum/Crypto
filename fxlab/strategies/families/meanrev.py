"""Mean-reversion families: RSI(2/Connors), Bollinger fade, z-score, %R, CCI,
stochastic, double-seven."""

from __future__ import annotations

from typing import Any, Callable

import numpy as np
import pandas as pd

from ... import indicators as ind

GenerateFn = Callable[[pd.DataFrame], pd.Series]


def _band_state(
    enter_long: pd.Series,
    enter_short: pd.Series,
    exit_long: pd.Series,
    exit_short: pd.Series,
) -> pd.Series:
    """Event-driven state: enter on extreme, exit to flat at the midline."""
    events = pd.Series(np.nan, index=enter_long.index)
    events = events.mask(exit_long, 0.0)
    events = events.mask(exit_short, 0.0)
    events = events.mask(enter_long, 1.0)
    events = events.mask(enter_short, -1.0)
    return events.ffill()


def make_rsi_meanrev(params: dict[str, Any]) -> GenerateFn:
    """RSI mean reversion (incl. Connors RSI(2)): buy oversold, sell overbought."""
    period = int(params["period"])
    lower, upper = float(params["lower"]), float(params["upper"])
    mid = (lower + upper) / 2.0

    def gen(df: pd.DataFrame) -> pd.Series:
        r = ind.rsi(df["close"], period)
        return _band_state(
            enter_long=r < lower,
            enter_short=r > upper,
            exit_long=r > mid,
            exit_short=r < mid,
        )

    return gen


def make_bollinger_fade(params: dict[str, Any]) -> GenerateFn:
    """Bollinger fade: short at upper band, long at lower band, exit at mid."""
    period, num_std = int(params["period"]), float(params["num_std"])

    def gen(df: pd.DataFrame) -> pd.Series:
        mid, upper, lower = ind.bollinger(df["close"], period, num_std)
        return _band_state(
            enter_long=df["close"] < lower,
            enter_short=df["close"] > upper,
            exit_long=df["close"] > mid,
            exit_short=df["close"] < mid,
        )

    return gen


def make_zscore_reversion(params: dict[str, Any]) -> GenerateFn:
    """Z-score reversion: fade rolling z-score extremes back to the mean."""
    period, thr = int(params["period"]), float(params["threshold"])

    def gen(df: pd.DataFrame) -> pd.Series:
        z = ind.zscore(df["close"], period)
        return _band_state(
            enter_long=z < -thr,
            enter_short=z > thr,
            exit_long=z > 0.0,
            exit_short=z < 0.0,
        )

    return gen


def make_williams_r(params: dict[str, Any]) -> GenerateFn:
    """Williams %R reversion: buy below -80, sell above -20."""
    period = int(params["period"])
    lower, upper = float(params["lower"]), float(params["upper"])
    mid = (lower + upper) / 2.0

    def gen(df: pd.DataFrame) -> pd.Series:
        wr = ind.williams_r(df, period)
        return _band_state(
            enter_long=wr < lower,
            enter_short=wr > upper,
            exit_long=wr > mid,
            exit_short=wr < mid,
        )

    return gen


def make_cci_reversion(params: dict[str, Any]) -> GenerateFn:
    """CCI reversion: buy below -threshold, sell above +threshold, exit at 0."""
    period, thr = int(params["period"]), float(params["threshold"])

    def gen(df: pd.DataFrame) -> pd.Series:
        c = ind.cci(df, period)
        return _band_state(
            enter_long=c < -thr,
            enter_short=c > thr,
            exit_long=c > 0.0,
            exit_short=c < 0.0,
        )

    return gen


def make_stochastic_reversion(params: dict[str, Any]) -> GenerateFn:
    """Stochastic reversion: buy when %K oversold, sell when overbought."""
    k_period, d_period = int(params["k_period"]), int(params["d_period"])
    lower, upper = float(params["lower"]), float(params["upper"])

    def gen(df: pd.DataFrame) -> pd.Series:
        k, d = ind.stochastic(df, k_period, d_period)
        return _band_state(
            enter_long=d < lower,
            enter_short=d > upper,
            exit_long=d > 50.0,
            exit_short=d < 50.0,
        )

    return gen


def make_double_seven(params: dict[str, Any]) -> GenerateFn:
    """Double-seven: in an uptrend (above long MA), buy N-bar low, sell N-bar high."""
    ma_period, n = int(params["ma_period"]), int(params["n"])

    def gen(df: pd.DataFrame) -> pd.Series:
        ma = ind.sma(df["close"], ma_period)
        uptrend = df["close"] > ma
        low_n = df["close"].rolling(n, min_periods=n).min()
        high_n = df["close"].rolling(n, min_periods=n).max()
        enter_long = uptrend & (df["close"] <= low_n)
        exit_long = df["close"] >= high_n
        events = pd.Series(np.nan, index=df.index)
        events = events.mask(exit_long, 0.0)
        events = events.mask(enter_long, 1.0)
        events = events.mask(~uptrend, 0.0)
        return events.ffill()

    return gen


FACTORIES: dict[str, object] = {
    "rsi_meanrev": make_rsi_meanrev,
    "bollinger_fade": make_bollinger_fade,
    "zscore_reversion": make_zscore_reversion,
    "williams_r": make_williams_r,
    "cci_reversion": make_cci_reversion,
    "stochastic_reversion": make_stochastic_reversion,
    "double_seven": make_double_seven,
}
