"""Combined-filter families: trend+pullback, Bollinger trend filter, MACD+ADX."""

from __future__ import annotations

from typing import Any, Callable

import numpy as np
import pandas as pd

from ... import indicators as ind

GenerateFn = Callable[[pd.DataFrame], pd.Series]


def make_trend_pullback(params: dict[str, Any]) -> GenerateFn:
    """Trend pullback: long-term MA up + short-term RSI oversold buy (symmetric)."""
    ma_period, rsi_period = int(params["ma_period"]), int(params["rsi_period"])
    lower, upper = float(params["lower"]), float(params["upper"])

    def gen(df: pd.DataFrame) -> pd.Series:
        ma = ind.sma(df["close"], ma_period)
        r = ind.rsi(df["close"], rsi_period)
        up = df["close"] > ma
        down = df["close"] < ma
        enter_long = up & (r < lower)
        exit_long = (r > 50.0) | down
        enter_short = down & (r > upper)
        exit_short = (r < 50.0) | up
        events = pd.Series(np.nan, index=df.index)
        events = events.mask(exit_long, 0.0)
        events = events.mask(exit_short, 0.0)
        events = events.mask(enter_long, 1.0)
        events = events.mask(enter_short, -1.0)
        return events.ffill()

    return gen


def make_bollinger_trend_filter(params: dict[str, Any]) -> GenerateFn:
    """Bollinger band-ride filtered by trend: ride breakouts only with the MA."""
    period, num_std = int(params["period"]), float(params["num_std"])
    ma_period = int(params["ma_period"])

    def gen(df: pd.DataFrame) -> pd.Series:
        mid, upper, lower = ind.bollinger(df["close"], period, num_std)
        ma = ind.sma(df["close"], ma_period)
        long_in = (df["close"] > upper) & (df["close"] > ma)
        short_in = (df["close"] < lower) & (df["close"] < ma)
        long_exit = df["close"] < mid
        short_exit = df["close"] > mid
        events = pd.Series(np.nan, index=df.index)
        events = events.mask(long_exit, 0.0)
        events = events.mask(short_exit, 0.0)
        events = events.mask(long_in, 1.0)
        events = events.mask(short_in, -1.0)
        return events.ffill()

    return gen


def make_macd_adx_combo(params: dict[str, Any]) -> GenerateFn:
    """MACD direction confirmed by ADX strength: trend only when ADX>threshold."""
    fast, slow, signal = int(params["fast"]), int(params["slow"]), int(params["signal"])
    adx_period, thr = int(params["adx_period"]), float(params["threshold"])

    def gen(df: pd.DataFrame) -> pd.Series:
        macd_line, signal_line, _ = ind.macd(df["close"], fast, slow, signal)
        _, _, adx = ind.dmi(df, adx_period)
        direction = pd.Series(
            np.where(macd_line > signal_line, 1.0,
                     np.where(macd_line < signal_line, -1.0, np.nan)),
            index=df.index,
        ).ffill()
        return direction.where(adx > thr, 0.0)

    return gen


FACTORIES: dict[str, object] = {
    "trend_pullback": make_trend_pullback,
    "bollinger_trend_filter": make_bollinger_trend_filter,
    "macd_adx_combo": make_macd_adx_combo,
}
