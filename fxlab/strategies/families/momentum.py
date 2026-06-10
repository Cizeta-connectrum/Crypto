"""Momentum families: ROC, dual-momentum, RSI momentum (with hysteresis)."""

from __future__ import annotations

from typing import Any, Callable

import numpy as np
import pandas as pd

from ... import indicators as ind

GenerateFn = Callable[[pd.DataFrame], pd.Series]


def make_roc_momentum(params: dict[str, Any]) -> GenerateFn:
    """Rate-of-change momentum: long when ROC>0, short when ROC<0."""
    period = int(params["period"])

    def gen(df: pd.DataFrame) -> pd.Series:
        r = ind.roc(df["close"], period)
        return pd.Series(
            np.where(r > 0.0, 1.0, np.where(r < 0.0, -1.0, np.nan)),
            index=df.index,
        ).ffill()

    return gen


def make_dual_momentum(params: dict[str, Any]) -> GenerateFn:
    """Dual-timeframe momentum: trade only when two ROC lookbacks agree."""
    fast, slow = int(params["fast"]), int(params["slow"])

    def gen(df: pd.DataFrame) -> pd.Series:
        rf = ind.roc(df["close"], fast)
        rs = ind.roc(df["close"], slow)
        long_ok = (rf > 0.0) & (rs > 0.0)
        short_ok = (rf < 0.0) & (rs < 0.0)
        return pd.Series(
            np.where(long_ok, 1.0, np.where(short_ok, -1.0, 0.0)),
            index=df.index,
        )

    return gen


def make_rsi_momentum(params: dict[str, Any]) -> GenerateFn:
    """RSI momentum with hysteresis: long > upper, short < lower, hold between."""
    period = int(params["period"])
    lower, upper = float(params["lower"]), float(params["upper"])

    def gen(df: pd.DataFrame) -> pd.Series:
        r = ind.rsi(df["close"], period)
        events = pd.Series(np.nan, index=df.index)
        events = events.mask(r > upper, 1.0)
        events = events.mask(r < lower, -1.0)
        return events.ffill()

    return gen


FACTORIES: dict[str, object] = {
    "roc_momentum": make_roc_momentum,
    "dual_momentum": make_dual_momentum,
    "rsi_momentum": make_rsi_momentum,
}
