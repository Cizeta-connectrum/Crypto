"""Oscillator families: TRIX signal, Aroon cross, CMO momentum."""

from __future__ import annotations

from typing import Any, Callable

import numpy as np
import pandas as pd

from ... import indicators as ind

GenerateFn = Callable[[pd.DataFrame], pd.Series]


def make_trix_signal(params: dict[str, Any]) -> GenerateFn:
    """TRIX zero-line cross: long when TRIX>0, short when TRIX<0."""
    period = int(params["period"])

    def gen(df: pd.DataFrame) -> pd.Series:
        t = ind.trix(df["close"], period)
        return pd.Series(
            np.where(t > 0.0, 1.0, np.where(t < 0.0, -1.0, np.nan)),
            index=df.index,
        ).ffill()

    return gen


def make_aroon_cross(params: dict[str, Any]) -> GenerateFn:
    """Aroon cross: long when Aroon-Up above Aroon-Down, else short."""
    period = int(params["period"])

    def gen(df: pd.DataFrame) -> pd.Series:
        up, down = ind.aroon(df, period)
        return pd.Series(
            np.where(up > down, 1.0, np.where(up < down, -1.0, np.nan)),
            index=df.index,
        ).ffill()

    return gen


def make_cmo_momentum(params: dict[str, Any]) -> GenerateFn:
    """Chande Momentum Oscillator: long above +threshold, short below -threshold."""
    period, thr = int(params["period"]), float(params["threshold"])

    def gen(df: pd.DataFrame) -> pd.Series:
        c = ind.cmo(df["close"], period)
        events = pd.Series(np.nan, index=df.index)
        events = events.mask(c > thr, 1.0)
        events = events.mask(c < -thr, -1.0)
        return events.ffill()

    return gen


FACTORIES: dict[str, object] = {
    "trix_signal": make_trix_signal,
    "aroon_cross": make_aroon_cross,
    "cmo_momentum": make_cmo_momentum,
}
