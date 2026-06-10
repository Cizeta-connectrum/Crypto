"""Seasonal families: day-of-week and turn-of-month effects."""

from __future__ import annotations

from typing import Any, Callable

import numpy as np
import pandas as pd

GenerateFn = Callable[[pd.DataFrame], pd.Series]


def make_day_of_week(params: dict[str, Any]) -> GenerateFn:
    """Day-of-week: hold long on a chosen weekday (0=Mon .. 6=Sun)."""
    weekday = int(params["weekday"])

    def gen(df: pd.DataFrame) -> pd.Series:
        dow = df.index.dayofweek
        return pd.Series(np.where(dow == weekday, 1.0, 0.0), index=df.index)

    return gen


def make_turn_of_month(params: dict[str, Any]) -> GenerateFn:
    """Turn-of-month: long during the last/first ``n`` trading days of months.

    Causal: the first ``n`` trading days are counted from the current month's
    start; the "last days" window uses the *calendar* days remaining in the
    month (known from the date alone, no future bars required).
    """
    n = int(params["n"])

    def gen(df: pd.DataFrame) -> pd.Series:
        idx = df.index
        ym = pd.Series(idx.year * 100 + idx.month, index=idx)
        pos_in_month = ym.groupby(ym).cumcount()  # causal: count from start
        days_in_month = idx.days_in_month
        cal_from_end = days_in_month - idx.day  # calendar days left this month
        first_window = pos_in_month < n
        last_window = pd.Series(cal_from_end < n, index=idx)
        long_window = first_window | last_window
        return pd.Series(np.where(long_window, 1.0, 0.0), index=idx)

    return gen


FACTORIES: dict[str, object] = {
    "day_of_week": make_day_of_week,
    "turn_of_month": make_turn_of_month,
}
