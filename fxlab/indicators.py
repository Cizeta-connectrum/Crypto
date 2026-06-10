"""Technical indicators as pure functions over OHLCV frames / price Series.

All functions return objects aligned to the input index and use only past
data (rolling/ewm), never future data.
"""

from __future__ import annotations

import pandas as pd


def atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    """Average True Range (Wilder's smoothing)."""
    prev_close = df["close"].shift(1)
    tr = pd.concat(
        [
            df["high"] - df["low"],
            (df["high"] - prev_close).abs(),
            (df["low"] - prev_close).abs(),
        ],
        axis=1,
    ).max(axis=1)
    return tr.ewm(alpha=1.0 / period, adjust=False, min_periods=period).mean()
