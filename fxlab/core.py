"""Core types and contracts shared by all FXLab modules.

This module is the frozen interface between the data layer, the strategy
library, the backtest engine and the UIs. Do not change signatures without
updating docs/DESIGN.md and every consumer.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

OHLCV_COLUMNS = ("open", "high", "low", "close", "volume")


class DataSourceError(RuntimeError):
    """Raised when a market-data source is unavailable or returns bad data."""


def validate_ohlcv(df: pd.DataFrame) -> pd.DataFrame:
    """Validate (and lightly normalize) an OHLCV frame; raise ValueError if invalid.

    Requirements: DatetimeIndex (tz-naive, strictly increasing, unique),
    float columns open/high/low/close/volume, no NaN in OHLC, high>=low.
    Returns the validated frame (columns ordered, dtypes coerced to float64).
    """
    if not isinstance(df.index, pd.DatetimeIndex):
        raise ValueError("OHLCV index must be a DatetimeIndex")
    if df.index.tz is not None:
        df = df.tz_localize(None)
    if not df.index.is_monotonic_increasing:
        raise ValueError("OHLCV index must be ascending")
    if df.index.has_duplicates:
        raise ValueError("OHLCV index must be unique")
    missing = [c for c in OHLCV_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"OHLCV frame missing columns: {missing}")
    out = df.loc[:, list(OHLCV_COLUMNS)].astype("float64")
    ohlc = out[["open", "high", "low", "close"]]
    if ohlc.isna().any().any():
        raise ValueError("OHLC columns contain NaN")
    if (out["high"] < out["low"]).any():
        raise ValueError("high < low encountered")
    return out


class Strategy(ABC):
    """A parameterized trading strategy.

    ``generate`` returns the *target position* in [-1, 1] for each bar,
    computed with information up to and including that bar's close. The
    engine shifts by one bar before applying returns — implementations must
    not shift, and must never use future data.
    """

    id: str
    family: str
    params: dict[str, Any]

    def __init__(self, id: str, family: str, params: dict[str, Any]):
        self.id = id
        self.family = family
        self.params = dict(params)

    @abstractmethod
    def generate(self, df: pd.DataFrame) -> pd.Series:
        """Return target position in [-1, 1], aligned to df.index."""

    def __repr__(self) -> str:  # pragma: no cover
        return f"<Strategy {self.id}>"


def clean_position(pos: pd.Series, index: pd.Index) -> pd.Series:
    """Align, fill and clip a raw position series to the engine's expectations."""
    pos = pos.reindex(index).astype("float64")
    pos = pos.replace([np.inf, -np.inf], np.nan).ffill().fillna(0.0)
    return pos.clip(-1.0, 1.0)


@dataclass
class BacktestResult:
    """Output of ``fxlab.engine.run_backtest``."""

    equity: pd.Series  # compounded, starts at 1.0
    returns: pd.Series  # per-bar simple returns net of costs
    position: pd.Series  # effective position actually held (post-shift)
    trades: pd.DataFrame  # entry_time, exit_time, direction, bars, ret
    metrics: dict[str, float] = field(default_factory=dict)
