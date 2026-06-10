"""FXLab backtest engine: vectorized & event-loop execution plus metrics."""

from __future__ import annotations

from .backtest import run_backtest
from .metrics import compute_metrics, infer_periods_per_year, max_drawdown

__all__ = [
    "run_backtest",
    "compute_metrics",
    "infer_periods_per_year",
    "max_drawdown",
]
