"""Trend-following families: MA crosses, MACD, ribbons, SuperTrend, PSAR, etc.

All factories take a ``params`` dict and return a ``generate(df) -> Series``
producing a target position in [-1, 1] using only past/current-bar data.
"""

from __future__ import annotations

from typing import Any, Callable

import numpy as np
import pandas as pd

from ... import indicators as ind

GenerateFn = Callable[[pd.DataFrame], pd.Series]


def _cross_state(fast: pd.Series, slow: pd.Series) -> pd.Series:
    """+1 while fast>slow, -1 while fast<slow, ffill across equal bars."""
    raw = pd.Series(np.where(fast > slow, 1.0, np.where(fast < slow, -1.0, np.nan)),
                    index=fast.index)
    return raw.ffill()


def make_sma_cross(params: dict[str, Any]) -> GenerateFn:
    """SMA crossover: long when fast SMA above slow SMA, else short."""
    fast, slow = int(params["fast"]), int(params["slow"])

    def gen(df: pd.DataFrame) -> pd.Series:
        return _cross_state(ind.sma(df["close"], fast), ind.sma(df["close"], slow))

    return gen


def make_ema_cross(params: dict[str, Any]) -> GenerateFn:
    """EMA crossover: long when fast EMA above slow EMA, else short."""
    fast, slow = int(params["fast"]), int(params["slow"])

    def gen(df: pd.DataFrame) -> pd.Series:
        return _cross_state(ind.ema(df["close"], fast), ind.ema(df["close"], slow))

    return gen


def make_hull_ma_trend(params: dict[str, Any]) -> GenerateFn:
    """Hull MA trend: long when HMA rising, short when falling."""
    period = int(params["period"])

    def gen(df: pd.DataFrame) -> pd.Series:
        h = ind.hull(df["close"], period)
        slope = h.diff()
        return _cross_state(slope, slope * 0.0)

    return gen


def make_kama_trend(params: dict[str, Any]) -> GenerateFn:
    """KAMA trend: long when price above KAMA, short when below."""
    period = int(params["period"])

    def gen(df: pd.DataFrame) -> pd.Series:
        k = ind.kama(df["close"], period)
        return _cross_state(df["close"], k)

    return gen


def make_macd_signal(params: dict[str, Any]) -> GenerateFn:
    """MACD signal-line cross: long when MACD above its signal line."""
    fast, slow, signal = int(params["fast"]), int(params["slow"]), int(params["signal"])

    def gen(df: pd.DataFrame) -> pd.Series:
        macd_line, signal_line, _ = ind.macd(df["close"], fast, slow, signal)
        return _cross_state(macd_line, signal_line)

    return gen


def make_ma_ribbon(params: dict[str, Any]) -> GenerateFn:
    """MA ribbon: scaled long/short by fraction of EMAs in correct order."""
    base, step, count = int(params["base"]), int(params["step"]), int(params["count"])
    periods = [base + i * step for i in range(count)]

    def gen(df: pd.DataFrame) -> pd.Series:
        emas = [ind.ema(df["close"], p) for p in periods]
        # +1 each adjacent pair correctly ordered (faster above slower), -1 inverse
        ups = pd.Series(0.0, index=df.index)
        for i in range(count - 1):
            ups += np.where(emas[i] > emas[i + 1], 1.0, -1.0)
        return ups / float(count - 1)

    return gen


def make_supertrend(params: dict[str, Any]) -> GenerateFn:
    """SuperTrend: follow the ATR-band trend direction (+1/-1)."""
    period, mult = int(params["period"]), float(params["mult"])

    def gen(df: pd.DataFrame) -> pd.Series:
        return ind.supertrend(df, period, mult)

    return gen


def make_psar_trend(params: dict[str, Any]) -> GenerateFn:
    """Parabolic SAR: long/short following the SAR flip direction."""
    step, mx = float(params["step"]), float(params["max"])

    def gen(df: pd.DataFrame) -> pd.Series:
        return ind.psar(df, af_start=step, af_step=step, af_max=mx)

    return gen


def make_ichimoku(params: dict[str, Any]) -> GenerateFn:
    """Ichimoku: TK-cross (mode 'tk') or Kumo breakout (mode 'kumo')."""
    conv, base, span_b = int(params["conv"]), int(params["base"]), int(params["span_b"])
    mode = str(params.get("mode", "tk"))

    def gen(df: pd.DataFrame) -> pd.Series:
        tenkan, kijun, sa, sb = ind.ichimoku(df, conv, base, span_b)
        if mode == "kumo":
            top = pd.concat([sa, sb], axis=1).max(axis=1)
            bot = pd.concat([sa, sb], axis=1).min(axis=1)
            raw = pd.Series(
                np.where(df["close"] > top, 1.0,
                         np.where(df["close"] < bot, -1.0, np.nan)),
                index=df.index,
            )
            return raw.ffill()
        return _cross_state(tenkan, kijun)

    return gen


def make_adx_trend(params: dict[str, Any]) -> GenerateFn:
    """ADX-filtered DI cross: take DI direction only when ADX > threshold."""
    period, thr = int(params["period"]), float(params["threshold"])

    def gen(df: pd.DataFrame) -> pd.Series:
        plus_di, minus_di, adx = ind.dmi(df, period)
        direction = _cross_state(plus_di, minus_di)
        gated = direction.where(adx > thr, 0.0)
        return gated

    return gen


FACTORIES: dict[str, object] = {
    "sma_cross": make_sma_cross,
    "ema_cross": make_ema_cross,
    "hull_ma_trend": make_hull_ma_trend,
    "kama_trend": make_kama_trend,
    "macd_signal": make_macd_signal,
    "ma_ribbon": make_ma_ribbon,
    "supertrend": make_supertrend,
    "psar_trend": make_psar_trend,
    "ichimoku": make_ichimoku,
    "adx_trend": make_adx_trend,
}
