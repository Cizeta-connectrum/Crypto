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


def sma(s: pd.Series, period: int) -> pd.Series:
    """Simple moving average."""
    return s.rolling(period, min_periods=period).mean()


def ema(s: pd.Series, period: int) -> pd.Series:
    """Exponential moving average (standard span convention)."""
    return s.ewm(span=period, adjust=False, min_periods=period).mean()


def wilder_ema(s: pd.Series, period: int) -> pd.Series:
    """Wilder's smoothing (RMA / modified EMA with alpha=1/period)."""
    return s.ewm(alpha=1.0 / period, adjust=False, min_periods=period).mean()


def wma(s: pd.Series, period: int) -> pd.Series:
    """Weighted moving average (linear weights 1..period)."""
    import numpy as np

    weights = np.arange(1, period + 1, dtype="float64")

    def _w(x: "np.ndarray") -> float:
        return float((x * weights).sum() / weights.sum())

    return s.rolling(period, min_periods=period).apply(_w, raw=True)


def hull(s: pd.Series, period: int) -> pd.Series:
    """Hull Moving Average: WMA(2*WMA(n/2) - WMA(n)) over sqrt(n)."""
    import numpy as np

    half = max(1, int(period // 2))
    sqrt_n = max(1, int(round(np.sqrt(period))))
    raw = 2.0 * wma(s, half) - wma(s, period)
    return wma(raw, sqrt_n)


def kama(s: pd.Series, period: int = 10, fast: int = 2, slow: int = 30) -> pd.Series:
    """Kaufman Adaptive Moving Average."""
    import numpy as np

    change = (s - s.shift(period)).abs()
    volatility = s.diff().abs().rolling(period, min_periods=period).sum()
    er = (change / volatility).replace([np.inf, -np.inf], np.nan).fillna(0.0)
    fast_sc = 2.0 / (fast + 1.0)
    slow_sc = 2.0 / (slow + 1.0)
    sc = (er * (fast_sc - slow_sc) + slow_sc) ** 2

    values = s.to_numpy(dtype="float64")
    sc_arr = sc.to_numpy(dtype="float64")
    out = np.full(len(values), np.nan, dtype="float64")
    # seed at first index where we have a full period window
    seed = period
    if seed < len(values):
        out[seed] = values[seed]
        for i in range(seed + 1, len(values)):
            prev = out[i - 1]
            out[i] = prev + sc_arr[i] * (values[i] - prev)
    return pd.Series(out, index=s.index)


def rsi(s: pd.Series, period: int = 14) -> pd.Series:
    """Relative Strength Index (Wilder smoothing)."""
    delta = s.diff()
    gain = delta.clip(lower=0.0)
    loss = -delta.clip(upper=0.0)
    avg_gain = wilder_ema(gain, period)
    avg_loss = wilder_ema(loss, period)
    rs = avg_gain / avg_loss.replace(0.0, float("nan"))
    out = 100.0 - 100.0 / (1.0 + rs)
    # when avg_loss == 0 -> RSI 100; when avg_gain==0 -> 0
    out = out.where(avg_loss != 0.0, 100.0)
    out = out.where(~((avg_gain == 0.0) & (avg_loss == 0.0)), 50.0)
    return out


def macd(
    s: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9
) -> tuple[pd.Series, pd.Series, pd.Series]:
    """MACD line, signal line, histogram."""
    macd_line = ema(s, fast) - ema(s, slow)
    signal_line = macd_line.ewm(span=signal, adjust=False, min_periods=signal).mean()
    hist = macd_line - signal_line
    return macd_line, signal_line, hist


def bollinger(
    s: pd.Series, period: int = 20, num_std: float = 2.0
) -> tuple[pd.Series, pd.Series, pd.Series]:
    """Bollinger Bands: (middle, upper, lower)."""
    mid = sma(s, period)
    sd = s.rolling(period, min_periods=period).std(ddof=0)
    return mid, mid + num_std * sd, mid - num_std * sd


def keltner(
    df: pd.DataFrame, period: int = 20, mult: float = 2.0, atr_period: int = 10
) -> tuple[pd.Series, pd.Series, pd.Series]:
    """Keltner Channels: EMA mid +/- mult*ATR."""
    mid = ema(df["close"], period)
    rng = atr(df, atr_period)
    return mid, mid + mult * rng, mid - mult * rng


def donchian(
    df: pd.DataFrame, period: int = 20
) -> tuple[pd.Series, pd.Series, pd.Series]:
    """Donchian channel using only completed prior bars (shifted by 1)."""
    upper = df["high"].rolling(period, min_periods=period).max().shift(1)
    lower = df["low"].rolling(period, min_periods=period).min().shift(1)
    mid = (upper + lower) / 2.0
    return mid, upper, lower


def stochastic(
    df: pd.DataFrame, k_period: int = 14, d_period: int = 3
) -> tuple[pd.Series, pd.Series]:
    """Stochastic oscillator %K and %D."""
    low_min = df["low"].rolling(k_period, min_periods=k_period).min()
    high_max = df["high"].rolling(k_period, min_periods=k_period).max()
    rng = (high_max - low_min).replace(0.0, float("nan"))
    k = 100.0 * (df["close"] - low_min) / rng
    k = k.fillna(50.0)
    d = k.rolling(d_period, min_periods=d_period).mean()
    return k, d


def williams_r(df: pd.DataFrame, period: int = 14) -> pd.Series:
    """Williams %R in [-100, 0]."""
    high_max = df["high"].rolling(period, min_periods=period).max()
    low_min = df["low"].rolling(period, min_periods=period).min()
    rng = (high_max - low_min).replace(0.0, float("nan"))
    out = -100.0 * (high_max - df["close"]) / rng
    return out.fillna(-50.0)


def cci(df: pd.DataFrame, period: int = 20) -> pd.Series:
    """Commodity Channel Index."""
    tp = (df["high"] + df["low"] + df["close"]) / 3.0
    ma = sma(tp, period)
    md = (tp - ma).abs().rolling(period, min_periods=period).mean()
    return (tp - ma) / (0.015 * md.replace(0.0, float("nan")))


def roc(s: pd.Series, period: int = 12) -> pd.Series:
    """Rate of change in percent."""
    return 100.0 * (s / s.shift(period) - 1.0)


def cmo(s: pd.Series, period: int = 14) -> pd.Series:
    """Chande Momentum Oscillator in [-100, 100]."""
    delta = s.diff()
    up = delta.clip(lower=0.0).rolling(period, min_periods=period).sum()
    down = (-delta.clip(upper=0.0)).rolling(period, min_periods=period).sum()
    denom = (up + down).replace(0.0, float("nan"))
    return (100.0 * (up - down) / denom).fillna(0.0)


def trix(s: pd.Series, period: int = 15) -> pd.Series:
    """TRIX: 1-bar % change of triple-smoothed EMA (in percent)."""
    e1 = ema(s, period)
    e2 = ema(e1, period)
    e3 = ema(e2, period)
    return 100.0 * (e3 / e3.shift(1) - 1.0)


def aroon(df: pd.DataFrame, period: int = 25) -> tuple[pd.Series, pd.Series]:
    """Aroon Up and Aroon Down."""
    def _since_high(x: "object") -> float:
        import numpy as np

        return float(len(x) - 1 - np.argmax(x))

    def _since_low(x: "object") -> float:
        import numpy as np

        return float(len(x) - 1 - np.argmin(x))

    win = period + 1
    high_idx = df["high"].rolling(win, min_periods=win).apply(_since_high, raw=True)
    low_idx = df["low"].rolling(win, min_periods=win).apply(_since_low, raw=True)
    up = 100.0 * (period - high_idx) / period
    down = 100.0 * (period - low_idx) / period
    return up, down


def dmi(
    df: pd.DataFrame, period: int = 14
) -> tuple[pd.Series, pd.Series, pd.Series]:
    """Directional Movement Index: (+DI, -DI, ADX) with Wilder smoothing."""
    import numpy as np

    up_move = df["high"].diff()
    down_move = -df["low"].diff()
    plus_dm = pd.Series(
        np.where((up_move > down_move) & (up_move > 0.0), up_move, 0.0),
        index=df.index,
    )
    minus_dm = pd.Series(
        np.where((down_move > up_move) & (down_move > 0.0), down_move, 0.0),
        index=df.index,
    )
    tr = atr(df, period)  # already Wilder-smoothed TR average
    plus_di = 100.0 * wilder_ema(plus_dm, period) / tr.replace(0.0, float("nan"))
    minus_di = 100.0 * wilder_ema(minus_dm, period) / tr.replace(0.0, float("nan"))
    dx = 100.0 * (plus_di - minus_di).abs() / (
        (plus_di + minus_di).replace(0.0, float("nan"))
    )
    adx = wilder_ema(dx.fillna(0.0), period)
    return plus_di, minus_di, adx


def zscore(s: pd.Series, period: int = 20) -> pd.Series:
    """Rolling z-score of a series."""
    mean = s.rolling(period, min_periods=period).mean()
    sd = s.rolling(period, min_periods=period).std(ddof=0)
    return (s - mean) / sd.replace(0.0, float("nan"))


def heikin_ashi(df: pd.DataFrame) -> pd.DataFrame:
    """Heikin-Ashi OHLC frame (causal: HA-open uses prior HA bar only)."""
    import numpy as np

    ha_close = (df["open"] + df["high"] + df["low"] + df["close"]) / 4.0
    open_v = df["open"].to_numpy(dtype="float64")
    close_v = ha_close.to_numpy(dtype="float64")
    ha_open = np.full(len(df), np.nan, dtype="float64")
    if len(df):
        ha_open[0] = (open_v[0] + close_v[0]) / 2.0
        for i in range(1, len(df)):
            ha_open[i] = (ha_open[i - 1] + close_v[i - 1]) / 2.0
    ha_open_s = pd.Series(ha_open, index=df.index)
    ha_high = pd.concat([df["high"], ha_open_s, ha_close], axis=1).max(axis=1)
    ha_low = pd.concat([df["low"], ha_open_s, ha_close], axis=1).min(axis=1)
    return pd.DataFrame(
        {"open": ha_open_s, "high": ha_high, "low": ha_low, "close": ha_close}
    )


def supertrend(df: pd.DataFrame, period: int = 10, mult: float = 3.0) -> pd.Series:
    """SuperTrend direction: +1 (up) / -1 (down), causal numpy loop."""
    import numpy as np

    hl2 = (df["high"] + df["low"]) / 2.0
    rng = atr(df, period)
    upper = (hl2 + mult * rng).to_numpy(dtype="float64")
    lower = (hl2 - mult * rng).to_numpy(dtype="float64")
    close = df["close"].to_numpy(dtype="float64")
    n = len(df)
    final_upper = np.full(n, np.nan)
    final_lower = np.full(n, np.nan)
    direction = np.ones(n, dtype="float64")
    for i in range(n):
        if i == 0 or np.isnan(upper[i]) or np.isnan(lower[i]):
            final_upper[i] = upper[i]
            final_lower[i] = lower[i]
            direction[i] = 1.0
            continue
        fu_prev = final_upper[i - 1]
        fl_prev = final_lower[i - 1]
        if np.isnan(fu_prev):
            final_upper[i] = upper[i]
            final_lower[i] = lower[i]
            direction[i] = 1.0 if close[i] >= lower[i] else -1.0
            continue
        final_upper[i] = (
            upper[i] if (upper[i] < fu_prev or close[i - 1] > fu_prev) else fu_prev
        )
        final_lower[i] = (
            lower[i] if (lower[i] > fl_prev or close[i - 1] < fl_prev) else fl_prev
        )
        prev_dir = direction[i - 1]
        if prev_dir == 1.0:
            direction[i] = -1.0 if close[i] < final_lower[i] else 1.0
        else:
            direction[i] = 1.0 if close[i] > final_upper[i] else -1.0
    out = pd.Series(direction, index=df.index)
    warmup = rng.isna()
    return out.where(~warmup, other=float("nan"))


def psar(
    df: pd.DataFrame, af_start: float = 0.02, af_step: float = 0.02, af_max: float = 0.2
) -> pd.Series:
    """Parabolic SAR trend direction: +1 long / -1 short, causal numpy loop."""
    import numpy as np

    high = df["high"].to_numpy(dtype="float64")
    low = df["low"].to_numpy(dtype="float64")
    n = len(df)
    direction = np.ones(n, dtype="float64")
    if n == 0:
        return pd.Series(direction, index=df.index)
    sar = np.zeros(n, dtype="float64")
    ep = high[0]
    af = af_start
    trend = 1  # start long
    sar[0] = low[0]
    for i in range(1, n):
        prev_sar = sar[i - 1]
        sar_i = prev_sar + af * (ep - prev_sar)
        if trend == 1:
            sar_i = min(sar_i, low[i - 1], low[i - 2] if i >= 2 else low[i - 1])
            if low[i] < sar_i:
                trend = -1
                sar_i = ep
                ep = low[i]
                af = af_start
            else:
                if high[i] > ep:
                    ep = high[i]
                    af = min(af + af_step, af_max)
        else:
            sar_i = max(sar_i, high[i - 1], high[i - 2] if i >= 2 else high[i - 1])
            if high[i] > sar_i:
                trend = 1
                sar_i = ep
                ep = high[i]
                af = af_start
            else:
                if low[i] < ep:
                    ep = low[i]
                    af = min(af + af_step, af_max)
        sar[i] = sar_i
        direction[i] = float(trend)
    return pd.Series(direction, index=df.index)


def ichimoku(
    df: pd.DataFrame, conv: int = 9, base: int = 26, span_b: int = 52
) -> tuple[pd.Series, pd.Series, pd.Series, pd.Series]:
    """Ichimoku: (tenkan, kijun, senkou_a, senkou_b) — spans NOT shifted forward.

    Spans are returned aligned to the current bar (no future displacement) so
    that comparisons remain causal.
    """
    def _mid(period: int) -> pd.Series:
        hh = df["high"].rolling(period, min_periods=period).max()
        ll = df["low"].rolling(period, min_periods=period).min()
        return (hh + ll) / 2.0

    tenkan = _mid(conv)
    kijun = _mid(base)
    senkou_a = (tenkan + kijun) / 2.0
    senkou_b = _mid(span_b)
    return tenkan, kijun, senkou_a, senkou_b
