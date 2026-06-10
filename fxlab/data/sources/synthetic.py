"""Synthetic OHLCV generator using regime-switching geometric Brownian motion.

Fully offline and deterministic: the random seed is derived from the
symbol + timeframe string when *seed* is not supplied explicitly.
"""

from __future__ import annotations

import hashlib
import logging
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

from fxlab.core import DataSourceError, validate_ohlcv

if TYPE_CHECKING:
    pass

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Per-asset-class annualised volatility (sigma)
# ---------------------------------------------------------------------------

_CLASS_SIGMA: dict[str, float] = {
    "fx": 0.08,
    "metal": 0.15,
    "crypto": 0.70,
}

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _derive_seed(symbol: str, timeframe: str) -> int:
    """Return a deterministic 32-bit seed from *symbol* + *timeframe*."""
    h = hashlib.sha256(f"{symbol}:{timeframe}".encode()).digest()
    return int.from_bytes(h[:4], "big")


def _bars_per_year(timeframe: str) -> float:
    """Map a timeframe string to the approximate number of bars per calendar year."""
    tf = timeframe.lower()
    mapping = {
        "1d": 252.0,
        "4h": 252.0 * 6,
        "1h": 252.0 * 24,
        "30m": 252.0 * 48,
        "15m": 252.0 * 96,
        "5m": 252.0 * 288,
        "1m": 252.0 * 1440,
        "1w": 52.0,
    }
    return mapping.get(tf, 252.0)


def _timeframe_freq(timeframe: str) -> str:
    """Return a pandas frequency string for *timeframe*."""
    tf = timeframe.lower()
    mapping = {
        "1d": "B",      # business days
        "4h": "4h",
        "1h": "h",
        "30m": "30min",
        "15m": "15min",
        "5m": "5min",
        "1m": "1min",
        "1w": "W-FRI",
    }
    return mapping.get(tf, "B")


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def generate(
    symbol: str,
    timeframe: str = "1d",
    start: str = "2015-01-01",
    end: str | None = None,
    seed: int | None = None,
    asset_class: str | None = None,
) -> pd.DataFrame:
    """Generate a synthetic OHLCV frame for *symbol*.

    The simulation uses a two-state regime-switching GBM:

    * **Noise regime** — zero drift, vol = per-class sigma.
    * **Trend regime** — non-zero drift ±μ, slightly elevated vol.

    Regime switches follow a Markov chain with a low transition probability.
    OHLC bars are constructed around the close path so that
    ``high ≥ max(open, close) ≥ min(open, close) ≥ low``.

    Volume is random for metals/crypto, zero for FX.

    Parameters
    ----------
    symbol:
        Canonical symbol (e.g. ``"XAUUSD"``).
    timeframe:
        Bar size (e.g. ``"1d"``, ``"1h"``).
    start:
        ISO-8601 start date.
    end:
        ISO-8601 end date (defaults to today).
    seed:
        Random seed.  If *None*, derived deterministically from
        *symbol* + *timeframe*.
    asset_class:
        ``"fx"``, ``"metal"`` or ``"crypto"``.  If *None* it is inferred
        from *symbol* using the ``SYMBOLS`` mapping in ``fxlab.data``.

    Returns
    -------
    pd.DataFrame
        Validated OHLCV frame.
    """
    # ---- resolve asset class ----
    if asset_class is None:
        # Lazy import to avoid circular dependency
        try:
            from fxlab.data import SYMBOLS  # type: ignore[import]
            meta = SYMBOLS.get(symbol.upper(), {})
            asset_class = meta.get("asset_class", "fx")
        except Exception:  # noqa: BLE001
            asset_class = "fx"

    sigma_annual = _CLASS_SIGMA.get(asset_class, 0.08)
    bpy = _bars_per_year(timeframe)
    dt = 1.0 / bpy  # year fraction per bar
    sigma = sigma_annual * np.sqrt(dt)
    # drift magnitude for trend regime (annualised ~0.2 × sigma_annual)
    mu_annual = sigma_annual * 0.25
    mu = mu_annual * dt

    # ---- build date index ----
    if end is None:
        end = pd.Timestamp.today().normalize().strftime("%Y-%m-%d")
    freq = _timeframe_freq(timeframe)
    try:
        index = pd.date_range(start=start, end=end, freq=freq, tz=None)
    except Exception as exc:
        raise DataSourceError(f"synthetic: invalid date range {start!r}–{end!r}: {exc}") from exc

    if len(index) == 0:
        raise DataSourceError(f"synthetic: empty date range {start!r}–{end!r}")

    n = len(index)

    # ---- random state ----
    actual_seed = seed if seed is not None else _derive_seed(symbol, timeframe)
    rng = np.random.default_rng(actual_seed)

    # ---- regime-switching Markov chain ----
    # states: 0=noise, 1=uptrend, 2=downtrend
    # p_switch: probability of leaving current regime each bar
    p_switch = 0.02 if timeframe == "1d" else 0.005
    states = np.zeros(n, dtype=int)
    state = 0
    transition = rng.random(n)
    new_state_choices = rng.integers(1, 3, size=n)  # 1=up, 2=down
    for i in range(1, n):
        if state != 0 and transition[i] < p_switch:
            state = 0
        elif state == 0 and transition[i] < p_switch:
            state = int(new_state_choices[i])
        states[i] = state

    drift = np.where(states == 1, mu, np.where(states == 2, -mu, 0.0))

    # ---- log-returns → close path ----
    eps = rng.standard_normal(n)
    log_rets = drift + sigma * eps
    # Geometric Brownian Motion: S_t = S_0 * exp(sum of log_rets)
    S0 = _start_price(symbol, asset_class)
    log_prices = np.concatenate([[np.log(S0)], np.log(S0) + np.cumsum(log_rets[1:])])
    closes = np.exp(log_prices)

    # ---- build open from previous close ----
    gap_sigma = sigma * 0.3
    gap_eps = rng.standard_normal(n)
    opens = np.empty(n)
    opens[0] = closes[0] * np.exp(gap_sigma * gap_eps[0])
    opens[1:] = closes[:-1] * np.exp(gap_sigma * gap_eps[1:])

    # ---- build high / low ----
    # intra-bar range proportional to sigma
    range_factor = np.abs(rng.standard_normal(n)) * sigma * 0.7 + sigma * 0.3
    bar_ranges = closes * range_factor  # absolute price range

    hi_raw = np.maximum(opens, closes) + bar_ranges * rng.uniform(0.3, 0.7, n)
    lo_raw = np.minimum(opens, closes) - bar_ranges * rng.uniform(0.3, 0.7, n)
    highs = np.maximum(hi_raw, np.maximum(opens, closes))
    lows = np.minimum(lo_raw, np.minimum(opens, closes))
    # safety clamp
    lows = np.minimum(lows, np.minimum(opens, closes))
    highs = np.maximum(highs, np.maximum(opens, closes))

    # ---- volume ----
    if asset_class == "fx":
        volumes = np.zeros(n)
    else:
        base_vol = 1_000_000.0 if asset_class == "metal" else 50_000.0
        volumes = rng.lognormal(mean=np.log(base_vol), sigma=0.5, size=n)

    df = pd.DataFrame(
        {
            "open": opens,
            "high": highs,
            "low": lows,
            "close": closes,
            "volume": volumes,
        },
        index=index,
    )

    try:
        return validate_ohlcv(df)
    except ValueError as exc:
        raise DataSourceError(f"synthetic validation failed for {symbol}: {exc}") from exc


def _start_price(symbol: str, asset_class: str) -> float:
    """Return a plausible starting price for the asset."""
    defaults: dict[str, float] = {
        "XAUUSD": 1200.0,
        "XAGUSD": 16.0,
        "EURUSD": 1.10,
        "USDJPY": 110.0,
        "GBPUSD": 1.30,
        "AUDUSD": 0.75,
        "USDCAD": 1.25,
        "USDCHF": 0.95,
        "NZDUSD": 0.68,
        "EURJPY": 125.0,
        "GBPJPY": 148.0,
        "BTCUSD": 30000.0,
        "ETHUSD": 2000.0,
        "SOLUSD": 100.0,
        "XRPUSD": 0.50,
    }
    fallback = {"fx": 1.0, "metal": 100.0, "crypto": 1000.0}
    return defaults.get(symbol.upper(), fallback.get(asset_class, 1.0))
