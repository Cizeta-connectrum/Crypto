"""Tests for the FXLab strategy library (300 strategies)."""

from __future__ import annotations

from collections import Counter

import numpy as np
import pandas as pd
import pytest

from fxlab import strategies
from fxlab.strategies import build_all

EXPECTED_TOTAL = 300


def _synthetic_ohlcv(n: int = 600, seed: int = 7, trend: float = 0.04) -> pd.DataFrame:
    """Seeded random walk -> close, with plausible OHLC and a mild drift."""
    rng = np.random.default_rng(seed)
    steps = rng.normal(loc=trend, scale=1.0, size=n)
    close = 100.0 + np.cumsum(steps)
    close = np.maximum(close, 1.0)
    idx = pd.date_range("2018-01-01", periods=n, freq="D")
    close_s = pd.Series(close, index=idx)
    prev = close_s.shift(1).fillna(close_s.iloc[0])
    open_ = prev + rng.normal(0.0, 0.3, n)
    noise_hi = np.abs(rng.normal(0.0, 0.8, n))
    noise_lo = np.abs(rng.normal(0.0, 0.8, n))
    high = np.maximum.reduce([open_, close, prev]) + noise_hi
    low = np.minimum.reduce([open_, close, prev]) - noise_lo
    vol = np.abs(rng.normal(1000.0, 200.0, n))
    df = pd.DataFrame(
        {"open": open_, "high": high, "low": low, "close": close, "volume": vol},
        index=idx,
    )
    return df.astype("float64")


@pytest.fixture(scope="module")
def df() -> pd.DataFrame:
    return _synthetic_ohlcv()


@pytest.fixture(scope="module")
def all_strats() -> list:
    return build_all()


# ---------------------------------------------------------------- counts/ids

def test_exactly_300(all_strats: list) -> None:
    assert len(all_strats) == EXPECTED_TOTAL


def test_unique_ids(all_strats: list) -> None:
    ids = [s.id for s in all_strats]
    dupes = [k for k, v in Counter(ids).items() if v > 1]
    assert not dupes, f"duplicate ids: {dupes}"
    assert len(ids) == EXPECTED_TOTAL


def test_ids_filesystem_safe(all_strats: list) -> None:
    import re

    pat = re.compile(r"^[A-Za-z0-9_]+$")
    bad = [s.id for s in all_strats if not pat.match(s.id)]
    assert not bad, f"unsafe ids: {bad}"


def test_family_and_params_present(all_strats: list) -> None:
    for s in all_strats:
        assert isinstance(s.family, str) and s.family
        assert isinstance(s.params, dict)


def test_at_least_25_families(all_strats: list) -> None:
    fams = {s.family for s in all_strats}
    assert len(fams) >= 25


def test_public_api(all_strats: list) -> None:
    assert strategies.ids() == [s.id for s in all_strats]
    first = all_strats[0]
    assert strategies.get(first.id).id == first.id
    with pytest.raises(KeyError):
        strategies.get("does_not_exist")


# ---------------------------------------------------------------- generate()

def test_generate_contract(all_strats: list, df: pd.DataFrame) -> None:
    for s in all_strats:
        pos = s.generate(df)
        assert isinstance(pos, pd.Series), s.id
        assert pos.index.equals(df.index), s.id
        assert str(pos.dtype) == "float64", s.id
        assert not pos.isna().any(), f"NaN in {s.id}"
        assert pos.min() >= -1.0 - 1e-9 and pos.max() <= 1.0 + 1e-9, s.id


def test_mostly_nonzero(all_strats: list, df: pd.DataFrame) -> None:
    nonzero = sum(1 for s in all_strats if (s.generate(df) != 0.0).any())
    assert nonzero >= 250, f"only {nonzero} strategies took a position"


def test_each_family_active(all_strats: list, df: pd.DataFrame) -> None:
    """Every family must have at least one nonzero strategy on a trending sample."""
    by_family: dict[str, list] = {}
    for s in all_strats:
        by_family.setdefault(s.family, []).append(s)
    flat = []
    for fam, members in by_family.items():
        active = any((m.generate(df) != 0.0).any() for m in members)
        if not active:
            flat.append(fam)
    assert not flat, f"families with no activity: {flat}"


def test_determinism(all_strats: list, df: pd.DataFrame) -> None:
    for s in all_strats[::7]:  # sample for speed; still broad coverage
        a = s.generate(df)
        b = s.generate(df)
        pd.testing.assert_series_equal(a, b)


# ---------------------------------------------------------------- lookahead

def _representatives(all_strats: list) -> list:
    """One representative strategy per family (first in build order)."""
    seen: dict[str, object] = {}
    for s in all_strats:
        if s.family not in seen:
            seen[s.family] = s
    return list(seen.values())


def test_no_lookahead_per_family(all_strats: list, df: pd.DataFrame) -> None:
    """Prefix invariance: value at bar n-1 is unaffected by future bars."""
    reps = _representatives(all_strats)
    for s in reps:
        full = s.generate(df)
        for n in (200, 400):
            prefix = s.generate(df.iloc[:n])
            assert prefix.index.equals(df.index[:n]), s.id
            # last bar of the prefix must match the full-run value at that bar
            diff = abs(float(prefix.iloc[-1]) - float(full.iloc[n - 1]))
            assert diff <= 1e-9, (
                f"lookahead in {s.id}: prefix[{n - 1}]={prefix.iloc[-1]} "
                f"full[{n - 1}]={full.iloc[n - 1]}"
            )
