"""Smoke tests for fxlab.cli using synthetic data.

These tests verify the end-to-end CLI paths:
  - list       → prints strategy count (300)
  - run        → returns 0 with synthetic data
  - sweep      → returns 0, CSV is created
  - report     → returns 0 reading the sweep CSV

Strategy imports are lazy inside test bodies so pytest collection does not
fail if fxlab.strategies is not yet present.
"""

from __future__ import annotations

import csv
import importlib
import sys
from pathlib import Path

import pytest

# ---------------------------------------------------------------------------
# Guard: skip all tests if fxlab.cli itself can't be imported
# ---------------------------------------------------------------------------

try:
    from fxlab.cli import main
    _CLI_AVAILABLE = True
except ImportError:
    _CLI_AVAILABLE = False

pytestmark = pytest.mark.skipif(
    not _CLI_AVAILABLE,
    reason="fxlab.cli not importable",
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _siblings_available() -> bool:
    """Return True only if all three sibling modules are importable."""
    for mod in ("fxlab.data", "fxlab.engine", "fxlab.strategies"):
        try:
            importlib.import_module(mod)
        except ImportError:
            return False
    return True


def _first_strategy_id() -> str:
    """Return the first strategy id from build_all(), or a fallback."""
    try:
        build_all = importlib.import_module("fxlab.strategies").build_all
        strategies = build_all()
        if strategies:
            return strategies[0].id
    except Exception:  # noqa: BLE001
        pass
    return "sma_cross_10_50"


def _first_family() -> str:
    """Return the first strategy family from build_all(), or a fallback."""
    try:
        build_all = importlib.import_module("fxlab.strategies").build_all
        strategies = build_all()
        if strategies:
            return strategies[0].family
    except Exception:  # noqa: BLE001
        pass
    return "sma_cross"


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


@pytest.mark.skipif(not _siblings_available(), reason="sibling modules not yet available")
def test_list_prints_300(capsys):
    """fxlab list should report exactly 300 strategies."""
    rc = main(["list"])
    assert rc == 0
    out = capsys.readouterr().out
    # The count line: "Total: 300"
    assert "300" in out, f"Expected '300' in output, got:\n{out}"


@pytest.mark.skipif(not _siblings_available(), reason="sibling modules not yet available")
def test_run_synthetic_returns_zero(capsys):
    """fxlab run with synthetic source should return exit code 0."""
    strategy_id = _first_strategy_id()
    rc = main([
        "run",
        "--strategy", strategy_id,
        "--symbol", "EURUSD",
        "--source", "synthetic",
        "--timeframe", "1d",
    ])
    assert rc == 0, f"Expected exit code 0, got {rc}"


@pytest.mark.skipif(not _siblings_available(), reason="sibling modules not yet available")
def test_sweep_creates_csv(tmp_path):
    """fxlab sweep with synthetic source should return 0 and create the CSV."""
    family = _first_family()
    out_csv = tmp_path / "s.csv"

    rc = main([
        "sweep",
        "--symbols", "EURUSD",
        "--strategies", f"family:{family}",
        "--source", "synthetic",
        "--timeframe", "1d",
        "--jobs", "1",
        "--out", str(out_csv),
    ])
    assert rc == 0, f"Expected exit code 0, got {rc}"
    assert out_csv.exists(), "Output CSV was not created"
    assert out_csv.stat().st_size > 0, "Output CSV is empty"

    # Validate CSV has expected columns
    with out_csv.open("r", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        rows = list(reader)
    assert len(rows) > 0, "CSV has no data rows"
    assert "strategy_id" in rows[0], "CSV missing 'strategy_id' column"
    assert "family" in rows[0], "CSV missing 'family' column"
    assert "symbol" in rows[0], "CSV missing 'symbol' column"


@pytest.mark.skipif(not _siblings_available(), reason="sibling modules not yet available")
def test_report_from_csv(tmp_path, capsys):
    """fxlab report on a sweep CSV should return 0 and print a leaderboard."""
    # First generate a sweep CSV
    family = _first_family()
    out_csv = tmp_path / "report_test.csv"

    rc_sweep = main([
        "sweep",
        "--symbols", "EURUSD",
        "--strategies", f"family:{family}",
        "--source", "synthetic",
        "--timeframe", "1d",
        "--jobs", "1",
        "--out", str(out_csv),
    ])
    assert rc_sweep == 0, f"Sweep failed with code {rc_sweep}"
    assert out_csv.exists(), "Sweep CSV not created"

    # Now run report on it
    capsys.readouterr()  # clear previous stdout
    rc_report = main([
        "report",
        "--in", str(out_csv),
        "--metric", "sharpe",
        "--top", "10",
    ])
    assert rc_report == 0, f"Expected exit code 0, got {rc_report}"
    out = capsys.readouterr().out
    assert "sharpe" in out.lower(), f"Expected 'sharpe' in report output, got:\n{out}"


# ---------------------------------------------------------------------------
# Parser-only tests (no siblings required)
# ---------------------------------------------------------------------------


def test_no_subcommand_returns_1():
    """Running with no subcommand should return exit code 1."""
    rc = main([])
    assert rc == 1


def test_unknown_subcommand_exits_nonzero():
    """An unknown subcommand should exit non-zero (argparse handles this)."""
    with pytest.raises(SystemExit) as exc_info:
        main(["nonexistent_subcommand"])
    assert exc_info.value.code != 0


def test_report_missing_file_returns_1(tmp_path):
    """report --in on a nonexistent file should return 1."""
    rc = main(["report", "--in", str(tmp_path / "does_not_exist.csv")])
    assert rc == 1


def test_list_without_strategies_module(monkeypatch, capsys):
    """list subcommand should return 1 gracefully when strategies module missing.

    We patch the import inside _cmd_list by making 'fxlab.strategies' raise
    ImportError on import, even if already cached in sys.modules.
    """
    import builtins
    import sys

    real_import = builtins.__import__

    def _blocking_import(name, *args, **kwargs):
        if name == "fxlab.strategies":
            raise ImportError("fxlab.strategies blocked for test")
        return real_import(name, *args, **kwargs)

    # Remove cached module so the lazy import in _cmd_list actually fires
    monkeypatch.delitem(sys.modules, "fxlab.strategies", raising=False)
    monkeypatch.setattr(builtins, "__import__", _blocking_import)

    rc = main(["list"])
    assert rc == 1
