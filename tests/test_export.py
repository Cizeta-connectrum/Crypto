"""Tests for fxlab.export.gsheets — no network required.

Unit tests
----------
* df_to_rows: NaN, inf, numpy types → plain Python values.

Integration tests (monkeypatched gspread)
-----------------------------------------
* export_sweep: "runs" header written once, run row appended with correct
  meta, detail worksheet created with the full table.
* CLI export subcommand: wiring verified via monkeypatched export_sweep.
"""

from __future__ import annotations

import math
import sys
import types
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock

import numpy as np
import pandas as pd
import pytest


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_results_df() -> pd.DataFrame:
    """Return a minimal results DataFrame resembling a sweep output."""
    return pd.DataFrame([
        {"strategy_id": "sma_cross_10_50", "family": "sma_cross", "symbol": "EURUSD",
         "sharpe": 1.23, "total_return": 0.15, "max_drawdown": -0.10},
        {"strategy_id": "sma_cross_20_100", "family": "sma_cross", "symbol": "EURUSD",
         "sharpe": float("nan"), "total_return": float("inf"), "max_drawdown": -0.20},
        {"strategy_id": "rsi_7", "family": "rsi", "symbol": "XAUUSD",
         "sharpe": 0.55, "total_return": 0.08, "max_drawdown": -0.05},
    ])


# ---------------------------------------------------------------------------
# Unit tests: df_to_rows
# ---------------------------------------------------------------------------


class TestDfToRows:
    """Pure unit tests for df_to_rows; no I/O."""

    def test_header_is_first_row(self) -> None:
        from fxlab.export.gsheets import df_to_rows

        df = pd.DataFrame({"a": [1], "b": [2]})
        rows = df_to_rows(df)
        assert rows[0] == ["a", "b"]

    def test_data_row_count(self) -> None:
        from fxlab.export.gsheets import df_to_rows

        df = pd.DataFrame({"x": [1, 2, 3]})
        rows = df_to_rows(df)
        # header + 3 data rows
        assert len(rows) == 4

    def test_nan_becomes_empty_string(self) -> None:
        from fxlab.export.gsheets import df_to_rows

        df = pd.DataFrame({"v": [float("nan"), 1.0]})
        rows = df_to_rows(df)
        assert rows[1][0] == ""  # NaN row
        assert rows[2][0] == 1.0  # normal row

    def test_positive_inf_becomes_string(self) -> None:
        from fxlab.export.gsheets import df_to_rows

        df = pd.DataFrame({"v": [float("inf")]})
        rows = df_to_rows(df)
        assert rows[1][0] == "inf"

    def test_negative_inf_becomes_string(self) -> None:
        from fxlab.export.gsheets import df_to_rows

        df = pd.DataFrame({"v": [float("-inf")]})
        rows = df_to_rows(df)
        assert rows[1][0] == "-inf"

    def test_numpy_int64_becomes_python_int(self) -> None:
        from fxlab.export.gsheets import df_to_rows

        df = pd.DataFrame({"n": np.array([42], dtype=np.int64)})
        rows = df_to_rows(df)
        val = rows[1][0]
        assert isinstance(val, int)
        assert val == 42

    def test_numpy_float64_becomes_python_float(self) -> None:
        from fxlab.export.gsheets import df_to_rows

        df = pd.DataFrame({"f": np.array([3.14], dtype=np.float64)})
        rows = df_to_rows(df)
        val = rows[1][0]
        assert isinstance(val, float)
        assert math.isclose(val, 3.14)

    def test_numpy_nan_float64_becomes_empty_string(self) -> None:
        from fxlab.export.gsheets import df_to_rows

        df = pd.DataFrame({"f": np.array([np.nan], dtype=np.float64)})
        rows = df_to_rows(df)
        assert rows[1][0] == ""

    def test_numpy_inf_float64_becomes_string(self) -> None:
        from fxlab.export.gsheets import df_to_rows

        df = pd.DataFrame({"f": np.array([np.inf], dtype=np.float64)})
        rows = df_to_rows(df)
        assert rows[1][0] == "inf"

    def test_numpy_bool_becomes_python_bool(self) -> None:
        from fxlab.export.gsheets import df_to_rows

        df = pd.DataFrame({"b": np.array([True, False], dtype=np.bool_)})
        rows = df_to_rows(df)
        assert rows[1][0] is True
        assert rows[2][0] is False

    def test_none_becomes_empty_string(self) -> None:
        from fxlab.export.gsheets import df_to_rows

        df = pd.DataFrame({"v": [None, "hello"]})
        rows = df_to_rows(df)
        assert rows[1][0] == ""
        assert rows[2][0] == "hello"

    def test_empty_dataframe(self) -> None:
        from fxlab.export.gsheets import df_to_rows

        df = pd.DataFrame({"a": pd.Series([], dtype=float)})
        rows = df_to_rows(df)
        # Only header row
        assert len(rows) == 1
        assert rows[0] == ["a"]

    def test_mixed_types(self) -> None:
        from fxlab.export.gsheets import df_to_rows

        df = pd.DataFrame({
            "int_col": np.array([1], dtype=np.int64),
            "float_col": np.array([2.5], dtype=np.float64),
            "nan_col": [float("nan")],
            "inf_col": [float("inf")],
            "str_col": ["hello"],
        })
        rows = df_to_rows(df)
        assert rows[1] == [1, 2.5, "", "inf", "hello"]


# ---------------------------------------------------------------------------
# Fake gspread infrastructure for integration tests
# ---------------------------------------------------------------------------


class _FakeWorksheet:
    """Minimal fake gspread Worksheet."""

    def __init__(self, title: str) -> None:
        self.title = title
        self._data: list[list[Any]] = []
        self.update_calls: list[tuple] = []
        self.append_calls: list[list] = []

    def get_all_values(self) -> list[list[Any]]:
        return list(self._data)

    def append_row(self, row: list, **kwargs: Any) -> None:
        self.append_calls.append(row)
        self._data.append(row)

    def update(self, range_name: str, values: list[list], **kwargs: Any) -> None:
        self.update_calls.append((range_name, values))
        # Simulate replacing existing data from A1
        self._data = list(values)


class _FakeSpreadsheet:
    """Minimal fake gspread Spreadsheet."""

    def __init__(self, title: str = "FXLab Test") -> None:
        self.title = title
        self.id = "fake_spreadsheet_id_12345"
        self.url = f"https://docs.google.com/spreadsheets/d/{self.id}"
        self._worksheets: dict[str, _FakeWorksheet] = {}

    def worksheet(self, title: str) -> _FakeWorksheet:
        if title not in self._worksheets:
            raise _FakeSpreadsheetNotFound(f"Worksheet {title!r} not found")
        return self._worksheets[title]

    def add_worksheet(self, title: str, rows: int = 100, cols: int = 26) -> _FakeWorksheet:
        ws = _FakeWorksheet(title=title)
        self._worksheets[title] = ws
        return ws

    def worksheets(self) -> list[_FakeWorksheet]:
        return list(self._worksheets.values())


class _FakeSpreadsheetNotFound(Exception):
    """Simulates gspread.SpreadsheetNotFound."""
    pass


class _FakeGspreadClient:
    """Minimal fake gspread client."""

    def __init__(self) -> None:
        self._spreadsheets: dict[str, _FakeSpreadsheet] = {}

    def open(self, title: str) -> _FakeSpreadsheet:
        if title in self._spreadsheets:
            return self._spreadsheets[title]
        raise _FakeSpreadsheetNotFound(f"Spreadsheet {title!r} not found")

    def open_by_url(self, url: str) -> _FakeSpreadsheet:
        for ss in self._spreadsheets.values():
            if ss.url == url:
                return ss
        raise _FakeSpreadsheetNotFound(f"No spreadsheet with URL {url!r}")

    def open_by_key(self, key: str) -> _FakeSpreadsheet:
        for ss in self._spreadsheets.values():
            if ss.id == key:
                return ss
        raise _FakeSpreadsheetNotFound(f"No spreadsheet with key {key!r}")

    def create(self, title: str) -> _FakeSpreadsheet:
        ss = _FakeSpreadsheet(title=title)
        self._spreadsheets[title] = ss
        return ss


def _make_fake_gspread_module(client: _FakeGspreadClient) -> types.ModuleType:
    """Build a fake 'gspread' module with the required attributes."""
    mod = types.ModuleType("gspread")
    mod.SpreadsheetNotFound = _FakeSpreadsheetNotFound  # type: ignore[attr-defined]
    mod.WorksheetNotFound = _FakeSpreadsheetNotFound  # type: ignore[attr-defined]
    mod.authorize = lambda creds: client  # type: ignore[attr-defined]
    return mod


def _make_fake_google_oauth2() -> tuple[types.ModuleType, types.ModuleType]:
    """Build fake google.oauth2.service_account modules."""
    # fake Credentials object
    fake_creds = MagicMock()

    # fake service_account module
    sa_mod = types.ModuleType("google.oauth2.service_account")
    sa_cls = MagicMock()
    sa_cls.from_service_account_info.return_value = fake_creds
    sa_cls.from_service_account_file.return_value = fake_creds
    sa_mod.Credentials = sa_cls  # type: ignore[attr-defined]

    # fake google.oauth2 package
    oauth2_mod = types.ModuleType("google.oauth2")
    oauth2_mod.service_account = sa_mod  # type: ignore[attr-defined]

    return oauth2_mod, sa_mod


# ---------------------------------------------------------------------------
# Integration tests: export_sweep
# ---------------------------------------------------------------------------


class TestExportSweep:
    """Integration tests using a fake gspread client injected via sys.modules."""

    @pytest.fixture()
    def fake_client(self, monkeypatch: pytest.MonkeyPatch) -> _FakeGspreadClient:
        """Inject fake gspread + google.oauth2 modules and return the fake client."""
        client = _FakeGspreadClient()

        # Build fake modules
        fake_gs_mod = _make_fake_gspread_module(client)
        oauth2_mod, sa_mod = _make_fake_google_oauth2()

        # Build google package mock if not already present
        google_mod = sys.modules.get("google")
        if google_mod is None:
            google_mod = types.ModuleType("google")
            monkeypatch.setitem(sys.modules, "google", google_mod)

        monkeypatch.setitem(sys.modules, "gspread", fake_gs_mod)
        monkeypatch.setitem(sys.modules, "google.oauth2", oauth2_mod)
        monkeypatch.setitem(sys.modules, "google.oauth2.service_account", sa_mod)

        # Force re-import of the gsheets module so it picks up the patched sys.modules
        monkeypatch.delitem(sys.modules, "fxlab.export.gsheets", raising=False)
        monkeypatch.delitem(sys.modules, "fxlab.export", raising=False)

        return client

    def test_returns_url(self, fake_client: _FakeGspreadClient) -> None:
        from fxlab.export.gsheets import export_sweep

        results = _make_results_df()
        url = export_sweep(results, {}, "FXLab Test", credentials={"type": "service_account"})
        assert url.startswith("https://docs.google.com/spreadsheets/d/")

    def test_runs_worksheet_created_with_header(self, fake_client: _FakeGspreadClient) -> None:
        from fxlab.export.gsheets import export_sweep, _RUNS_HEADER

        results = _make_results_df()
        export_sweep(results, {}, "FXLab Test", credentials={"type": "service_account"})

        ss = fake_client._spreadsheets["FXLab Test"]
        runs_ws = ss._worksheets["runs"]

        # First row of the sheet should be the header
        assert runs_ws._data[0] == _RUNS_HEADER

    def test_run_row_appended_once(self, fake_client: _FakeGspreadClient) -> None:
        from fxlab.export.gsheets import export_sweep, _RUNS_HEADER

        results = _make_results_df()
        meta = {
            "symbols": "EURUSD XAUUSD",
            "timeframe": "1d",
            "source": "synthetic",
            "start": "2020-01-01",
            "end": "2024-12-31",
            "cost_bps": 1.0,
            "sl_atr": 2.0,
            "tp_atr": 4.0,
        }
        export_sweep(results, meta, "FXLab Test", credentials={"type": "service_account"})

        ss = fake_client._spreadsheets["FXLab Test"]
        runs_ws = ss._worksheets["runs"]

        # header + exactly one appended data row
        assert len(runs_ws._data) == 2

        run_row = runs_ws._data[1]
        # symbols, timeframe, source in the row
        assert run_row[_RUNS_HEADER.index("symbols")] == "EURUSD XAUUSD"
        assert run_row[_RUNS_HEADER.index("timeframe")] == "1d"
        assert run_row[_RUNS_HEADER.index("source")] == "synthetic"
        assert run_row[_RUNS_HEADER.index("n_results")] == len(results)

    def test_run_row_best_strategy_and_sharpe(self, fake_client: _FakeGspreadClient) -> None:
        """best_strategy = highest sharpe; NaN rows excluded."""
        from fxlab.export.gsheets import export_sweep, _RUNS_HEADER

        results = _make_results_df()
        export_sweep(results, {}, "FXLab Test", credentials={"type": "service_account"})

        ss = fake_client._spreadsheets["FXLab Test"]
        run_row = ss._worksheets["runs"]._data[1]

        # sma_cross_10_50 has sharpe=1.23, which is highest valid sharpe
        assert run_row[_RUNS_HEADER.index("best_strategy")] == "sma_cross_10_50"
        assert math.isclose(float(run_row[_RUNS_HEADER.index("best_sharpe")]), 1.23)

    def test_detail_worksheet_created(self, fake_client: _FakeGspreadClient) -> None:
        from fxlab.export.gsheets import export_sweep

        results = _make_results_df()
        export_sweep(results, {}, "FXLab Test", credentials={"type": "service_account"})

        ss = fake_client._spreadsheets["FXLab Test"]
        # There should be one worksheet besides "runs"
        detail_sheets = [k for k in ss._worksheets if k != "runs"]
        assert len(detail_sheets) == 1

    def test_detail_worksheet_has_header_and_data(self, fake_client: _FakeGspreadClient) -> None:
        from fxlab.export.gsheets import export_sweep

        results = _make_results_df()
        export_sweep(results, {}, "FXLab Test", credentials={"type": "service_account"})

        ss = fake_client._spreadsheets["FXLab Test"]
        detail_sheets = [k for k in ss._worksheets if k != "runs"]
        detail_ws = ss._worksheets[detail_sheets[0]]

        # update() was called with all rows (header + data)
        assert len(detail_ws.update_calls) == 1
        _range, written_rows = detail_ws.update_calls[0]
        assert _range == "A1"
        # header row
        assert written_rows[0] == list(results.columns)
        # total rows = header + len(results)
        assert len(written_rows) == len(results) + 1

    def test_detail_worksheet_nan_replaced(self, fake_client: _FakeGspreadClient) -> None:
        """NaN values in the detail sheet should be empty strings."""
        from fxlab.export.gsheets import export_sweep

        results = _make_results_df()
        export_sweep(results, {}, "FXLab Test", credentials={"type": "service_account"})

        ss = fake_client._spreadsheets["FXLab Test"]
        detail_sheets = [k for k in ss._worksheets if k != "runs"]
        detail_ws = ss._worksheets[detail_sheets[0]]

        _range, written_rows = detail_ws.update_calls[0]
        # Flatten all values and check no raw NaN/inf floats survive
        flat = [cell for row in written_rows[1:] for cell in row]
        for cell in flat:
            if isinstance(cell, float):
                assert not math.isnan(cell), f"NaN found in exported data: {cell}"

    def test_second_export_appends_second_run_row(self, fake_client: _FakeGspreadClient) -> None:
        """Calling export_sweep twice should append two rows to 'runs'."""
        from fxlab.export.gsheets import export_sweep

        results = _make_results_df()
        export_sweep(results, {}, "FXLab Test", credentials={"type": "service_account"})
        export_sweep(results, {}, "FXLab Test", credentials={"type": "service_account"})

        ss = fake_client._spreadsheets["FXLab Test"]
        runs_ws = ss._worksheets["runs"]

        # header + 2 data rows
        assert len(runs_ws._data) == 3

    def test_spreadsheet_created_when_not_found(self, fake_client: _FakeGspreadClient) -> None:
        """If the spreadsheet title doesn't exist it should be created."""
        from fxlab.export.gsheets import export_sweep

        results = _make_results_df()
        assert "Brand New Sheet" not in fake_client._spreadsheets

        export_sweep(results, {}, "Brand New Sheet", credentials={"type": "service_account"})

        assert "Brand New Sheet" in fake_client._spreadsheets


# ---------------------------------------------------------------------------
# CLI export subcommand tests
# ---------------------------------------------------------------------------


class TestCliExport:
    """Tests for the `fxlab export` CLI subcommand."""

    def _make_csv(self, tmp_path: Path) -> Path:
        """Write a minimal sweep CSV and return its path."""
        out = tmp_path / "sweep.csv"
        df = _make_results_df()
        df.to_csv(out, index=False)
        return out

    def _patch_export_sweep(
        self,
        monkeypatch: pytest.MonkeyPatch,
        fake_fn: Any,
    ) -> None:
        """Patch export_sweep on the already-imported gsheets module."""
        import fxlab.export.gsheets as _gs_mod
        monkeypatch.setattr(_gs_mod, "export_sweep", fake_fn)

    def test_export_subcommand_calls_export_sweep(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """CLI export should call export_sweep with the right spreadsheet arg."""
        from fxlab.cli import main

        captured: dict[str, Any] = {}

        def _fake_export(
            results: pd.DataFrame,
            run_meta: dict,
            spreadsheet: str,
            credentials: Any = None,
        ) -> str:
            captured["results"] = results
            captured["run_meta"] = run_meta
            captured["spreadsheet"] = spreadsheet
            return "https://docs.google.com/spreadsheets/d/fake123"

        self._patch_export_sweep(monkeypatch, _fake_export)

        csv_path = self._make_csv(tmp_path)
        rc = main(["export", "--in", str(csv_path), "--gsheet", "My Sheet"])
        assert rc == 0
        assert captured.get("spreadsheet") == "My Sheet"

    def test_export_subcommand_missing_file_returns_1(
        self,
        tmp_path: Path,
    ) -> None:
        from fxlab.cli import main

        rc = main(["export", "--in", str(tmp_path / "missing.csv"), "--gsheet", "X"])
        assert rc == 1

    def test_export_subcommand_gsheet_exception_still_returns_0(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
        capsys: pytest.CaptureFixture,
    ) -> None:
        """Even if export_sweep raises, CLI export should return 0 (graceful)."""
        from fxlab.cli import main

        def _failing_export(*args: Any, **kwargs: Any) -> str:
            raise RuntimeError("gspread not installed")

        self._patch_export_sweep(monkeypatch, _failing_export)

        csv_path = self._make_csv(tmp_path)
        rc = main(["export", "--in", str(csv_path), "--gsheet", "My Sheet"])
        assert rc == 0
        err = capsys.readouterr().err
        assert "Warning" in err or "failed" in err.lower()

    def test_export_subcommand_passes_timeframe_source(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """--timeframe and --source should appear in run_meta."""
        from fxlab.cli import main

        captured: dict[str, Any] = {}

        def _fake_export(
            results: pd.DataFrame,
            run_meta: dict,
            spreadsheet: str,
            credentials: Any = None,
        ) -> str:
            captured["run_meta"] = run_meta
            return "https://docs.google.com/spreadsheets/d/fake"

        self._patch_export_sweep(monkeypatch, _fake_export)

        csv_path = self._make_csv(tmp_path)
        rc = main([
            "export",
            "--in", str(csv_path),
            "--gsheet", "Sheet X",
            "--timeframe", "1h",
            "--source", "synthetic",
        ])
        assert rc == 0
        assert captured["run_meta"]["timeframe"] == "1h"
        assert captured["run_meta"]["source"] == "synthetic"
