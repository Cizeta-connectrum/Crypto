"""Tests for ranking-history Google Sheets sync — no network required.

Covers
------
* _resolve_credentials: FXLAB_GSHEET_CREDENTIALS containing raw JSON key
  content routes to ``Credentials.from_service_account_info``; a file path
  routes to ``Credentials.from_service_account_file``.
* append_ranking_history: worksheet creation, header row, row appends,
  repeated appends accumulate.
* read_ranking_history: missing worksheet → empty DataFrame; numeric
  coercion applied to metric columns but not identifier columns.
"""

from __future__ import annotations

import json
import sys
import types
from typing import Any
from unittest.mock import MagicMock

import pandas as pd
import pytest


# ---------------------------------------------------------------------------
# Fake gspread / google.oauth2 infrastructure
# ---------------------------------------------------------------------------


class _FakeWorksheetNotFound(Exception):
    pass


class _FakeWorksheet:
    def __init__(self, title: str) -> None:
        self.title = title
        self._data: list[list[Any]] = []

    def get_all_values(self) -> list[list[Any]]:
        return list(self._data)

    def append_row(self, row: list, **kwargs: Any) -> None:
        self._data.append(list(row))

    def append_rows(self, rows: list[list], **kwargs: Any) -> None:
        self._data.extend(list(r) for r in rows)


class _FakeSpreadsheet:
    def __init__(self, title: str) -> None:
        self.title = title
        self.id = "fake_id_1234567890"
        self.url = f"https://docs.google.com/spreadsheets/d/{self.id}"
        self._worksheets: dict[str, _FakeWorksheet] = {}

    def worksheet(self, title: str) -> _FakeWorksheet:
        if title not in self._worksheets:
            raise _FakeWorksheetNotFound(title)
        return self._worksheets[title]

    def add_worksheet(self, title: str, rows: int = 100, cols: int = 26) -> _FakeWorksheet:
        ws = _FakeWorksheet(title)
        self._worksheets[title] = ws
        return ws


class _FakeGspreadClient:
    def __init__(self) -> None:
        self._spreadsheets: dict[str, _FakeSpreadsheet] = {}

    def open(self, title: str) -> _FakeSpreadsheet:
        if title in self._spreadsheets:
            return self._spreadsheets[title]
        raise _FakeWorksheetNotFound(title)

    def open_by_url(self, url: str) -> _FakeSpreadsheet:
        for ss in self._spreadsheets.values():
            if ss.url == url:
                return ss
        raise _FakeWorksheetNotFound(url)

    def open_by_key(self, key: str) -> _FakeSpreadsheet:
        for ss in self._spreadsheets.values():
            if ss.id == key:
                return ss
        raise _FakeWorksheetNotFound(key)

    def create(self, title: str) -> _FakeSpreadsheet:
        ss = _FakeSpreadsheet(title)
        self._spreadsheets[title] = ss
        return ss


@pytest.fixture()
def fake_env(monkeypatch: pytest.MonkeyPatch) -> dict[str, Any]:
    """Inject fake gspread + google.oauth2 modules; return handles."""
    client = _FakeGspreadClient()

    gs_mod = types.ModuleType("gspread")
    gs_mod.SpreadsheetNotFound = _FakeWorksheetNotFound  # type: ignore[attr-defined]
    gs_mod.WorksheetNotFound = _FakeWorksheetNotFound  # type: ignore[attr-defined]
    gs_mod.authorize = lambda creds: client  # type: ignore[attr-defined]

    fake_creds = MagicMock(name="fake_credentials")
    sa_mod = types.ModuleType("google.oauth2.service_account")
    sa_cls = MagicMock(name="Credentials")
    sa_cls.from_service_account_info.return_value = fake_creds
    sa_cls.from_service_account_file.return_value = fake_creds
    sa_mod.Credentials = sa_cls  # type: ignore[attr-defined]

    oauth2_mod = types.ModuleType("google.oauth2")
    oauth2_mod.service_account = sa_mod  # type: ignore[attr-defined]

    if "google" not in sys.modules:
        monkeypatch.setitem(sys.modules, "google", types.ModuleType("google"))
    monkeypatch.setitem(sys.modules, "gspread", gs_mod)
    monkeypatch.setitem(sys.modules, "google.oauth2", oauth2_mod)
    monkeypatch.setitem(sys.modules, "google.oauth2.service_account", sa_mod)

    # Clean credential-related env so resolution is deterministic
    monkeypatch.delenv("FXLAB_GSHEET_CREDENTIALS", raising=False)
    monkeypatch.delenv("GOOGLE_APPLICATION_CREDENTIALS", raising=False)
    monkeypatch.delenv("FXLAB_GSHEET_SPREADSHEET", raising=False)

    return {"client": client, "sa_cls": sa_cls, "creds": fake_creds}


_SA_INFO = {
    "type": "service_account",
    "project_id": "demo",
    "client_email": "sa@demo.iam.gserviceaccount.com",
}


# ---------------------------------------------------------------------------
# _resolve_credentials: JSON-in-env vs path-in-env
# ---------------------------------------------------------------------------


class TestResolveCredentialsJsonEnv:
    def test_env_json_string_uses_from_service_account_info(
        self, fake_env: dict, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from fxlab.export.gsheets import _resolve_credentials

        monkeypatch.setenv("FXLAB_GSHEET_CREDENTIALS", json.dumps(_SA_INFO))
        creds = _resolve_credentials(None)

        sa_cls = fake_env["sa_cls"]
        assert creds is fake_env["creds"]
        sa_cls.from_service_account_info.assert_called_once()
        assert sa_cls.from_service_account_info.call_args[0][0] == _SA_INFO
        sa_cls.from_service_account_file.assert_not_called()

    def test_env_json_with_leading_whitespace_detected(
        self, fake_env: dict, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from fxlab.export.gsheets import _resolve_credentials

        monkeypatch.setenv("FXLAB_GSHEET_CREDENTIALS", "  \n" + json.dumps(_SA_INFO))
        _resolve_credentials(None)
        fake_env["sa_cls"].from_service_account_info.assert_called_once()
        fake_env["sa_cls"].from_service_account_file.assert_not_called()

    def test_env_invalid_json_raises_runtime_error(
        self, fake_env: dict, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from fxlab.export.gsheets import _resolve_credentials

        monkeypatch.setenv("FXLAB_GSHEET_CREDENTIALS", "{not valid json")
        with pytest.raises(RuntimeError, match="could not be parsed"):
            _resolve_credentials(None)

    def test_env_path_uses_from_service_account_file(
        self, fake_env: dict, monkeypatch: pytest.MonkeyPatch, tmp_path: Any
    ) -> None:
        from fxlab.export.gsheets import _resolve_credentials

        key_file = tmp_path / "key.json"
        key_file.write_text(json.dumps(_SA_INFO), encoding="utf-8")
        monkeypatch.setenv("FXLAB_GSHEET_CREDENTIALS", str(key_file))

        creds = _resolve_credentials(None)

        sa_cls = fake_env["sa_cls"]
        assert creds is fake_env["creds"]
        sa_cls.from_service_account_file.assert_called_once()
        assert sa_cls.from_service_account_file.call_args[0][0] == str(key_file)
        sa_cls.from_service_account_info.assert_not_called()

    def test_env_missing_path_raises(
        self, fake_env: dict, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from fxlab.export.gsheets import _resolve_credentials

        monkeypatch.setenv("FXLAB_GSHEET_CREDENTIALS", "/no/such/key.json")
        with pytest.raises(RuntimeError, match="missing file"):
            _resolve_credentials(None)

    def test_explicit_json_string_uses_from_service_account_info(
        self, fake_env: dict
    ) -> None:
        from fxlab.export.gsheets import _resolve_credentials

        _resolve_credentials(json.dumps(_SA_INFO))
        fake_env["sa_cls"].from_service_account_info.assert_called_once()
        fake_env["sa_cls"].from_service_account_file.assert_not_called()


# ---------------------------------------------------------------------------
# append_ranking_history / read_ranking_history
# ---------------------------------------------------------------------------


def _make_history_df() -> pd.DataFrame:
    return pd.DataFrame([
        {"run_at": "2026-06-10T12:00:00", "symbol": "XAUUSD", "timeframe": "1d",
         "strategy": "sma_cross_10_50", "family": "sma_cross",
         "sharpe": 1.5, "n_trades": 42},
        {"run_at": "2026-06-10T12:00:00", "symbol": "XAUUSD", "timeframe": "1d",
         "strategy": "rsi_7", "family": "rsi",
         "sharpe": 0.9, "n_trades": 13},
    ])


class TestAppendRankingHistory:
    def test_creates_worksheet_with_header_and_rows(self, fake_env: dict) -> None:
        from fxlab.export.gsheets import _RANKING_WS_TITLE, append_ranking_history

        df = _make_history_df()
        url = append_ranking_history(df, credentials={"type": "service_account"})

        assert url.startswith("https://docs.google.com/spreadsheets/d/")
        # Default spreadsheet title used
        ss = fake_env["client"]._spreadsheets["FXLab Results"]
        ws = ss._worksheets[_RANKING_WS_TITLE]
        assert ws._data[0] == list(df.columns)
        assert len(ws._data) == len(df) + 1

    def test_second_append_accumulates_without_duplicate_header(
        self, fake_env: dict
    ) -> None:
        from fxlab.export.gsheets import _RANKING_WS_TITLE, append_ranking_history

        df = _make_history_df()
        append_ranking_history(df, credentials={"type": "service_account"})
        append_ranking_history(df, credentials={"type": "service_account"})

        ss = fake_env["client"]._spreadsheets["FXLab Results"]
        ws = ss._worksheets[_RANKING_WS_TITLE]
        assert len(ws._data) == 2 * len(df) + 1
        header_rows = [r for r in ws._data if r == list(df.columns)]
        assert len(header_rows) == 1

    def test_spreadsheet_env_override(
        self, fake_env: dict, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from fxlab.export.gsheets import append_ranking_history

        monkeypatch.setenv("FXLAB_GSHEET_SPREADSHEET", "My Custom Sheet")
        append_ranking_history(_make_history_df(), credentials={"type": "service_account"})
        assert "My Custom Sheet" in fake_env["client"]._spreadsheets


class TestReadRankingHistory:
    def test_missing_worksheet_returns_empty_df(self, fake_env: dict) -> None:
        from fxlab.export.gsheets import read_ranking_history

        df = read_ranking_history(credentials={"type": "service_account"})
        assert isinstance(df, pd.DataFrame)
        assert df.empty

    def test_roundtrip_with_numeric_coercion(self, fake_env: dict) -> None:
        from fxlab.export.gsheets import append_ranking_history, read_ranking_history

        append_ranking_history(_make_history_df(), credentials={"type": "service_account"})
        # Sheets returns everything as strings; simulate that
        ss = fake_env["client"]._spreadsheets["FXLab Results"]
        ws = ss._worksheets["ranking_history"]
        ws._data = [[str(c) for c in row] for row in ws._data]

        df = read_ranking_history(credentials={"type": "service_account"})

        assert len(df) == 2
        # Numeric columns coerced
        assert df["sharpe"].dtype.kind == "f"
        assert float(df["sharpe"].iloc[0]) == 1.5
        # Identifier columns kept as strings
        assert df["strategy"].iloc[0] == "sma_cross_10_50"
        assert df["symbol"].iloc[0] == "XAUUSD"
        assert df["run_at"].iloc[0] == "2026-06-10T12:00:00"

    def test_padded_grid_and_repeated_header(self, fake_env: dict) -> None:
        """Wider-than-header grids (empty padded names), duplicate column
        names, and stray header rows in the data must not break the read."""
        from fxlab.export.gsheets import append_ranking_history, read_ranking_history

        append_ranking_history(_make_history_df(), credentials={"type": "service_account"})
        ss = fake_env["client"]._spreadsheets["FXLab Results"]
        ws = ss._worksheets["ranking_history"]
        ws._data = [[str(c) for c in row] for row in ws._data]
        # Simulate a grid wider than the header: pad every row with empties,
        # then add a duplicated column name and a repeated header row.
        header = ws._data[0]
        ws._data = [row + ["", ""] for row in ws._data]
        ws._data[0][-1] = "sharpe"  # duplicate name
        ws._data.insert(2, list(ws._data[0]))  # stray header row inside data

        df = read_ranking_history(credentials={"type": "service_account"})

        assert len(df) == 2  # repeated header row dropped
        assert list(df.columns).count("sharpe") == 1
        assert "" not in df.columns
        assert df["sharpe"].dtype.kind == "f"
        assert float(df["sharpe"].iloc[0]) == 1.5
        assert set(header) - {""} <= set(df.columns) | {""}
