"""Google Sheets export for FXLab sweep results.

Credential resolution (in priority order)
------------------------------------------
1. Explicit ``credentials=`` argument — either a ``dict`` of service-account
   info (as returned by ``json.load`` on the key file) or a ``str``/
   ``pathlib.Path`` pointing to the JSON key file on disk.
2. Environment variable ``FXLAB_GSHEET_CREDENTIALS`` — either a path to a
   JSON key file, or the raw JSON key content itself (a value whose first
   non-whitespace character is ``{``).  The raw-JSON form is convenient on
   platforms where secrets are exposed as environment variables only
   (e.g. Hugging Face Spaces).
3. Environment variable ``GOOGLE_APPLICATION_CREDENTIALS`` — path to a JSON
   key file (standard Google SDK convention).

If none of the above resolves, an attempt is made to use Application Default
Credentials (ADC) via ``google.auth.default``; this works on GCP managed
environments (Cloud Run, Vertex, etc.) but will raise on a plain workstation.

Required OAuth scopes: ``spreadsheets`` + ``drive`` (needed to create new
spreadsheets and to open spreadsheets by title via Drive search).

Optional dependency: ``pip install fxlab[gsheets]``
  - gspread >= 6.0
  - google-auth >= 2.0
"""

from __future__ import annotations

import json
import logging
import math
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Header definition for the index "runs" worksheet
# ---------------------------------------------------------------------------

_RUNS_HEADER: list[str] = [
    "run_id",
    "timestamp_utc",
    "symbols",
    "timeframe",
    "source",
    "start",
    "end",
    "cost_bps",
    "sl_atr",
    "tp_atr",
    "n_results",
    "best_strategy",
    "best_sharpe",
]

_SCOPES: list[str] = [
    "https://spreadsheets.google.com/feeds",
    "https://www.googleapis.com/auth/drive",
]

# Worksheet that accumulates ranking-run history (Streamlit Ranking tab).
_RANKING_WS_TITLE = "ranking_history"

# Default spreadsheet target when none is given explicitly or via env.
_DEFAULT_SPREADSHEET = "FXLab Results"

# Columns that must never be numerically coerced when reading history back.
_RANKING_TEXT_COLS: frozenset[str] = frozenset({
    "strategy", "family", "symbol", "timeframe", "source",
    "run_at", "start", "end",
})


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _lazy_import_gspread() -> Any:
    """Import gspread, raising a friendly error if not installed."""
    try:
        import gspread  # type: ignore[import]
        return gspread
    except ImportError as exc:
        raise RuntimeError(
            "gspread is not installed. "
            "Run: pip install fxlab[gsheets]"
        ) from exc


def _lazy_import_service_account() -> Any:
    """Import google.oauth2.service_account, raising a friendly error if not installed."""
    try:
        from google.oauth2 import service_account  # type: ignore[import]
        return service_account
    except ImportError as exc:
        raise RuntimeError(
            "google-auth is not installed. "
            "Run: pip install fxlab[gsheets]"
        ) from exc


def _is_json_value(value: str) -> bool:
    """True if the value looks like inline JSON rather than a file path."""
    return value.lstrip().startswith("{")


def _resolve_credentials(credentials: dict | str | Path | None) -> Any:
    """Resolve credentials to a google.oauth2.service_account.Credentials object.

    Parameters
    ----------
    credentials:
        A dict (service-account info), a path str/Path to a JSON key file, a
        str of raw JSON key content, or None.  When None the environment is
        checked; see module docstring.

    Returns
    -------
    google.oauth2.service_account.Credentials with the required scopes.

    Raises
    ------
    RuntimeError
        If no credentials can be resolved.
    """
    sa_mod = _lazy_import_service_account()

    # --- 1. Explicit dict ---
    if isinstance(credentials, dict):
        return sa_mod.Credentials.from_service_account_info(credentials, scopes=_SCOPES)

    # --- 2. Explicit path or raw JSON string ---
    if isinstance(credentials, (str, Path)):
        if isinstance(credentials, str) and _is_json_value(credentials):
            try:
                info = json.loads(credentials)
            except json.JSONDecodeError as exc:
                raise RuntimeError(
                    "credentials looks like JSON but could not be parsed"
                ) from exc
            return sa_mod.Credentials.from_service_account_info(info, scopes=_SCOPES)
        path = Path(credentials)
        if not path.exists():
            raise RuntimeError(f"Credentials file not found: {path}")
        return sa_mod.Credentials.from_service_account_file(str(path), scopes=_SCOPES)

    # --- 3. Env: FXLAB_GSHEET_CREDENTIALS (path OR raw JSON key content) ---
    env_val = os.environ.get("FXLAB_GSHEET_CREDENTIALS")
    if env_val:
        if _is_json_value(env_val):
            try:
                info = json.loads(env_val)
            except json.JSONDecodeError as exc:
                raise RuntimeError(
                    "FXLAB_GSHEET_CREDENTIALS looks like JSON but could not "
                    "be parsed"
                ) from exc
            return sa_mod.Credentials.from_service_account_info(info, scopes=_SCOPES)
        p = Path(env_val)
        if not p.exists():
            raise RuntimeError(
                f"FXLAB_GSHEET_CREDENTIALS points to missing file: {p}"
            )
        return sa_mod.Credentials.from_service_account_file(str(p), scopes=_SCOPES)

    # --- 4. Env: GOOGLE_APPLICATION_CREDENTIALS ---
    gac = os.environ.get("GOOGLE_APPLICATION_CREDENTIALS")
    if gac:
        p = Path(gac)
        if not p.exists():
            raise RuntimeError(
                f"GOOGLE_APPLICATION_CREDENTIALS points to missing file: {p}"
            )
        return sa_mod.Credentials.from_service_account_file(str(p), scopes=_SCOPES)

    # --- 5. Application Default Credentials ---
    try:
        import google.auth  # type: ignore[import]
        creds, _ = google.auth.default(scopes=_SCOPES)
        return creds
    except Exception as exc:  # noqa: BLE001
        raise RuntimeError(
            "No Google credentials found. Provide the credentials= argument, "
            "set FXLAB_GSHEET_CREDENTIALS or GOOGLE_APPLICATION_CREDENTIALS "
            "to the path of a service-account JSON key, or configure "
            "Application Default Credentials. "
            "Run: pip install fxlab[gsheets]"
        ) from exc


def _open_or_create_spreadsheet(gc: Any, spreadsheet: str) -> Any:
    """Open a spreadsheet by URL, ID, or title; create by title if not found.

    Parameters
    ----------
    gc:
        Authorised gspread client.
    spreadsheet:
        Full URL, bare spreadsheet ID (44-char base58 string), or a title.

    Returns
    -------
    gspread.Spreadsheet
    """
    gspread = _lazy_import_gspread()

    # --- URL ---
    if spreadsheet.startswith("https://"):
        logger.debug("Opening spreadsheet by URL: %s", spreadsheet)
        return gc.open_by_url(spreadsheet)

    # --- Looks like a spreadsheet key (no spaces, reasonable length) ---
    if " " not in spreadsheet and len(spreadsheet) >= 20:
        try:
            logger.debug("Trying to open spreadsheet by key: %s", spreadsheet)
            return gc.open_by_key(spreadsheet)
        except gspread.SpreadsheetNotFound:
            pass

    # --- Title ---
    try:
        logger.debug("Trying to open spreadsheet by title: %s", spreadsheet)
        return gc.open(spreadsheet)
    except gspread.SpreadsheetNotFound:
        pass

    # --- Create ---
    logger.info("Spreadsheet %r not found; creating a new one.", spreadsheet)
    return gc.create(spreadsheet)


def _ensure_runs_worksheet(spreadsheet: Any) -> Any:
    """Return the 'runs' worksheet, creating it with a header row if absent.

    If the spreadsheet is brand-new it will already contain a default
    'Sheet1'; the header is only written when the sheet is empty.
    """
    gspread = _lazy_import_gspread()

    try:
        ws = spreadsheet.worksheet("runs")
    except gspread.WorksheetNotFound:
        ws = spreadsheet.add_worksheet(title="runs", rows=1000, cols=len(_RUNS_HEADER))

    existing = ws.get_all_values()
    if not existing or existing[0] != _RUNS_HEADER:
        if not existing:
            ws.append_row(_RUNS_HEADER, value_input_option="USER_ENTERED")
        else:
            # Sheet exists but header doesn't match — overwrite row 1
            # Keyword args: gspread 6.x swapped the positional order of
            # (range_name, values); keywords are unambiguous in 5.x and 6.x.
            ws.update(values=[_RUNS_HEADER], range_name="A1",
                      value_input_option="USER_ENTERED")

    return ws


# ---------------------------------------------------------------------------
# Public helpers
# ---------------------------------------------------------------------------


def df_to_rows(df: pd.DataFrame) -> list[list[Any]]:
    """Convert a DataFrame to a list of plain-Python rows suitable for gspread.

    The first row is the header (column names).  Subsequent rows are the
    data values.  Special values are normalised:

    - ``float("nan")`` / ``numpy.nan``  → ``""`` (empty string)
    - ``float("inf")`` / ``float("-inf")``  → ``"inf"`` / ``"-inf"``
    - numpy scalar types (int64, float64, …)  → built-in ``int`` / ``float``
    - ``pandas.NaT`` / ``None``  → ``""``

    Parameters
    ----------
    df:
        Any DataFrame.

    Returns
    -------
    list[list[Any]]
        ``[header_row, *data_rows]``
    """

    def _coerce(val: Any) -> Any:
        # pandas/numpy NA sentinels
        if val is None:
            return ""
        try:
            import numpy as np  # noqa: PLC0415
            if isinstance(val, float) and math.isnan(val):
                return ""
            if isinstance(val, float) and math.isinf(val):
                return "inf" if val > 0 else "-inf"
            # numpy scalars → Python builtins
            if isinstance(val, np.integer):
                return int(val)
            if isinstance(val, np.floating):
                f = float(val)
                if math.isnan(f):
                    return ""
                if math.isinf(f):
                    return "inf" if f > 0 else "-inf"
                return f
            if isinstance(val, np.bool_):
                return bool(val)
        except ImportError:
            pass
        # pandas NaT and similar
        try:
            import pandas as _pd  # noqa: PLC0415
            if _pd.isna(val):
                return ""
        except (TypeError, ImportError):
            pass
        return val

    header: list[Any] = list(df.columns)
    data_rows: list[list[Any]] = [
        [_coerce(cell) for cell in row]
        for row in df.itertuples(index=False, name=None)
    ]
    return [header, *data_rows]


# ---------------------------------------------------------------------------
# Main public API
# ---------------------------------------------------------------------------


def export_sweep(
    results: pd.DataFrame,
    run_meta: dict,
    spreadsheet: str,
    credentials: dict | str | Path | None = None,
) -> str:
    """Export a sweep results DataFrame to Google Sheets and return the URL.

    Behaviour
    ---------
    The function maintains two sets of worksheets:

    * **"runs"** — a permanent index of every export.  If the sheet does not
      exist it is created with a fixed header row (see :data:`_RUNS_HEADER`).
      One row is appended per call.

    * **<run_id>** — a per-run detail sheet containing the full ``results``
      table (header + data rows).  The worksheet is sized to
      ``len(results) + 1`` rows.

    Parameters
    ----------
    results:
        DataFrame of sweep results (typically produced by ``fxlab sweep``).
        Each row is one (strategy, symbol) result.
    run_meta:
        Metadata dict with optional keys:
        ``symbols``, ``timeframe``, ``source``, ``start``, ``end``,
        ``cost_bps``, ``sl_atr``, ``tp_atr``.
        Missing keys default to ``""``.
    spreadsheet:
        The target spreadsheet: a full ``https://docs.google.com/…`` URL,
        a bare spreadsheet ID, or a title string.  If a spreadsheet with that
        title does not exist in the service account's Drive it will be created
        (the resulting sheet lives in the SA's Drive — share an existing sheet
        you own with the SA's ``client_email`` for Editor access instead, as
        described in the Google Sheets export section of README.md).
    credentials:
        Optional explicit credentials.  See module docstring for the full
        resolution order.

    Returns
    -------
    str
        The ``https://docs.google.com/spreadsheets/d/<id>`` URL.

    Raises
    ------
    RuntimeError
        If gspread / google-auth are not installed, or if no credentials can
        be resolved.
    """
    gspread = _lazy_import_gspread()

    creds = _resolve_credentials(credentials)
    gc = gspread.authorize(creds)

    # ------------------------------------------------------------------
    # Open / create spreadsheet
    # ------------------------------------------------------------------
    ssheet = _open_or_create_spreadsheet(gc, spreadsheet)
    url: str = ssheet.url

    # ------------------------------------------------------------------
    # Build run-level summary
    # ------------------------------------------------------------------
    now_utc = datetime.now(tz=timezone.utc)
    run_id = now_utc.strftime("%Y%m%d-%H%M%S")
    timestamp_str = now_utc.strftime("%Y-%m-%dT%H:%M:%SZ")

    # Find best row by sharpe
    best_strategy = ""
    best_sharpe: float | str = ""
    if not results.empty and "sharpe" in results.columns:
        try:
            numeric_sharpe = pd.to_numeric(results["sharpe"], errors="coerce")
            valid = numeric_sharpe.dropna()
            if not valid.empty:
                best_idx = int(valid.idxmax())
                best_strategy = str(results.at[best_idx, "strategy_id"]) if "strategy_id" in results.columns else ""
                best_sharpe = float(valid[best_idx])
        except Exception:  # noqa: BLE001
            pass

    run_row: list[Any] = [
        run_id,
        timestamp_str,
        str(run_meta.get("symbols", "")),
        str(run_meta.get("timeframe", "")),
        str(run_meta.get("source", "")),
        str(run_meta.get("start", "")),
        str(run_meta.get("end", "")),
        str(run_meta.get("cost_bps", "")),
        str(run_meta.get("sl_atr", "")),
        str(run_meta.get("tp_atr", "")),
        len(results),
        best_strategy,
        best_sharpe if best_sharpe != "" else "",
    ]

    # ------------------------------------------------------------------
    # Ensure "runs" worksheet and append summary row
    # ------------------------------------------------------------------
    runs_ws = _ensure_runs_worksheet(ssheet)
    runs_ws.append_row(run_row, value_input_option="USER_ENTERED")
    logger.info("Appended run %s to 'runs' worksheet.", run_id)

    # ------------------------------------------------------------------
    # Create per-run detail worksheet
    # ------------------------------------------------------------------
    n_rows = max(len(results) + 1, 2)
    n_cols = max(len(results.columns), 1) if not results.empty else 1
    detail_ws = ssheet.add_worksheet(title=run_id, rows=n_rows, cols=n_cols)
    detail_rows = df_to_rows(results)
    if detail_rows:
        detail_ws.update(
            values=detail_rows,
            range_name="A1",
            value_input_option="USER_ENTERED",
        )
    logger.info("Created detail worksheet '%s' with %d rows.", run_id, len(results))

    return url


# ---------------------------------------------------------------------------
# Ranking history (Streamlit Ranking tab)
# ---------------------------------------------------------------------------


def _resolve_spreadsheet_target(spreadsheet: str | None) -> str:
    """Default spreadsheet target: arg → env FXLAB_GSHEET_SPREADSHEET → fixed."""
    return (
        spreadsheet
        or os.environ.get("FXLAB_GSHEET_SPREADSHEET")
        or _DEFAULT_SPREADSHEET
    )


def append_ranking_history(
    df: pd.DataFrame,
    spreadsheet: str | None = None,
    credentials: dict | str | Path | None = None,
) -> str:
    """Append ranking rows to the 'ranking_history' worksheet.

    ``spreadsheet`` defaults to env ``FXLAB_GSHEET_SPREADSHEET`` or
    ``"FXLab Results"``.  The worksheet is created with a header row (the
    DataFrame's columns) if missing.

    Parameters
    ----------
    df:
        Ranking rows including run metadata columns (run_at, symbol, …).
    spreadsheet:
        Target spreadsheet URL / ID / title (optional, see above).
    credentials:
        Optional explicit credentials; see module docstring.

    Returns
    -------
    str
        The spreadsheet URL.
    """
    gspread = _lazy_import_gspread()

    creds = _resolve_credentials(credentials)
    gc = gspread.authorize(creds)
    ssheet = _open_or_create_spreadsheet(gc, _resolve_spreadsheet_target(spreadsheet))

    try:
        ws = ssheet.worksheet(_RANKING_WS_TITLE)
    except gspread.WorksheetNotFound:
        ws = ssheet.add_worksheet(
            title=_RANKING_WS_TITLE,
            rows=max(len(df) + 1, 2),
            cols=max(len(df.columns), 1),
        )

    rows = df_to_rows(df)
    header, data_rows = rows[0], rows[1:]

    existing = ws.get_all_values()
    if not existing:
        ws.append_row(header, value_input_option="USER_ENTERED")
    if data_rows:
        ws.append_rows(data_rows, value_input_option="USER_ENTERED")

    logger.info(
        "Appended %d ranking rows to worksheet %r.", len(data_rows), _RANKING_WS_TITLE
    )
    return str(ssheet.url)


def read_ranking_history(
    spreadsheet: str | None = None,
    credentials: dict | str | Path | None = None,
) -> pd.DataFrame:
    """Read the full 'ranking_history' worksheet into a DataFrame.

    Returns an empty DataFrame if the worksheet doesn't exist.  Numeric
    columns are coerced with ``pd.to_numeric(errors="coerce")``; identifier
    columns (strategy / family / symbol / timeframe / source / run_at /
    start / end) are left as strings.

    Parameters
    ----------
    spreadsheet:
        Target spreadsheet URL / ID / title.  Defaults to env
        ``FXLAB_GSHEET_SPREADSHEET`` or ``"FXLab Results"``.
    credentials:
        Optional explicit credentials; see module docstring.
    """
    gspread = _lazy_import_gspread()

    creds = _resolve_credentials(credentials)
    gc = gspread.authorize(creds)
    ssheet = _open_or_create_spreadsheet(gc, _resolve_spreadsheet_target(spreadsheet))

    try:
        ws = ssheet.worksheet(_RANKING_WS_TITLE)
    except gspread.WorksheetNotFound:
        return pd.DataFrame()

    values = ws.get_all_values()
    if not values:
        return pd.DataFrame()

    # Locate the header row: appends can leave stray/blank rows above it,
    # and a sheet without any recognizable header is unusable.
    header_idx = None
    for i, row in enumerate(values):
        cells = [str(c).strip() for c in row]
        if "strategy" in cells and "run_at" in cells:
            header_idx = i
            break
    if header_idx is None:
        logger.warning(
            "Worksheet %r has no recognizable header row; returning empty.",
            _RANKING_WS_TITLE,
        )
        return pd.DataFrame()

    header = [str(c).strip() for c in values[header_idx]]
    # Skip stray repeats of the header row mixed into the data.
    data = [
        row
        for row in values[header_idx + 1 :]
        if [str(c).strip() for c in row] != header
    ]
    df = pd.DataFrame(data, columns=header)
    # The sheet grid may be wider than the header (padded empty names) and
    # may contain duplicate column names; both break per-column coercion.
    df = df.loc[:, [bool(c) for c in df.columns]]
    df = df.loc[:, ~pd.Index(df.columns).duplicated(keep="first")]
    for col in df.columns:
        if col not in _RANKING_TEXT_COLS:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    return df
