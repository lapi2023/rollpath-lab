# src/utils.py
from __future__ import annotations
from pathlib import Path
from typing import Optional, Union, List
import pandas as pd

# ---------------------------------------------------------------------------
# Unified CSV reader for strict two-column (Date, Price) files
# ---------------------------------------------------------------------------

# Translation map to normalize various separators to ASCII hyphen '-'.
# Includes full-width forms and common Unicode dashes.
_DATE_SEP_TRANSLATION = {
    ord("／"): "-",
    ord("．"): "-",
    ord("－"): "-",
    0x2010: "-",  # hyphen
    0x2011: "-",  # non-breaking hyphen
    0x2012: "-",  # figure dash
    0x2013: "-",  # en dash
    0x2014: "-",  # em dash
    0x2212: "-",  # minus sign
    ord("/"): "-",
    ord("."): "-",
}

def _normalize_dates_robust(s: pd.Series) -> pd.Series:
    """
    Normalize heterogeneous date strings to naive daily timestamps.

    Accepted inputs (non-exhaustive):
    - 'YYYY-MM-DD'
    - 'YYYY/M/D'
    - 'YYYY/M/D HH:MM:SS'
    - With timezone offsets (e.g., '2025-01-02T15:30:00-05:00')
    - Full-width or Unicode separators (slashes/dashes/dots)

    Steps:
    1) Translate separators to ASCII hyphen '-'.
    2) Parse via pandas.to_datetime(..., errors='coerce', utc=True).
    3) Drop timezone (tz-naive) and floor to day.
    """
    s = s.astype(str).str.strip().str.translate(_DATE_SEP_TRANSLATION)

    # Parse with UTC-awareness so offsets like -05:00 are handled consistently.
    dt = pd.to_datetime(s, errors="coerce", utc=True)

    # Convert to naive and floor to day (00:00).
    return dt.dt.tz_convert(None).dt.floor("D")


# ---------------------------------------------------------------------------
# Robust date normalization
# ---------------------------------------------------------------------------

_DATE_SEP_TRANSLATION = {
    ord("／"): "-",
    ord("．"): "-",
    ord("－"): "-",
    0x2010: "-",  # hyphen
    0x2011: "-",  # non-breaking hyphen
    0x2012: "-",  # figure dash
    0x2013: "-",  # en dash
    0x2014: "-",  # em dash
    0x2212: "-",  # minus sign
    ord("/"): "-",
    ord("."): "-",
}

def _normalize_dates_robust(s: pd.Series) -> pd.Series:
    """
    Normalize heterogeneous date strings to naive daily timestamps.

    Accepts (non-exhaustive):
      - 'YYYY-MM-DD'
      - 'YYYY/M/D'
      - 'YYYY/M/D HH:MM:SS'
      - '31-Dec-1999', 'Jan 2 2020' (English month tokens)
      - timezone offsets like '-05:00'
      - full-width and Unicode separators

    Steps:
      1) Canonicalize separators to ASCII '-' (slash/dot/full-width/Unicode dashes).
      2) Parse via pandas.to_datetime(..., utc=True).
      3) Drop timezone (tz-naive) and floor to day.
    """
    s = s.astype(str).str.strip().str.translate(_DATE_SEP_TRANSLATION)
    dt = pd.to_datetime(s, errors="coerce", utc=True)
    return dt.dt.tz_convert(None).dt.floor("D")


# ---------------------------------------------------------------------------
# Unified, flexible CSV reader
# ---------------------------------------------------------------------------

_PRICE_CANDIDATES: List[str] = [
    "Adj Close", "Adjusted Close", "Price", "Close", "Last", "PX_LAST"
]

def read_price_csv_two_col(
    path: Union[str, Path],
    *,
    assume_header: Optional[bool] = None,
    verbose: bool = False,
) -> pd.DataFrame:
    """
    Read a CSV into a standardized two-column DataFrame ['Date','Price'].

    Layouts accepted:
      * Exactly two columns with a header where the first is 'Date'
        (the second is renamed to 'Price').
      * Two-column raw file without a header (interpreted as ['Date','Price']).
      * Files with more than two columns where a price-like column exists:
        {'Adj Close','Adjusted Close','Price','Close','Last','PX_LAST'} (case-insensitive).

    - Dates are robustly parsed (supports English month tokens, timezones, and mixed separators).
    - Output is always sorted ascending by Date; duplicates by Date keep the last occurrence.
    - Returns DataFrame with columns ['Date','Price'].

    Parameters
    ----------
    path : str or pathlib.Path
        Input CSV path.
    assume_header : Optional[bool], default None
        Force headered (True) or headerless (False) parsing. If None, auto-detect.
    verbose : bool, default False
        If True, prints a brief range summary.

    Raises
    ------
    ValueError
        If no valid rows remain after parsing.
    """
    p = Path(path).resolve()

    def _read_headered() -> pd.DataFrame:
        df0 = pd.read_csv(p)
        df0.columns = [str(c).strip() for c in df0.columns]
        return df0

    def _read_headerless_two() -> pd.DataFrame:
        return pd.read_csv(p, header=None, usecols=[0, 1], names=["Date", "Price"])

    # Step 1: read
    if assume_header is True:
        df = _read_headered()
    elif assume_header is False:
        df = _read_headerless_two()
    else:
        # Try headered; if <2 columns, fallback to headerless-two
        try:
            df_try = _read_headered()
            if df_try.shape[1] >= 2:
                df = df_try
            else:
                df = _read_headerless_two()
        except Exception:
            df = _read_headerless_two()

    # Step 2: choose Date and Price columns
    cols = [str(c) for c in df.columns]
    lc = {c.lower(): c for c in cols}
    date_col: Optional[str] = lc.get("date")

    if date_col is None:
        # When no explicit 'Date' header, assume the LEFT-most column is Date.
        # (Project requirement: "left column = Date" holds for all inputs.)
        date_col = cols[0]

    # Determine price_col:
    if len(cols) == 2 and cols[0].lower() == date_col.lower():
        # exactly two columns -> the other is price
        price_col = cols[1]
    else:
        # try candidate names (case-insensitive)
        price_col = None
        for cand in _PRICE_CANDIDATES:
            if cand.lower() in lc:
                price_col = lc[cand.lower()]
                break
        if price_col is None:
            # fallback: the right-most non-date column
            non_date = [c for c in cols if c != date_col]
            if not non_date:
                raise ValueError(f"No price-like column found in {p.name}.")
            price_col = non_date[-1]

    # Subset & rename
    df = df[[date_col, price_col]].rename(columns={date_col: "Date", price_col: "Price"})

    # Step 3: robust filtering & parsing
    # (Do NOT pre-filter by regex for digits, because month-text rows like 'Jan 2 2020'
    #  would be excluded. Instead, parse first and then drop NaT.)
    df["Date"] = _normalize_dates_robust(df["Date"])
    df["Price"] = pd.to_numeric(df["Price"], errors="coerce")

    # Drop invalid rows
    df = df.dropna(subset=["Date", "Price"])

    # If multiple rows share the same calendar Date (e.g., "YYYY/MM/DD HH:MM:SS"),
    # keep the last occurrence (often the latest timestamp).
    df = df.sort_values("Date").drop_duplicates(subset=["Date"], keep="last").reset_index(drop=True)

    if df.empty:
        raise ValueError(f"No valid rows after parsing Date/Price: {p.name}")

    if verbose:
        first, last, n = df["Date"].min(), df["Date"].max(), len(df)
        print(f"[info] Range: {first.date()} -> {last.date()} rows={n}")

    return df[["Date", "Price"]]

def get_window_and_ppy(invest_val: int, invest_unit: str, data_freq: str) -> tuple[int, int]:
    """Compute (window length, periods-per-year) from data frequency and investment horizon."""
    if data_freq == "daily":
        ppy = 252
        if invest_unit == "days":
            window = invest_val
        elif invest_unit == "months":
            window = int(invest_val * 21)
        elif invest_unit == "years":
            window = invest_val * 252
    elif data_freq == "monthly":
        ppy = 12
        if invest_unit == "days":
            raise ValueError("月次データに対して日単位の投資期間は指定できません")
        elif invest_unit == "months":
            window = invest_val
        elif invest_unit == "years":
            window = invest_val * 12
    elif data_freq == "yearly":
        ppy = 1
        if invest_unit in ["days", "months"]:
            raise ValueError("年次データに対して日/月単位の投資期間は指定できません")
        elif invest_unit == "years":
            window = invest_val
    else:
        raise ValueError(f"Unknown DATA_FREQ: {data_freq}")

    return window, ppy

def format_number_kmg(num: float) -> str:
    """Format a number using compact financial units.

    Examples:
        950 -> '950'
        12_300 -> '12.3K'
        4_560_000 -> '4.56M'
        7_890_000_000 -> '7.89B'
        1_230_000_000_000 -> '1.23T'

    Notes:
        - Uses K/M/B/T for thousand/million/billion/trillion.
        - Keeps up to ~3 significant digits via general format.
    """
    try:
        x = float(num)
    except Exception:
        return str(num)
    ax = abs(x)
    if ax >= 1_000_000_000_000:
        return f"{x / 1_000_000_000_000:g}T"
    if ax >= 1_000_000_000:
        return f"{x / 1_000_000_000:g}B"
    if ax >= 1_000_000:
        return f"{x / 1_000_000:g}M"
    if ax >= 1_000:
        return f"{x / 1_000:g}K"
    # Plain (no scientific) for small numbers
    return f"{x:g}"


def format_integer_commas(num: float | int) -> str:
    """Format as an integer with thousands separators (comma)."""
    try:
        return f"{int(round(float(num))):,}"
    except Exception:
        return str(num)
