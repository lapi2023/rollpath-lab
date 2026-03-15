# src/commands/generate_spx_total_return.py
from __future__ import annotations

"""
Generate an S&P 500 daily **Total Return (TR)** series by combining:
1) a CSV of **daily S&P 500 price levels** (Date, Price), and
2) a CSV of **monthly dividend yield** (annualized, in percent).

The script:
- Resolves the latest matching files for the price and dividend inputs (wildcards allowed).
- Parses and normalizes dates from both sources (robust to several separators and header forms).
- Converts monthly dividend yield into a daily “drift” component based on the actual day gap.
- Builds the total-return price path by compounding daily: r_TR = r_price + yield * dt,
  where dt is the fraction of a year using ACT/365 (i.e., (Δdays)/365).
- Scales the computed TR path so that its last value equals the last observed price,
  preserving TR shape while aligning to the observable price level.
- Saves a CSV named: ``^SPX_{start}_{end}_daily_TR.csv`` to the output directory.

Input requirements
------------------
Price file (daily):
    - Columns are inferred. Accepts:
        * Explicit headers "Date" and "Price", **or**
        * Two-column file with header row: "Date" + <any column> (renamed to "Price"), **or**
        * No header file, interpreted as two columns (Date, Price).
    - Date strings can use '/', '-', '.', and common Unicode dash variants.
    - The series is sorted by Date after parsing.

Dividend file (monthly):
    - Must have columns: "Date" and "Value".
    - Two accepted formats for "Date" (examples):
        * ``dd-MMM-yy`` (e.g., 31-Dec-99). A century boundary is inferred from order so that
          years around 2000 are mapped to either 19xx or 20xx consistently.
        * A compact month-day-year code with a two-letter month token (e.g., "Ja120023").
          See the function docstring for details on supported tokens.

Computation details
-------------------
- Daily price return is the simple return: ``r_price = Price[t]/Price[t-1] - 1``.
- The dividend component is derived by spreading the **annualized** dividend yield over
  the actual day gap using ACT/365: ``drift = yield_annual * (Δdays / 365)``.
- The daily TR return is ``r_TR = r_price + drift``, and the TR “price” is obtained by
  compounding ``(1 + r_TR)`` from the first day, then scaled to end at the last spot price.

Output
------
- CSV with two columns: "Date" (formatted as `YYYY/M/D` with no leading zeros on Unix
  and `YYYY/M/D` with platform-appropriate directives on Windows) and "Price" (float).
- File is written to `--outdir` with name inferred from the date span of the TR series.

Examples
--------
Run with defaults (uses latest matching files in the data hierarchy):

    python -m src.commands.generate_spx_total_return

Specify explicit inputs and date filters:

    python -m src.commands.generate_spx_total_return \\
        --spx 'data/^spx_d_*_to_*.csv' \\
        --dividend 'data/raw/SPX Dividend Yield by Month_*.csv' \\
        --start 1990-01-01 --end 2025-12-31 \\
        --outdir data

Notes
-----
- The script prints informative summaries (input resolutions, ranges, and output path).
- If you maintain your own price history column names, ensure "Date" and "Price" exist
  (or let the loader infer and rename as described above).
"""

import argparse
from pathlib import Path
import platform
import pandas as pd
import re


def _contains_wildcard(s: str) -> bool:
    """
    Check whether the path specification contains shell wildcard characters.

    Parameters
    ----------
    s : str
        Path specification string.

    Returns
    -------
    bool
        True if any of '*', '?', '[' or ']' is present; False otherwise.
    """
    return any(ch in s for ch in '*?[]')


def _try_extract_end_token(p: Path) -> str:
    """
    Attempt to extract an end-date token from a filename pattern like ``*_to_YYYYMMDD``.

    Parameters
    ----------
    p : pathlib.Path
        Path to inspect.

    Returns
    -------
    str
        The matched numeric token (e.g., '20241231') if found; otherwise an empty string.
    """
    m = re.search(r'_to_(\d{6,8})', p.name)
    return m.group(1) if m else ''


def resolve_latest(spec: str) -> Path:
    """
    Resolve a path specification to the most likely **latest** file.

    This helper supports direct paths (no wildcards) and wildcard patterns.
    For wildcard matches, candidates are scored by:
      1. CSV preference (CSV files rank higher than non-CSV),
      2. presence/lexicographic order of an end-date token like ``_to_YYYYMMDD``,
      3. file modification time (newest last-modified wins if ties persist).

    Parameters
    ----------
    spec : str
        Direct path or a glob-style wildcard pattern.

    Returns
    -------
    pathlib.Path
        The resolved path.

    Raises
    ------
    FileNotFoundError
        If the direct path does not exist, or no files match the wildcard.
    """
    s = str(spec)
    p = Path(s)
    if not _contains_wildcard(s):
        if p.exists():
            return p
        raise FileNotFoundError(f"Path not found: {p}")

    cands = list(Path('.').glob(s))
    if not cands:
        raise FileNotFoundError(f"No files matched: {spec}")

    def _score(path: Path):
        end_tok = _try_extract_end_token(path)
        return (1 if path.suffix.lower() == '.csv' else 0, end_tok, path.stat().st_mtime)

    best = max(cands, key=_score)
    print(f"[info] Resolved '{spec}' -> {best}")
    return best


def _normalize_dates_for_spx(s: pd.Series) -> pd.Series:
    """
    Normalize date-like strings from the SPX daily file into UTC-naïve calendar dates.

    The function is robust to ASCII and common Unicode separators ('/', '-', '.', various dashes).
    Values are coerced with `pandas.to_datetime(..., utc=True)`, then converted to naïve
    datetimes and floored to day.

    Parameters
    ----------
    s : pandas.Series
        Series of raw date strings.

    Returns
    -------
    pandas.Series
        Datetime64 series (naïve), daily granularity.
    """
    s = s.astype(str).str.strip()
    trans = {
        ord('／'): '-', ord('．'): '-', ord('－'): '-',
        0x2010: '-', 0x2011: '-', 0x2012: '-', 0x2013: '-', 0x2014: '-', 0x2212: '-',
        ord('/'): '-', ord('.'): '-',
    }
    s = s.str.translate(trans)
    dt = pd.to_datetime(s, errors='coerce', utc=True)
    return dt.dt.tz_convert(None).dt.floor('D')


def _read_spx_daily(spx_csv: Path) -> pd.DataFrame:
    """
    Read and standardize a daily S&P 500 price series from CSV.

    The loader accepts several common layouts:
      - Exactly two columns with header where first is 'Date' (second renamed to 'Price').
      - Two-column raw file without header (interpreted as 'Date', 'Price').
      - Files where the price column is named one of {'Close', 'Adj Close', 'Last', 'PX_LAST'}.

    The function:
      * trims/normalizes dates,
      * coerces price to numeric,
      * drops rows with missing Date or Price,
      * sorts by Date, and
      * returns only ['Date', 'Price'].

    Parameters
    ----------
    spx_csv : pathlib.Path
        Path to the SPX daily CSV.

    Returns
    -------
    pandas.DataFrame
        DataFrame with columns ['Date', 'Price'] sorted ascending by Date.

    Notes
    -----
    A brief range summary is printed if the dataframe is not empty.
    """
    df = pd.read_csv(spx_csv)
    df.columns = [str(c).strip() for c in df.columns]

    if len(df.columns) == 2 and df.columns[0].lower() == 'date':
        second = df.columns[1]
        df = df.rename(columns={second: 'Price'})
    else:
        # Try no-header case
        df = pd.read_csv(spx_csv, header=None, names=['Date', 'Price'])

    if 'Date' not in df.columns or 'Price' not in df.columns:
        for cand in [c for c in df.columns if str(c).lower() in ('close', 'adj close', 'last', 'px_last')]:
            df = df.rename(columns={cand: 'Price'})
            break

    df = df[['Date', 'Price']]
    df = df[df['Date'].astype(str).str.match(r'^\s*\d')]
    df['Date'] = _normalize_dates_for_spx(df['Date'])
    df['Price'] = pd.to_numeric(df['Price'], errors='coerce')
    df = df.dropna(subset=['Date', 'Price']).sort_values('Date').reset_index(drop=True)

    if not df.empty:
        print(f"[info] SPX range: {df['Date'].min().date()} -> {df['Date'].max().date()} rows={len(df)}")
    return df[['Date', 'Price']]


_MONTH_3 = {"jan": 1, "feb": 2, "mar": 3, "apr": 4, "may": 5, "jun": 6,
            "jul": 7, "aug": 8, "sep": 9, "oct": 10, "nov": 11, "dec": 12}
_MONTH_2 = {"ja": 1, "fe": 2, "ma": 3, "ap": 4, "my": 5, "jn": 6, "ju": 6,
            "jl": 7, "au": 8, "se": 9, "oc": 10, "no": 11, "de": 12}


def _parse_dividend_csv_with_century_boundary(div_file: Path) -> pd.DataFrame:
    """
    Parse a monthly dividend yield CSV and resolve ambiguous 2-digit years around 2000.

    Expected columns: 'Date', 'Value' where 'Value' is the dividend yield in percent
    (e.g., "1.82%"). Two input date formats are supported:

    1) ``dd-MMM-yy`` (e.g., "31-Dec-99"):
        - A "century boundary" is inferred from the **row order**:
          if the 2-digit year increases from one row to the next (e.g., 99 -> 00),
          rows **before** that boundary are mapped to 2000+yy, and rows **after** to 1900+yy.
          If no boundary is detected, a sensible default is used:
          years <= 30 -> 2000+yy, otherwise 1900+yy.

    2) Two-letter month + day + 4-digit year (e.g., "Ja120023"):
        - Month tokens are mapped by `_MONTH_2`:
            {'ja','fe','ma','ap','my','jn','ju','jl','au','se','oc','no','de'} -> 1..12
        - The rest is interpreted as day and 4-digit year.

    After parsing:
      * Rows are de-duplicated at Year-Month granularity (keeping the last occurrence).
      * The 'Value' percent is converted to a fraction in 'DivYield'.
      * Output is sorted by ParsedDate and reduced to ['YearMonth', 'DivYield'].

    Parameters
    ----------
    div_file : pathlib.Path
        Path to the dividend yield CSV.

    Returns
    -------
    pandas.DataFrame
        DataFrame with columns:
            - 'YearMonth' : str, e.g., '2024-10'
            - 'DivYield'  : float, monthly datapoint expressed as **annualized fraction**
                            (e.g., 0.0182 for 1.82%)

    Raises
    ------
    ValueError
        If required columns are missing.
    """
    df = pd.read_csv(div_file)
    if 'Date' not in df.columns or 'Value' not in df.columns:
        raise ValueError(f"Unexpected columns in {div_file.name}: {df.columns.tolist()}")

    df['row_id'] = range(len(df))
    dcol = df['Date'].astype(str).str.strip()

    # Pattern 1: dd-MMM-yy
    p1_mask = dcol.str.match(r"^\d{1,2}-[A-Za-z]{3}-\d{2}$")
    p1 = df.loc[p1_mask].copy()
    if not p1.empty:
        p1['day'] = p1['Date'].str.extract(r"^(\d{1,2})-")[0].astype(int)
        p1['mon'] = p1['Date'].str.extract(r"^\d{1,2}-([A-Za-z]{3})-")[0].str.lower().map(_MONTH_3)
        p1['yy'] = p1['Date'].str.extract(r"-(\d{2})$")[0].astype(int)

        yy = p1['yy'].to_numpy(); rid = p1['row_id'].to_numpy()
        boundary_row = None
        for i in range(1, len(yy)):
            if yy[i] > yy[i-1]:
                boundary_row = rid[i]; break

        def map_year(row) -> int:
            if boundary_row is None:
                return 2000 + int(row['yy']) if int(row['yy']) <= 30 else 1900 + int(row['yy'])
            return (2000 + int(row['yy'])) if (int(row['row_id']) < boundary_row) else (1900 + int(row['yy']))

        p1['year'] = p1.apply(map_year, axis=1)
        p1['ParsedDate'] = pd.to_datetime(dict(year=p1['year'], month=p1['mon'], day=p1['day']), errors='coerce')

    # Pattern 2: Two-letter month + day + 4-digit year
    p2_mask = ~p1_mask
    p2 = df.loc[p2_mask].copy()
    if not p2.empty:
        m = p2['Date'].str.extract(r"^([A-Z][a-z])(\d{1,2})(\d{4})$")
        valid = ~m.isna().any(axis=1)
        p2 = p2.loc[valid].copy()
        p2['mon'] = m.loc[valid, 0].str.lower().map(_MONTH_2)
        p2['day'] = m.loc[valid, 1].astype(int)
        p2['year'] = m.loc[valid, 2].astype(int)
        p2['ParsedDate'] = pd.to_datetime(dict(year=p2['year'], month=p2['mon'], day=p2['day']), errors='coerce')

    parsed = pd.concat([p1, p2], ignore_index=True, sort=False)
    parsed = parsed.dropna(subset=['ParsedDate']).copy()
    parsed['YearMonth'] = parsed['ParsedDate'].dt.to_period('M').astype(str)
    parsed['DivYield'] = parsed['Value'].astype(str).str.replace('%', '', regex=False).astype(float) / 100.0
    parsed = parsed.sort_values('ParsedDate').drop_duplicates(subset=['YearMonth'], keep='last')

    if not parsed.empty:
        print(f"[info] Dividend YM range: {parsed['YearMonth'].min()} -> {parsed['YearMonth'].max()} rows={len(parsed)}")
    return parsed[['YearMonth', 'DivYield']]


def _to_daily_series_from_monthly(monthly_df: pd.DataFrame, key: str, daily_dates: pd.Series) -> pd.Series:
    """
    Convert a monthly series into a daily series aligned to specific dates.

    Each daily date is mapped to the corresponding YYYY-MM of `monthly_df[key]`.
    Missing months are forward-filled then back-filled to cover all days.

    Parameters
    ----------
    monthly_df : pandas.DataFrame
        Must contain 'YearMonth' and the `key` column.
    key : str
        The column in `monthly_df` to map (e.g., 'DivYield').
    daily_dates : pandas.Series
        Target daily dates (datetime64).

    Returns
    -------
    pandas.Series
        A series indexed like `daily_dates` with values taken from the monthly series.
    """
    m = monthly_df.set_index('YearMonth')[key].copy()
    ym = pd.to_datetime(daily_dates, errors='coerce').dt.to_period('M').astype(str)
    out = ym.map(m).ffill().bfill()
    out.index = daily_dates.index
    return out


def build_spx_total_return(spx_df: pd.DataFrame, div_daily: pd.Series) -> pd.DataFrame:
    """
    Build a daily **Total Return** series from daily prices and a daily dividend-yield series.

    Steps
    -----
    1) Compute simple daily price returns.
    2) Compute the day fraction dt = Δdays / 365.0 (ACT/365).
    3) Convert the **annualized** dividend yield into a daily drift: drift = div_daily * dt.
    4) Combine: r_TR = r_price + drift.
    5) Compound (1 + r_TR) from the first day and scale by the first price.

    Parameters
    ----------
    spx_df : pandas.DataFrame
        Daily prices with columns ['Date', 'Price'] sorted ascending by Date.
    div_daily : pandas.Series
        Daily dividend yield (annualized fractional form), indexed like `spx_df`.

    Returns
    -------
    pandas.DataFrame
        A DataFrame with the same 'Date' and a 'Price' column representing the TR path
        before final end-point alignment to the spot price.
    """
    out = spx_df.copy()
    r_spx = out['Price'].pct_change().fillna(0.0)

    date = out['Date']
    dt = (date.diff().dt.days.fillna(0)).astype(float) / 365.0
    drift = div_daily.astype(float) * dt
    r_tr = r_spx + drift

    p0 = float(out['Price'].iloc[0])
    tr_price = (1.0 + r_tr).cumprod() * p0
    return pd.DataFrame({'Date': out['Date'], 'Price': tr_price})


def infer_output_name(start_ts, end_ts) -> str:
    """
    Construct an output filename from start and end timestamps.

    Parameters
    ----------
    start_ts, end_ts : Any
        Values accepted by `pandas.to_datetime`.

    Returns
    -------
    str
        A filename like ``^SPX_YYYY-MM-DD_YYYY-MM-DD_daily_TR.csv``.
    """
    start = pd.to_datetime(start_ts).date().isoformat()
    end = pd.to_datetime(end_ts).date().isoformat()
    return f"^SPX_{start}_{end}_daily_TR.csv"


def main(argv: list[str] | None = None) -> int:
    """
    CLI entry point to generate and save the SPX daily Total Return series.

    Parameters
    ----------
    argv : list of str, optional
        Command-line arguments (excluding the program name). If None, `argparse`
        will consume arguments from `sys.argv`.

    Returns
    -------
    int
        Zero on success; non-zero on error (raised exceptions will terminate the program).

    Command-line options
    --------------------
    --spx : str
        Path or wildcard for the daily price CSV (default: "data/^spx_d_*_to_*.csv").
    --dividend : str
        Path or wildcard for the monthly dividend yield CSV
        (default: "data/raw/SPX Dividend Yield by Month_*.csv").
    --start : str
        Optional inclusive start date (e.g., "1990-01-01").
    --end : str
        Optional inclusive end date (e.g., "2025-12-31").
    --outdir : str
        Output directory (default: "data").

    Side effects
    ------------
    - Writes the TR CSV file to `--outdir`.
    - Prints input resolutions, ranges, output path, and row/period summary.

    Notes
    -----
    The function attempts to align the final TR level to the last observed spot price.
    This preserves the relative TR evolution while making the series end at the
    known price level, which is often convenient for chart overlays.
    """
    ap = argparse.ArgumentParser(description="Generate S&P 500 daily Total Return series from daily (Date,Price) and monthly dividend yield.")
    ap.add_argument("--spx", dest="spx", default=str(Path("data") / "^spx_d_*_to_*.csv"))
    ap.add_argument("--dividend", dest="dividend", default=str(Path("data/raw") / "SPX Dividend Yield by Month_*.csv"))
    ap.add_argument("--start", type=str, default=None)
    ap.add_argument("--end", type=str, default=None)
    ap.add_argument("--outdir", type=str, default="data")
    args = ap.parse_args(argv)

    spx_path = resolve_latest(args.spx)
    div_path = resolve_latest(args.dividend)

    spx = _read_spx_daily(spx_path)
    if args.start:
        spx = spx[spx['Date'] >= pd.to_datetime(args.start)]
    if args.end:
        spx = spx[spx['Date'] <= pd.to_datetime(args.end)]
    spx = spx.reset_index(drop=True)

    if spx.empty:
        raise ValueError("SPX daily input has no rows after applying start/end filters.")

    div_m = _parse_dividend_csv_with_century_boundary(div_path)
    div_d = _to_daily_series_from_monthly(div_m, 'DivYield', spx['Date']).astype(float)

    tr_df = build_spx_total_return(spx, div_d)

    # Align end-point so TR last value equals the last observed spot price
    try:
        spx_last = float(spx['Price'].iloc[-1])
        tr_last = float(tr_df['Price'].iloc[-1])
        if tr_last != 0.0:
            scale = spx_last / tr_last
            tr_df['Price'] = tr_df['Price'] * scale
            tr_df.at[tr_df.index[-1], 'Price'] = spx_last
    except Exception as e:
        print(f"[warn] Align-final failed: {e}")

    period_start = tr_df['Date'].min(); period_end = tr_df['Date'].max(); rows = len(tr_df)
    outdir = Path(args.outdir); outdir.mkdir(parents=True, exist_ok=True)
    out_path = outdir / infer_output_name(period_start, period_end)

    def _format_dates_for_output(s: pd.Series) -> pd.Series:
        """
        Format dates using platform-appropriate directives (no leading zeros for month/day).

        On Windows, uses ``%Y/%#m/%#d``; on other systems, uses ``%Y/%-m/%-d``.
        Falls back to ``%Y/%m/%d`` if the platform-specific format fails.
        """
        dt = pd.to_datetime(s, errors='coerce')
        fmt = "%Y/%#m/%#d" if platform.system() == "Windows" else "%Y/%-m/%-d"
        try:
            return dt.dt.strftime(fmt)
        except Exception:
            return dt.dt.strftime("%Y/%m/%d")

    tr_df['Date'] = _format_dates_for_output(tr_df['Date'])
    tr_df.to_csv(out_path, index=False)

    start_iso = pd.to_datetime(period_start).date().isoformat()
    end_iso = pd.to_datetime(period_end).date().isoformat()
    print(f"Input SPX : {spx_path}")
    print(f"Input Div : {div_path}")
    print(f"Saved TR  : {out_path}")
    print(f"Rows: {rows}  Period: {start_iso} to {end_iso}")
    return 0


if __name__ == '__main__':
    raise SystemExit(main())