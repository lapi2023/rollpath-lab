# src/commands/generate_spx_total_return.py
from __future__ import annotations
import argparse
from pathlib import Path
import platform
import pandas as pd
import re

# (kept close to original; comments simplified to English)

def _contains_wildcard(s: str) -> bool:
    return any(ch in s for ch in '*?[]')


def _try_extract_end_token(p: Path) -> str:
    m = re.search(r'_to_(\d{6,8})', p.name)
    return m.group(1) if m else ''


def resolve_latest(spec: str) -> Path:
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
        return (1 if path.suffix.lower()=='.csv' else 0, end_tok, path.stat().st_mtime)
    best = max(cands, key=_score)
    print(f"[info] Resolved '{spec}' -> {best}")
    return best


def _normalize_dates_for_spx(s: pd.Series) -> pd.Series:
    s = s.astype(str).str.strip()
    trans = {ord('／'):'-', ord('．'):'-', ord('－'):'-',
             0x2010:'-',0x2011:'-',0x2012:'-',0x2013:'-',0x2014:'-',0x2212:'-',
             ord('/'):'-', ord('.'):'-'}
    s = s.str.translate(trans)
    dt = pd.to_datetime(s, errors='coerce', utc=True)
    return dt.dt.tz_convert(None).dt.floor('D')


def _read_spx_daily(spx_csv: Path) -> pd.DataFrame:
    df = pd.read_csv(spx_csv)
    df.columns = [str(c).strip() for c in df.columns]
    if len(df.columns) == 2 and df.columns[0].lower() == 'date':
        second = df.columns[1]
        df = df.rename(columns={second: 'Price'})
    else:
        df = pd.read_csv(spx_csv, header=None, names=['Date', 'Price'])
    if 'Date' not in df.columns or 'Price' not in df.columns:
        for cand in [c for c in df.columns if str(c).lower() in ('close','adj close','last','px_last')]:
            df = df.rename(columns={cand: 'Price'})
            break
    df = df[['Date','Price']]
    df = df[df['Date'].astype(str).str.match(r'^\s*\d')]
    df['Date'] = _normalize_dates_for_spx(df['Date'])
    df['Price'] = pd.to_numeric(df['Price'], errors='coerce')
    df = df.dropna(subset=['Date','Price']).sort_values('Date').reset_index(drop=True)
    if not df.empty:
        print(f"[info] SPX range: {df['Date'].min().date()} -> {df['Date'].max().date()} rows={len(df)}")
    return df[['Date','Price']]

_MONTH_3 = {"jan":1,"feb":2,"mar":3,"apr":4,"may":5,"jun":6,
            "jul":7,"aug":8,"sep":9,"oct":10,"nov":11,"dec":12}
_MONTH_2 = {"ja":1,"fe":2,"ma":3,"ap":4,"my":5,"jn":6,"ju":6,
            "jl":7,"au":8,"se":9,"oc":10,"no":11,"de":12}


def _parse_dividend_csv_with_century_boundary(div_file: Path) -> pd.DataFrame:
    df = pd.read_csv(div_file)
    if 'Date' not in df.columns or 'Value' not in df.columns:
        raise ValueError(f"Unexpected columns in {div_file.name}: {df.columns.tolist()}")
    df['row_id'] = range(len(df))
    dcol = df['Date'].astype(str).str.strip()
    p1_mask = dcol.str.match(r"^\d{1,2}-[A-Za-z]{3}-\d{2}$")
    p1 = df.loc[p1_mask].copy()
    if not p1.empty:
        p1['day'] = p1['Date'].str.extract(r"^(\d{1,2})-")[0].astype(int)
        p1['mon'] = p1['Date'].str.extract(r"^\d{1,2}-([A-Za-z]{3})-")[0].str.lower().map(_MONTH_3)
        p1['yy']  = p1['Date'].str.extract(r"-(\d{2})$")[0].astype(int)
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
    p2_mask = ~p1_mask
    p2 = df.loc[p2_mask].copy()
    if not p2.empty:
        m = p2['Date'].str.extract(r"^([A-Z][a-z])(\d{1,2})(\d{4})$")
        valid = ~m.isna().any(axis=1)
        p2 = p2.loc[valid].copy()
        p2['mon']  = m.loc[valid, 0].str.lower().map(_MONTH_2)
        p2['day']  = m.loc[valid, 1].astype(int)
        p2['year'] = m.loc[valid, 2].astype(int)
        p2['ParsedDate'] = pd.to_datetime(dict(year=p2['year'], month=p2['mon'], day=p2['day']), errors='coerce')
    parsed = pd.concat([p1, p2], ignore_index=True, sort=False)
    parsed = parsed.dropna(subset=['ParsedDate']).copy()
    parsed['YearMonth'] = parsed['ParsedDate'].dt.to_period('M').astype(str)
    parsed['DivYield'] = parsed['Value'].astype(str).str.replace('%','', regex=False).astype(float) / 100.0
    parsed = parsed.sort_values('ParsedDate').drop_duplicates(subset=['YearMonth'], keep='last')
    if not parsed.empty:
        print(f"[info] Dividend YM range: {parsed['YearMonth'].min()} -> {parsed['YearMonth'].max()} rows={len(parsed)}")
    return parsed[['YearMonth','DivYield']]


def _to_daily_series_from_monthly(monthly_df: pd.DataFrame, key: str, daily_dates: pd.Series) -> pd.Series:
    m = monthly_df.set_index('YearMonth')[key].copy()
    ym = pd.to_datetime(daily_dates, errors='coerce').dt.to_period('M').astype(str)
    out = ym.map(m).ffill().bfill()
    out.index = daily_dates.index
    return out


def build_spx_total_return(spx_df: pd.DataFrame, div_daily: pd.Series) -> pd.DataFrame:
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
    start = pd.to_datetime(start_ts).date().isoformat()
    end = pd.to_datetime(end_ts).date().isoformat()
    return f"^SPX_{start}_{end}_daily_TR.csv"


def main(argv: list[str] | None = None) -> int:
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

    try:
        spx_last = float(spx['Price'].iloc[-1])
        tr_last  = float(tr_df['Price'].iloc[-1])
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
        dt = pd.to_datetime(s, errors='coerce')
        fmt = "%Y/%#m/%#d" if platform.system()=="Windows" else "%Y/%-m/%-d"
        try:
            return dt.dt.strftime(fmt)
        except Exception:
            return dt.dt.strftime("%Y/%m/%d")

    tr_df['Date'] = _format_dates_for_output(tr_df['Date'])
    tr_df.to_csv(out_path, index=False)
    start_iso = pd.to_datetime(period_start).date().isoformat()
    end_iso   = pd.to_datetime(period_end).date().isoformat()
    print(f"Input SPX : {spx_path}")
    print(f"Input Div : {div_path}")
    print(f"Saved TR  : {out_path}")
    print(f"Rows: {rows}  Period: {start_iso} to {end_iso}")
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
