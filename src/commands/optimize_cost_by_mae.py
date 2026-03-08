
# src/commands/optimize_cost_by_mae.py
from __future__ import annotations
import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple, Dict
import numpy as np
import pandas as pd

try:
    from src.settings import DATA_DIR
except Exception:
    DATA_DIR = Path('data')

from src.commands.generate_leveraged_etf import (
    simulate_lever_price,
    read_spx_daily,
    dividends_tag,
)
from src.dividend_loader import (
    DividendInputPaths,
    load_monthly_dividend_and_riskfree,
    to_daily_series_from_monthly,
)

# ---- I/O helpers ----

def _find_latest(glob_pattern: str) -> Path:
    cands = list(DATA_DIR.glob(glob_pattern))
    if not cands:
        raise FileNotFoundError("No files matched: data/" + glob_pattern)
    return max(cands, key=lambda p: p.stat().st_mtime)


@dataclass
class ActualMeta:
    path: Path
    tag: Optional[str]


def _read_actual_yf(p: Path) -> pd.DataFrame:
    df = pd.read_csv(p)
    if "Date" not in df.columns:
        raise ValueError("'Date' column not found in " + p.name)
    price_col = None
    for cand in ("Adj Close", "Close"):
        if cand in df.columns:
            price_col = cand
            break
    if price_col is None:
        lc = {c.lower(): c for c in df.columns}
        for cand in ("adj close", "close"):
            if cand in lc:
                price_col = lc[cand]
                break
    if price_col is None:
        raise ValueError("Neither 'Adj Close' nor 'Close' found in " + p.name)
    dt = pd.to_datetime(df["Date"], errors="coerce", utc=True)
    df["Date"] = dt.dt.tz_convert(None).dt.floor("D")
    df[price_col] = pd.to_numeric(df[price_col], errors="coerce")
    df = df.dropna(subset=["Date", price_col]).sort_values("Date").reset_index(drop=True)
    return df[["Date", price_col]].rename(columns={price_col: "Price"})

# ---- losses ----

def _mae_loss(sim_norm: np.ndarray, act_norm: np.ndarray) -> float:
    return float(np.mean(np.abs(sim_norm / act_norm - 1.0) * 100.0))


def _huber_loss(sim_norm: np.ndarray, act_norm: np.ndarray, delta: float = 1.0) -> float:
    r = (sim_norm / act_norm - 1.0) * 100.0
    abs_r = np.abs(r)
    quad = np.minimum(abs_r, delta)
    lin = abs_r - quad
    return float(np.mean(0.5 * quad**2 + delta * lin))


def _trimmed_mae_loss(sim_norm: np.ndarray, act_norm: np.ndarray, trim: float = 0.02) -> float:
    r = np.sort(np.abs((sim_norm / act_norm - 1.0) * 100.0))
    n = len(r)
    if n == 0:
        return float('nan')
    k = int(n * trim)
    if 2 * k >= n:
        return float(np.mean(r))
    return float(np.mean(r[k:n-k]))

# ---- optimizer primitives ----

def _golden_section_minimize(f, lo: float, hi: float, tol: float = 1e-7, max_iter: int = 200) -> Tuple[float, float]:
    phi = (1 + 5 ** 0.5) / 2
    invphi = 1 / phi
    a, b = float(lo), float(hi)
    c = b - invphi * (b - a)
    d = a + invphi * (b - a)
    fc, fd = f(c), f(d)
    it = 0
    while it < max_iter and abs(b - a) > tol * (abs(c) + abs(d) + 1e-12):
        if fc < fd:
            b, d, fd = d, c, fc
            c = b - invphi * (b - a)
            fc = f(c)
        else:
            a, c, fc = c, d, fd
            d = a + invphi * (b - a)
            fd = f(d)
        it += 1
    x = (a + b) / 2
    return x, f(x)

# ---- simulation & objective ----

def _simulate_for_symbol(
    symbol: str,
    spx_df: pd.DataFrame,
    div_d: pd.Series,
    rf_d: pd.Series,
    alpha: float,
    beta: float,
    cost: float,
    include_dividends: bool,
    start: Optional[pd.Timestamp],
    end: Optional[pd.Timestamp],
) -> pd.DataFrame:
    L = 3 if symbol.upper() == "SPXL" else 2
    x = spx_df
    if start is not None:
        x = x[x["Date"] >= start]
    if end is not None:
        x = x[x["Date"] <= end]
    x = x.reset_index(drop=True)
    div_y = div_d.reindex(x.index)
    rf_y = rf_d.reindex(x.index)
    sim = simulate_lever_price(
        spx=x, div_y=div_y, rf_y=rf_y, L=L, cost_annual=cost,
        include_dividends=include_dividends, base_from_spx=True,
        borrow_alpha=alpha, borrow_beta=beta,
    )
    return sim


def _align_and_normalize(df_actual: pd.DataFrame, df_sim: pd.DataFrame) -> pd.DataFrame:
    a = df_actual.rename(columns={"Price": "actual"})
    s = df_sim.rename(columns={"Price": "sim"})
    m = pd.merge(a, s, on="Date", how="inner")
    if m.empty:
        raise ValueError("No overlapping dates after alignment.")
    m = m.sort_values("Date").reset_index(drop=True)
    m["norm_actual"] = m["actual"] / float(m["actual"].iloc[0])
    m["norm_sim"] = m["sim"] / float(m["sim"].iloc[0])
    return m


def _build_objective(
    symbol: str,
    df_actual: pd.DataFrame,
    spx_df: pd.DataFrame,
    div_d: pd.Series,
    rf_d: pd.Series,
    include_dividends: bool,
    start: Optional[pd.Timestamp],
    end: Optional[pd.Timestamp],
    loss: str = "mae",
    huber_delta: float = 1.0,
    trimmed: float = 0.02,
):
    def _loss_fn(alpha: float, beta: float, cost: float) -> float:
        sim = _simulate_for_symbol(
            symbol=symbol, spx_df=spx_df, div_d=div_d, rf_d=rf_d,
            alpha=alpha, beta=beta, cost=cost,
            include_dividends=include_dividends, start=start, end=end,
        )
        merged = _align_and_normalize(df_actual, sim)
        if loss == "mae":
            return _mae_loss(merged["norm_sim"].to_numpy(), merged["norm_actual"].to_numpy())
        elif loss == "huber":
            return _huber_loss(merged["norm_sim"].to_numpy(), merged["norm_actual"].to_numpy(), delta=huber_delta)
        elif loss == "trimmed":
            return _trimmed_mae_loss(merged["norm_sim"].to_numpy(), merged["norm_actual"].to_numpy(), trim=trimmed)
        else:
            return _mae_loss(merged["norm_sim"].to_numpy(), merged["norm_actual"].to_numpy())
    return _loss_fn

# ---- coordinate-descent wrapper ----

def _coordinate_descent(
    objective,
    alpha_lo: float, alpha_hi: float,
    beta_lo: float, beta_hi: float,
    cost_lo: float, cost_hi: float,
    alpha0: float = 1.0,
    beta0: float = 0.0,
    cost0: float = 0.02,
    tol: float = 1e-5,
    max_outer: int = 12,
) -> Tuple[Tuple[float, float, float], float]:
    alpha = float(np.clip(alpha0, alpha_lo, alpha_hi))
    beta  = float(np.clip(beta0,  beta_lo,  beta_hi))
    cost  = float(np.clip(cost0,  cost_lo,  cost_hi))
    best_obj = objective(alpha, beta, cost)
    for _ in range(max_outer):
        def f_cost(c):
            return objective(alpha, beta, float(c))
        cost, _ = _golden_section_minimize(f_cost, cost_lo, cost_hi, tol=1e-7, max_iter=200)

        def f_alpha(a):
            return objective(float(a), beta, cost)
        alpha, _ = _golden_section_minimize(f_alpha, alpha_lo, alpha_hi, tol=1e-7, max_iter=200)

        def f_beta(b):
            return objective(alpha, float(b), cost)
        beta, _ = _golden_section_minimize(f_beta, beta_lo, beta_hi, tol=1e-7, max_iter=200)

        new_obj = objective(alpha, beta, cost)
        if abs(best_obj - new_obj) < tol:
            best_obj = new_obj
            break
        best_obj = new_obj
    return (float(alpha), float(beta), float(cost)), float(best_obj)

# ---- end-to-end per symbol ----

def _optimize_for_symbol(
    symbol: str,
    include_dividends: bool,
    spx_csv: Path,
    dividend_csv: Path,
    tbill_1920_1934_csv: Path,
    tb3ms_1934_now_csv: Path,
    alpha_lo: float, alpha_hi: float,
    beta_lo: float, beta_hi: float,
    cost_lo: float, cost_hi: float,
    loss: str, huber_delta: float, trimmed: float,
    start: Optional[str], end: Optional[str],
    alpha0: float, beta0: float, cost0: float,
    save_best_sim: bool,
) -> Tuple[Tuple[float, float, float], float, pd.DataFrame]:
    actual_glob = symbol + "_*_daily_*.csv"
    actual_path = _find_latest(actual_glob)
    df_actual = _read_actual_yf(actual_path)

    ts_start = pd.to_datetime(start) if start else None
    ts_end   = pd.to_datetime(end) if end else None
    if ts_start is not None:
        df_actual = df_actual[df_actual["Date"] >= ts_start]
    if ts_end is not None:
        df_actual = df_actual[df_actual["Date"] <= ts_end]
    if df_actual.empty:
        raise ValueError("Actual series is empty after applying the date window.")

    spx_df = read_spx_daily(spx_csv)
    paths = DividendInputPaths(
        sp500_dividend_monthly_csv=dividend_csv,
        tbill_1920_1934_monthly_csv=tbill_1920_1934_csv,
        tb3ms_1934_now_monthly_csv=tb3ms_1934_now_csv,
    )
    div_m, rf_m = load_monthly_dividend_and_riskfree(paths)
    div_d = to_daily_series_from_monthly(div_m, "DivYield", spx_df["Date"]) 
    rf_d  = to_daily_series_from_monthly(rf_m,  "RiskFree", spx_df["Date"]) 

    obj = _build_objective(
        symbol=symbol, df_actual=df_actual, spx_df=spx_df, div_d=div_d, rf_d=rf_d,
        include_dividends=include_dividends, start=ts_start, end=ts_end,
        loss=loss, huber_delta=huber_delta, trimmed=trimmed,
    )

    cost_lo_eff = max(cost_lo, 0.0087)
    cost0_eff = max(cost0, 0.0087)

    (alpha, beta, cost), best_obj = _coordinate_descent(
        objective=obj,
        alpha_lo=alpha_lo, alpha_hi=alpha_hi,
        beta_lo=beta_lo,   beta_hi=beta_hi,
        cost_lo=cost_lo_eff, cost_hi=cost_hi,
        alpha0=alpha0, beta0=beta0, cost0=cost0_eff,
        tol=1e-5, max_outer=12,
    )

    sim_best = _simulate_for_symbol(
        symbol=symbol, spx_df=spx_df, div_d=div_d, rf_d=rf_d,
        alpha=alpha, beta=beta, cost=cost, include_dividends=include_dividends,
        start=ts_start, end=ts_end,
    )
    merged = _align_and_normalize(df_actual, sim_best)

    if save_best_sim:
        outdir = DATA_DIR
        div_tag = dividends_tag(include_dividends)
        a = ("{:.2f}".format(alpha)).replace(".00", "")
        b = ("{:.2f}%".format(beta * 100)).replace(".00%", "%")
        start_iso = pd.to_datetime(sim_best['Date'].min()).date().isoformat()
        end_iso   = pd.to_datetime(sim_best['Date'].max()).date().isoformat()
        prefix = "^spxl_simulated_d" if symbol.upper()=="SPXL" else "^sso_simulated_d"
        outname = prefix + "_" + div_tag + "_" + ("{:.2f}%cost".format(cost*100)).replace(".00", "") + "_a" + a + "_b" + b + "_" + start_iso + "_" + end_iso + ".csv"
        outpath = outdir / outname
        sim_best.to_csv(outpath, index=False)
        print("Saved best simulated series: " + str(outpath))

    return (alpha, beta, cost), best_obj, merged


def main() -> None:
    ap = argparse.ArgumentParser(description="Optimize alpha/beta/fixed-cost by minimizing Actual-vs-Simulated divergence.")
    ap.add_argument("--symbols", type=str, nargs="+", default=["SPXL", "SSO"], choices=["SPXL", "SSO"])
    ap.add_argument("--alpha-lo", type=float, default=1.0)
    ap.add_argument("--alpha-hi", type=float, default=1.5)
    ap.add_argument("--beta-lo",  type=float, default=0.0)
    ap.add_argument("--beta-hi",  type=float, default=0.03)
    ap.add_argument("--cost-lo",  type=float, default=0.0087)
    ap.add_argument("--cost-hi",  type=float, default=0.05)
    ap.add_argument("--loss", type=str, default="mae", choices=["mae","huber","trimmed"]) 
    ap.add_argument("--huber-delta", type=float, default=1.0, dest="huber_delta")
    ap.add_argument("--trim", type=float, default=0.02)
    ap.add_argument("--start", type=str, default=None)
    ap.add_argument("--end",   type=str, default=None)
    ap.add_argument("--include-dividends", action="store_true")
    ap.add_argument("--no-dividends", action="store_true")
    ap.add_argument("--spx-csv",         type=str, default=str(_find_latest("^spx_d_*.csv")))
    ap.add_argument("--dividend-csv",    type=str, default=str(_find_latest("raw/SPX Dividend Yield by Month_*.csv")))
    ap.add_argument("--tbill-1920-1934-csv", type=str, default=str(_find_latest("raw/Yields on Short-Term United States Securities*.csv")))
    ap.add_argument("--tb3ms-1934-now-csv",   type=str, default=str(_find_latest("raw/3-Month Treasury Bill Secondary Market Rate, Discount Basis (TB3MS)*.csv")))
    ap.add_argument("--alpha0", type=float, default=1.0)
    ap.add_argument("--beta0",  type=float, default=0.005)
    ap.add_argument("--cost0",  type=float, default=0.02)
    ap.add_argument("--save-best-sim", action="store_true")
    args = ap.parse_args()

    include_dividends = True
    if args.include_dividends:
        include_dividends = True
    if args.no_dividends:
        include_dividends = False

    spx_csv = Path(args.spx_csv)
    if not str(spx_csv).startswith('data/'):
        spx_csv = DATA_DIR / args.spx_csv
    dividend_csv = Path(args.dividend_csv)
    if not str(dividend_csv).startswith('data/'):
        dividend_csv = DATA_DIR / args.dividend_csv
    tbill_1920_1934_csv = Path(args.tbill_1920_1934_csv)
    if not str(tbill_1920_1934_csv).startswith('data/'):
        tbill_1920_1934_csv = DATA_DIR / args.tbill_1920_1934_csv
    tb3ms_1934_now_csv = Path(args.tb3ms_1934_now_csv)
    if not str(tb3ms_1934_now_csv).startswith('data/'):
        tb3ms_1934_now_csv = DATA_DIR / args.tb3ms_1934_now_csv

    print("=== Joint optimization by coordinate-descent (alpha, beta, cost) ===")
    results: Dict[str, Tuple[Tuple[float,float,float], float]] = {}
    for sym in args.symbols:
        print("-- " + sym + " --")
        best, obj, _merged = _optimize_for_symbol(
            symbol=sym, include_dividends=include_dividends,
            spx_csv=spx_csv, dividend_csv=dividend_csv,
            tbill_1920_1934_csv=tbill_1920_1934_csv, tb3ms_1934_now_csv=tb3ms_1934_now_csv,
            alpha_lo=args.alpha_lo, alpha_hi=args.alpha_hi,
            beta_lo=args.beta_lo,   beta_hi=args.beta_hi,
            cost_lo=args.cost_lo,   cost_hi=args.cost_hi,
            loss=args.loss, huber_delta=args.huber_delta, trimmed=args.trim,
            start=args.start, end=args.end,
            alpha0=args.alpha0, beta0=args.beta0, cost0=args.cost0,
            save_best_sim=args.save_best_sim,
        )
        (alpha, beta, cost) = best
        msg = (
            sym + ": "
            + "alpha=" + "{:.6f}".format(alpha) + ", "
            + "beta=" + "{:.4f}%/yr".format(beta * 100) + ", "
            + "cost=" + "{:.3f}%/yr".format(cost * 100) + " (loss=" + "{:.3f}".format(obj) + ")"
        )
        print(msg)
        results[sym] = (best, obj)

    print("=== Recommended parameters ===")
    for sym, ((alpha, beta, cost), obj) in results.items():
        rec = (
            sym + ": "
            + "alpha=" + "{:.6f}".format(alpha) + ", "
            + "beta=" + "{:.4f}%/yr".format(beta * 100) + ", "
            + "cost=" + "{:.3f}%/yr".format(cost * 100) + " (loss=" + "{:.3f}".format(obj) + ")"
        )
        print(rec)


if __name__ == "__main__":
    main()
