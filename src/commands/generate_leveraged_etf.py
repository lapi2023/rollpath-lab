# src/commands/generate_leveraged_etf.py
"""Generate simulated daily price series for leveraged S&P 500 ETFs.

This module provides a command-line entry point that creates synthetic daily
historical price series for leveraged S&P 500 ETFs such as:

* ``SSO`` (2x leverage)
* ``SPXL`` (3x leverage)

The simulation combines:

* a daily S&P 500 price series,
* monthly S&P 500 dividend yield data converted to daily frequency, and
* monthly short-term risk-free rate data converted to daily frequency.

The simulated return stream applies leveraged daily exposure to the underlying
index, optionally includes dividend carry, subtracts financing/borrowing costs
for the leveraged portion, and subtracts an annual fixed cost assumption.

The resulting CSV outputs are intended for research, backtesting, and long-run
historical reconstruction where the live ETF history is shorter than the period
being analyzed.

Notes
-----
* This model assumes **daily reset leverage**.
* Borrowing cost is modeled as ``borrow_alpha * risk_free + borrow_beta``.
* Output filenames are self-describing and encode dividend mode, annual cost,
  borrow parameters, and covered date range.
"""
from __future__ import annotations
import argparse
from pathlib import Path
import sys
import pandas as pd
try:
 from src import settings # noqa: F401
except Exception:
 class settings: # fallbacks
  """Fallback configuration used when ``src.settings`` is unavailable.

  These defaults allow the script to run in a reduced environment, such as a
  test harness or a standalone execution context where the project settings
  module is not present.
  """
  INCLUDE_DIVIDENDS = True
  BASE_FROM_SPX = True
  BORROW_ALPHA = 1.0
  BORROW_BETA = 0.0
from src.commands.generate_spx_total_return import infer_output_name # optional reuse
from src.dividend_loader import (
 DividendInputPaths,
 load_monthly_dividend_and_riskfree,
 to_daily_series_from_monthly,
)
# --- keep the original implementation (shortened header & English comments) ---
from typing import Optional
import math
import numpy as np
def _find_latest(path_glob: str) -> Path:
 """Return the most recently modified file that matches a glob pattern.

 Parameters
 ----------
 path_glob:
  Glob pattern evaluated relative to the current working directory.

 Returns
 -------
 pathlib.Path
  The newest matching file based on modification time.

 Raises
 ------
 FileNotFoundError
  If no files match the supplied pattern.
 """
 cands = list(Path(".").glob(path_glob))
 if not cands:
  raise FileNotFoundError(f"No file matched: {path_glob}")
 return max(cands, key=lambda p: p.stat().st_mtime)
def read_spx_daily(spx_csv: Path) -> pd.DataFrame:
 """Load and normalize a daily S&P 500 price CSV.

 The function accepts a small range of CSV layouts and normalizes the result
 to a two-column DataFrame containing ``Date`` and ``Price``. If the input has
 exactly two columns and no explicit ``Date`` column, the function assumes the
 first column contains dates and the second contains prices.

 Parameters
 ----------
 spx_csv:
  Path to the CSV file containing daily S&P 500 observations.

 Returns
 -------
 pandas.DataFrame
  A cleaned DataFrame with columns:

  * ``Date``: parsed pandas timestamps
  * ``Price``: numeric price/index values

 Raises
 ------
 ValueError
  If a date column or usable price column cannot be identified.
 """
 df = pd.read_csv(spx_csv)
 if "Date" not in df.columns and df.shape[1] == 2:
  df = df.rename(columns={df.columns[0]: "Date", df.columns[1]: "Price"})
 if "Date" not in df.columns:
  raise ValueError(f"'Date' column not found in {spx_csv.name}.")
 price_col = None
 if "Price" in df.columns:
  price_col = "Price"
 elif "Close" in df.columns:
  price_col = "Close"
 else:
  lc = {c.lower(): c for c in df.columns}
  for cand in ("price", "close"):
   if cand in lc:
    price_col = lc[cand]
    break
 if price_col is None:
  raise ValueError("Could not detect a price column.")
 df = df[df["Date"].astype(str).str.match(r"^\s*\d")]
 df["Date"] = pd.to_datetime(df["Date"])
 df[price_col] = pd.to_numeric(df[price_col], errors="coerce")
 df = df.dropna(subset=["Date", price_col]).sort_values("Date").reset_index(drop=True)
 return df[["Date", price_col]].rename(columns={price_col: "Price"})
def annual_to_daily_additive(rate_annual: pd.Series, dt_years: pd.Series) -> pd.Series:
 """Convert annualized additive rates into step-level additive returns.

 Parameters
 ----------
 rate_annual:
  Annualized rate series in decimal form.
 dt_years:
  Length of each observation interval in years.

 Returns
 -------
 pandas.Series
  Additive return contribution for each time step, computed as
  ``rate_annual * dt_years``.
 """
 return rate_annual * dt_years
def simulate_lever_price(
 spx: pd.DataFrame,
 div_y: pd.Series,
 rf_y: pd.Series,
 L: int,
 cost_annual: float,
 include_dividends: bool,
 base_from_spx: bool,
 borrow_alpha: float,
 borrow_beta: float,
) -> pd.DataFrame:
 """Simulate a daily-reset leveraged ETF price series.

 Parameters
 ----------
 spx:
  Daily S&P 500 DataFrame with columns ``Date`` and ``Price``.
 div_y:
  Daily dividend yield series aligned to the S&P 500 dates, expressed as an
  annualized decimal yield.
 rf_y:
  Daily risk-free rate series aligned to the S&P 500 dates, expressed as an
  annualized decimal rate.
 L:
  Daily leverage multiplier (for example, ``2`` for SSO or ``3`` for SPXL).
 cost_annual:
  Annual fixed cost/expense ratio in decimal form.
 include_dividends:
  Whether to include dividend carry in the simulated return stream.
 base_from_spx:
  Whether to initialize the simulated ETF level from the first S&P 500 price
  rather than from ``1.0``.
 borrow_alpha:
  Multiplier applied to the risk-free rate when computing financing cost.
 borrow_beta:
  Additional annualized spread added to the borrowing cost.

 Returns
 -------
 pandas.DataFrame
  A DataFrame with columns ``Date`` and ``Price`` representing the simulated
  leveraged ETF path.

 Notes
 -----
 The model computes the per-step simulated return as leveraged S&P 500 return
 plus carry/financing drift minus annual fee drag converted to the step size.
 This is a simplified reconstruction model and does not attempt to replicate
 every real-world source of ETF tracking error.
 """
 out = spx.copy()
 r_spx = out["Price"].pct_change().fillna(0.0)
 date = out["Date"]
 dt = (date.diff().dt.days.fillna(0)).astype(float) / 365.0
 div_ann = div_y.astype(float)
 rf_ann = rf_y.astype(float)
 borrow_ann = borrow_alpha * rf_ann + borrow_beta
 if include_dividends:
  carry = L * div_ann - (L - 1) * borrow_ann
 else:
  carry = -(L - 1) * borrow_ann
 drift = annual_to_daily_additive(carry, dt)
 fee = annual_to_daily_additive(pd.Series(cost_annual, index=dt.index), dt)
 r_sim = L * r_spx + drift - fee
 p0 = float(out["Price"].iloc[0]) if base_from_spx else 1.0
 price_sim = (1.0 + r_sim).cumprod() * p0
 return pd.DataFrame({"Date": out["Date"], "Price": price_sim})
def dividends_tag(include_dividends: bool) -> str:
 """Return a short output tag describing dividend treatment.

 Parameters
 ----------
 include_dividends:
  Whether dividends are included in the simulation.

 Returns
 -------
 str
  ``"TR"`` when dividends are included, otherwise ``"PX"``.
 """
 return "TR" if include_dividends else "PX"
def borrow_tag(alpha: float, beta: float) -> str:
 """Format borrow parameters as a compact filename tag.

 Parameters
 ----------
 alpha:
  Borrow-rate multiplier applied to the risk-free rate.
 beta:
  Extra annualized borrow spread in decimal form.

 Returns
 -------
 str
  Encoded tag such as ``a1_b0%``.
 """
 b = f"{beta * 100:.2f}%".replace(".00%", "%")
 a = f"{alpha:.2f}".replace(".00", "")
 return f"a{a}_b{b}"
def resolve_param(cli_value: float | None, settings_attr: str, fallback: float) -> float:
 """Resolve a numeric parameter from CLI input, settings, or fallback.

 Resolution order is:

 1. Explicit command-line argument
 2. Corresponding attribute in ``src.settings``
 3. Hard-coded fallback value

 Parameters
 ----------
 cli_value:
  Value passed from the command line, if any.
 settings_attr:
  Name of the settings attribute to look up.
 fallback:
  Value used when neither CLI nor settings provides a parameter.

 Returns
 -------
 float
  The resolved numeric value.
 """
 if cli_value is not None:
  return float(cli_value)
 try:
  import src.settings as _s
  v = getattr(_s, settings_attr, None)
  if v is not None:
   return float(v)
 except Exception:
  pass
 return float(fallback)
def resolve_param_multi(
 cli_value_symbol: float | None,
 settings_attr_symbol: str,
 cli_value_common: float | None,
 settings_attr_common: str,
 fallback: float,
) -> float:
 """Resolve symbol-specific parameters with common fallback levels.

 This helper supports a multi-level precedence chain, allowing symbol-specific
 settings (for example, SPXL-only borrow assumptions) to override common values.

 Parameters
 ----------
 cli_value_symbol:
  Symbol-specific value provided via CLI.
 settings_attr_symbol:
  Symbol-specific settings attribute name.
 cli_value_common:
  Shared/common CLI value.
 settings_attr_common:
  Shared/common settings attribute name.
 fallback:
  Final default if no other source provides a value.

 Returns
 -------
 float
  The resolved parameter value.
 """
 if cli_value_symbol is not None:
  return float(cli_value_symbol)
 try:
  from src import settings as _s
  v = getattr(_s, settings_attr_symbol, None)
  if v is not None:
   return float(v)
 except Exception:
  pass
 if cli_value_common is not None:
  return float(cli_value_common)
 try:
  from src import settings as _s
  v = getattr(_s, settings_attr_common, None)
  if v is not None:
   return float(v)
 except Exception:
  pass
 return float(fallback)
def main(argv: list[str]) -> int:
 """Run the leveraged ETF generation workflow from command-line arguments.

 Parameters
 ----------
 argv:
  Command-line arguments excluding the program name.

 Returns
 -------
 int
  Process exit code. Returns ``0`` on success.

 Side Effects
 ------------
 * Reads source CSV files from disk.
 * Expands monthly dividend and risk-free inputs to daily series.
 * Simulates SSO and SPXL price paths.
 * Writes output CSV files into the requested output directory.
 * Prints saved file paths and effective model assumptions.
 """
 ap = argparse.ArgumentParser(description="Generate simulated SSO (2x) and SPXL (3x) series.")
 ap.add_argument("--spx-csv", default=str(_find_latest("data/^spx_d_*.csv")))
 ap.add_argument("--dividend-csv", default=str(_find_latest("data/raw/SPX Dividend Yield by Month_*.csv")))
 ap.add_argument("--tbill-1920-1934-csv", default=str(_find_latest("data/raw/Yields on Short-Term United States Securities*.csv")))
 ap.add_argument("--tb3ms-1934-now-csv", default=str(_find_latest("data/raw/*TB3MS*.csv")))
 ap.add_argument("--cost-spxl", type=float, default=None)
 ap.add_argument("--cost-sso", type=float, default=None)
 ap.add_argument("--include-dividends", action="store_true")
 ap.add_argument("--no-dividends", action="store_true")
 ap.add_argument("--borrow-alpha", type=float, default=None)
 ap.add_argument("--borrow-beta", type=float, default=None)
 ap.add_argument("--borrow-alpha-spxl", type=float, default=None)
 ap.add_argument("--borrow-beta-spxl", type=float, default=None)
 ap.add_argument("--borrow-alpha-sso", type=float, default=None)
 ap.add_argument("--borrow-beta-sso", type=float, default=None)
 ap.add_argument("--start", type=str, default="1920-01-01")
 ap.add_argument("--end", type=str, default=None)
 ap.add_argument("--outdir", type=str, default="data")
 args = ap.parse_args(argv)
 include_dividends = settings.INCLUDE_DIVIDENDS
 if args.include_dividends:
  include_dividends = True
 if args.no_dividends:
  include_dividends = False
 base_from_spx = settings.BASE_FROM_SPX
 spx = read_spx_daily(Path(args.spx_csv))
 if args.start:
  spx = spx[spx["Date"] >= pd.to_datetime(args.start)]
 if args.end:
  spx = spx[spx["Date"] <= pd.to_datetime(args.end)]
 spx = spx.reset_index(drop=True)
 paths = DividendInputPaths(
  sp500_dividend_monthly_csv=Path(args.dividend_csv),
  tbill_1920_1934_monthly_csv=Path(args.tbill_1920_1934_csv),
  tb3ms_1934_now_monthly_csv=Path(args.tb3ms_1934_now_csv),
 )
 div_m, rf_m = load_monthly_dividend_and_riskfree(paths)
 div_d = to_daily_series_from_monthly(div_m, "DivYield", spx["Date"])
 rf_d = to_daily_series_from_monthly(rf_m, "RiskFree", spx["Date"])
 cost_spxl = resolve_param(args.cost_spxl, "SPXL_FIXED_COST", 0.0215)
 cost_sso = resolve_param(args.cost_sso, "SSO_FIXED_COST", 0.00215)
 alpha_spxl = resolve_param_multi(args.borrow_alpha_spxl, "SPXL_BORROW_ALPHA", args.borrow_alpha, "BORROW_ALPHA", 1.0)
 beta_spxl = resolve_param_multi(args.borrow_beta_spxl, "SPXL_BORROW_BETA", args.borrow_beta, "BORROW_BETA", 0.0)
 alpha_sso = resolve_param_multi(args.borrow_alpha_sso, "SSO_BORROW_ALPHA", args.borrow_alpha, "BORROW_ALPHA", 1.0)
 beta_sso = resolve_param_multi(args.borrow_beta_sso, "SSO_BORROW_BETA", args.borrow_beta, "BORROW_BETA", 0.0)
 sso = simulate_lever_price(spx, div_d, rf_d, L=2, cost_annual=cost_sso, include_dividends=include_dividends,
 base_from_spx=base_from_spx, borrow_alpha=alpha_sso, borrow_beta=beta_sso)
 spxl = simulate_lever_price(spx, div_d, rf_d, L=3, cost_annual=cost_spxl, include_dividends=include_dividends,
 base_from_spx=base_from_spx, borrow_alpha=alpha_spxl, borrow_beta=beta_spxl)
 outdir = Path(args.outdir); outdir.mkdir(parents=True, exist_ok=True)
 def cost_tag(v: float) -> str:
  """Format annual cost as a compact filename fragment."""
  return f"{(v*100):.2f}%cost".replace(".00", "")
 div_tag = dividends_tag(include_dividends)
 brw_tag_spxl = borrow_tag(alpha_spxl, beta_spxl)
 brw_tag_sso = borrow_tag(alpha_sso, beta_sso)
 start_iso = pd.to_datetime(spxl['Date'].min()).date().isoformat()
 end_iso = pd.to_datetime(spxl['Date'].max()).date().isoformat()
 spxl_path = outdir / f"^spxl_simulated_d_{div_tag}_{cost_tag(cost_spxl)}_{brw_tag_spxl}_{start_iso}_{end_iso}.csv"
 sso_path = outdir / f"^sso_simulated_d_{div_tag}_{cost_tag(cost_sso)}_{brw_tag_sso}_{start_iso}_{end_iso}.csv"
 spxl.to_csv(spxl_path, index=False)
 sso.to_csv(sso_path, index=False)
 print(f"Saved: {spxl_path}")
 print(f"Saved: {sso_path}")
 print(
  f"INCLUDE_DIVIDENDS={include_dividends} BASE_FROM_SPX={base_from_spx} "
  f"COST_SPXL={cost_spxl*100:.2f}% COST_SSO={cost_sso*100:.2f}% "
  f"[SPXL] ALPHA={alpha_spxl:.6f} BETA={beta_spxl*100:.4f}%/yr "
  f"[SSO] ALPHA={alpha_sso:.6f} BETA={beta_sso*100:.4f}%/yr"
 )
 return 0
if __name__ == "__main__":
 sys.exit(main(sys.argv[1:]))
