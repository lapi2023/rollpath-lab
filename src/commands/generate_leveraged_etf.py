# src/commands/generate_leveraged_etf.py
"""
Generate simulated daily price series for leveraged ETFs.

Supported models
----------------
1) dividend_carry
   Used by:
   - SSO
   - SPXL

   Formula:
       r_sim = L * r_index
               + [L * dividend_yield - (L - 1) * borrow_cost] * dt
               - fee * dt

2) constant_effective_carry
   Used by:
   - TQQQ
   - QLD

   Formula:
       r_sim = L * r_index
               + [effective_carry - (L - 1) * borrow_cost] * dt
               - fee * dt

   This is intended for total-return reconstruction when long dividend-yield
   or total-return index histories are not available.

Notes
-----
- All simulations assume daily-reset leverage.
- Borrowing cost is modeled as:
      borrow_cost = borrow_alpha * risk_free_rate + borrow_beta
- For constant_effective_carry symbols, setting --no-dividends forces the
  carry term to zero and therefore produces a price-only approximation.
- Output filenames are self-describing and encode model parameters.
"""

from __future__ import annotations

import argparse
from copy import deepcopy
from pathlib import Path
import sys
import fnmatch
from typing import Dict, Iterable, Optional, Set

import pandas as pd

from src import settings
from src.dividend_loader import (
    DividendInputPaths,
    load_monthly_dividend_and_riskfree,
    to_daily_series_from_monthly,
)
from src.utils import read_price_csv_two_col


def find_latest_file(
        glob_pattern: str,
        base_dir: Optional[Path] = None,
        exclude_substrings: Optional[Iterable[str]] = None,
) -> Path:
 """
 Return the most recently modified file that matches a glob pattern.

 This implementation performs case-insensitive filename matching so that
 patterns such as '*tqqq_d_*.csv' will also match files like:
 - TQQQ_d_2025.csv
 - tqqq_D_sample.CSV
 - ^TQQQ_D_2026-03-01.csv

 Parameters
 ----------
 glob_pattern
     Glob-style pattern relative to `base_dir`. Examples:
     - '*tqqq_d_*.csv'
     - 'raw/*TB3MS*.csv'
 base_dir
     Base directory where the pattern is evaluated.
 exclude_substrings
     Optional iterable of substrings. Any file whose name contains one of
     these substrings (case-insensitive) will be ignored.

 Returns
 -------
 pathlib.Path
     The most recently modified matching file.

 Raises
 ------
 FileNotFoundError
     If no usable file matches the pattern.
 """
 base = base_dir or settings.DATA_DIR
 excludes = tuple(s.casefold() for s in (exclude_substrings or ()))

 pattern_path = Path(glob_pattern)
 search_dir = base / pattern_path.parent
 name_pattern = pattern_path.name.casefold()

 if not search_dir.exists():
  raise FileNotFoundError(
   f"Search directory does not exist: '{search_dir}' "
   f"(from pattern '{glob_pattern}')"
  )

 candidates: list[Path] = []

 for p in search_dir.iterdir():
  if not p.is_file():
   continue

  filename_cf = p.name.casefold()

  if any(sub in filename_cf for sub in excludes):
   continue

  if fnmatch.fnmatch(filename_cf, name_pattern):
   candidates.append(p)

 if not candidates:
  raise FileNotFoundError(
   f"No usable file matched '{glob_pattern}' inside '{base}'."
  )

 return max(candidates, key=lambda p: p.stat().st_mtime)


def read_index_daily(index_csv: Path) -> pd.DataFrame:
    """
    Load a daily index CSV as a strict two-column (Date, Price) file.
    This wrapper delegates to the unified reader in src.utils.
    """
    df = read_price_csv_two_col(index_csv)
    return df[["Date", "Price"]]


def annual_to_step_additive(rate_annual: pd.Series | float, dt_years: pd.Series) -> pd.Series:
    """
    Convert annualized additive rates into step-level additive returns.
    """
    if isinstance(rate_annual, pd.Series):
        return rate_annual.astype(float) * dt_years
    return pd.Series(float(rate_annual), index=dt_years.index) * dt_years


def simulate_lever_price_dividend(
    index_df: pd.DataFrame,
    div_y: pd.Series,
    rf_y: pd.Series,
    leverage: int,
    cost_annual: float,
    include_dividends: bool,
    base_from_index: bool,
    borrow_alpha: float,
    borrow_beta: float,
) -> pd.DataFrame:
    """
    Simulate a daily-reset leveraged ETF path using the dividend-carry model.
    """
    out = index_df.copy()
    r_index = out["Price"].pct_change().fillna(0.0)
    dt = (out["Date"].diff().dt.days.fillna(0)).astype(float) / 365.0

    div_ann = div_y.astype(float)
    rf_ann = rf_y.astype(float)
    borrow_ann = borrow_alpha * rf_ann + borrow_beta

    if include_dividends:
        carry_ann = leverage * div_ann - (leverage - 1) * borrow_ann
    else:
        carry_ann = -(leverage - 1) * borrow_ann

    drift = annual_to_step_additive(carry_ann, dt)
    fee = annual_to_step_additive(float(cost_annual), dt)

    r_sim = leverage * r_index + drift - fee
    p0 = float(out["Price"].iloc[0]) if base_from_index else 1.0
    price_sim = (1.0 + r_sim).cumprod() * p0

    return pd.DataFrame({"Date": out["Date"], "Price": price_sim})


def simulate_lever_price_constant_carry(
    index_df: pd.DataFrame,
    rf_y: pd.Series,
    leverage: int,
    cost_annual: float,
    include_dividends: bool,
    base_from_index: bool,
    borrow_alpha: float,
    borrow_beta: float,
    carry_annual: float,
) -> pd.DataFrame:
    """
    Simulate a daily-reset leveraged ETF path using the constant effective carry model.

    If `include_dividends` is False, the carry term is set to zero, which yields a
    price-only approximation.
    """
    out = index_df.copy()
    r_index = out["Price"].pct_change().fillna(0.0)
    dt = (out["Date"].diff().dt.days.fillna(0)).astype(float) / 365.0

    rf_ann = rf_y.astype(float)
    borrow_ann = borrow_alpha * rf_ann + borrow_beta

    effective_carry_ann = float(carry_annual) if include_dividends else 0.0
    net_drift_ann = effective_carry_ann - (leverage - 1) * borrow_ann

    drift = annual_to_step_additive(net_drift_ann, dt)
    fee = annual_to_step_additive(float(cost_annual), dt)

    r_sim = leverage * r_index + drift - fee
    p0 = float(out["Price"].iloc[0]) if base_from_index else 1.0
    price_sim = (1.0 + r_sim).cumprod() * p0

    return pd.DataFrame({"Date": out["Date"], "Price": price_sim})


def simulate_from_spec(
    index_df: pd.DataFrame,
    div_y: Optional[pd.Series],
    rf_y: pd.Series,
    spec: Dict[str, float | int | str],
    include_dividends: bool,
    base_from_index: bool,
) -> pd.DataFrame:
    """
    Dispatch to the appropriate simulation model using a symbol spec.
    """
    model = str(spec["model"])
    leverage = int(spec["leverage"])
    cost_annual = float(spec["cost_annual"])
    borrow_alpha = float(spec["borrow_alpha"])
    borrow_beta = float(spec["borrow_beta"])
    carry_annual = float(spec.get("carry_annual", 0.0))

    if model == "dividend_carry":
        if div_y is None:
            raise ValueError("div_y must be provided for the dividend_carry model.")
        return simulate_lever_price_dividend(
            index_df=index_df,
            div_y=div_y,
            rf_y=rf_y,
            leverage=leverage,
            cost_annual=cost_annual,
            include_dividends=include_dividends,
            base_from_index=base_from_index,
            borrow_alpha=borrow_alpha,
            borrow_beta=borrow_beta,
        )

    if model == "constant_effective_carry":
        return simulate_lever_price_constant_carry(
            index_df=index_df,
            rf_y=rf_y,
            leverage=leverage,
            cost_annual=cost_annual,
            include_dividends=include_dividends,
            base_from_index=base_from_index,
            borrow_alpha=borrow_alpha,
            borrow_beta=borrow_beta,
            carry_annual=carry_annual,
        )

    raise ValueError(f"Unsupported simulation model: {model}")


def dividends_tag(include_dividends: bool) -> str:
    """
    Return a short output tag describing dividend treatment.
    """
    return "TR" if include_dividends else "PX"


def borrow_tag(alpha: float, beta: float) -> str:
    """
    Format borrow parameters as a compact filename tag.
    """
    a = f"{alpha:.6f}".rstrip("0").rstrip(".")
    b = f"{beta * 100:.4f}%".rstrip("0").rstrip(".").replace(".%", "%")
    return f"a{a}_b{b}"


def carry_tag(carry_annual: float) -> str:
    """
    Format constant effective carry as a compact filename tag.
    """
    c = f"{carry_annual * 100:.4f}%".rstrip("0").rstrip(".").replace(".%", "%")
    return f"carry{c}"


def cost_tag(cost_annual: float) -> str:
    """
    Format annual cost as a compact filename tag.
    """
    c = f"{cost_annual * 100:.4f}%".rstrip("0").rstrip(".").replace(".%", "%")
    return f"{c}cost"


def build_output_filename(
    symbol: str,
    spec: Dict[str, float | int | str],
    include_dividends: bool,
    start_date: pd.Timestamp,
    end_date: pd.Timestamp,
) -> str:
    """
    Build a self-describing output filename for a simulated series.
    """
    prefix = str(spec["output_prefix"])
    mode_tag = dividends_tag(include_dividends)
    model = str(spec["model"])
    cost_part = cost_tag(float(spec["cost_annual"]))
    borrow_part = borrow_tag(float(spec["borrow_alpha"]), float(spec["borrow_beta"]))

    parts = [prefix, mode_tag, model, cost_part, borrow_part]

    if model == "constant_effective_carry":
        parts.append(carry_tag(float(spec.get("carry_annual", 0.0))))

    parts.append(start_date.date().isoformat())
    parts.append(end_date.date().isoformat())

    return "_".join(parts) + ".csv"


def save_simulated_series(
    df: pd.DataFrame,
    outdir: Path,
    symbol: str,
    spec: Dict[str, float | int | str],
    include_dividends: bool,
) -> Path:
    """
    Save a simulated series using the standard output naming convention.
    """
    outdir.mkdir(parents=True, exist_ok=True)
    filename = build_output_filename(
        symbol=symbol,
        spec=spec,
        include_dividends=include_dividends,
        start_date=pd.to_datetime(df["Date"].min()),
        end_date=pd.to_datetime(df["Date"].max()),
    )
    outpath = outdir / filename
    df.to_csv(outpath, index=False)
    return outpath


def resolve_symbol_spec(
    symbol: str,
    cli_cost: Optional[float],
    cli_borrow_alpha: Optional[float],
    cli_borrow_beta: Optional[float],
    cli_carry_annual: Optional[float],
) -> Dict[str, float | int | str]:
    """
    Resolve a symbol simulation spec from settings with optional CLI overrides.
    """
    key = symbol.upper()
    if key not in settings.SIMULATION_SPECS:
        raise KeyError(f"Simulation spec not found for symbol '{symbol}'.")

    spec = deepcopy(settings.SIMULATION_SPECS[key])

    if cli_cost is not None:
        spec["cost_annual"] = float(cli_cost)
    if cli_borrow_alpha is not None:
        spec["borrow_alpha"] = float(cli_borrow_alpha)
    if cli_borrow_beta is not None:
        spec["borrow_beta"] = float(cli_borrow_beta)
    if cli_carry_annual is not None:
        spec["carry_annual"] = float(cli_carry_annual)

    return spec


def parse_args(argv: list[str]) -> argparse.Namespace:
    """
    Parse command-line arguments.
    """
    ap = argparse.ArgumentParser(
        description="Generate simulated daily leveraged ETF series."
    )
    ap.add_argument(
        "--symbols",
        nargs="+",
        default=settings.SIMULATION_DEFAULT_SYMBOLS,
        choices=sorted(settings.SIMULATION_SPECS.keys()),
        help="Symbols to simulate.",
    )
    ap.add_argument("--spx-csv", type=str, default=None)
    ap.add_argument("--ndx-csv", type=str, default=None)
    ap.add_argument("--dividend-csv", type=str, default=None)
    ap.add_argument("--tbill-1920-1934-csv", type=str, default=None)
    ap.add_argument("--tb3ms-1934-now-csv", type=str, default=None)

    ap.add_argument("--cost", type=float, default=None, help="Override annual cost for all selected symbols.")
    ap.add_argument("--borrow-alpha", type=float, default=None, help="Override borrow alpha for all selected symbols.")
    ap.add_argument("--borrow-beta", type=float, default=None, help="Override borrow beta for all selected symbols.")
    ap.add_argument(
        "--carry-annual",
        type=float,
        default=None,
        help="Override constant effective carry (annualized) for constant_effective_carry symbols.",
    )

    ap.add_argument("--include-dividends", action="store_true")
    ap.add_argument("--no-dividends", action="store_true")

    ap.add_argument("--start", type=str, default=None)
    ap.add_argument("--end", type=str, default=None)
    ap.add_argument("--outdir", type=str, default=str(settings.DATA_DIR))
    return ap.parse_args(argv)


def main(argv: list[str]) -> int:
    """
    Run the leveraged ETF generation workflow.
    """
    args = parse_args(argv)

    include_dividends = settings.INCLUDE_DIVIDENDS
    if args.include_dividends:
        include_dividends = True
    if args.no_dividends:
        include_dividends = False

    base_from_index = settings.BASE_FROM_INDEX

    spx_csv = Path(args.spx_csv) if args.spx_csv else find_latest_file(
        settings.INDEX_FILE_GLOBS["SP500"],
        base_dir=settings.DATA_DIR,
        exclude_substrings=settings.EXCLUDE_FILENAME_SUBSTRINGS,
    )
    ndx_csv = Path(args.ndx_csv) if args.ndx_csv else find_latest_file(
        settings.INDEX_FILE_GLOBS["NDX"],
        base_dir=settings.DATA_DIR,
        exclude_substrings=settings.EXCLUDE_FILENAME_SUBSTRINGS,
    )

    dividend_csv = Path(args.dividend_csv) if args.dividend_csv else find_latest_file(
        settings.RAW_DATA_FILES["sp500_dividend_monthly"],
        base_dir=settings.DATA_DIR,
    )
    tbill_1920_1934_csv = (
        Path(args.tbill_1920_1934_csv)
        if args.tbill_1920_1934_csv
        else find_latest_file(
            settings.RAW_DATA_FILES["tbill_1920_1934_monthly"],
            base_dir=settings.DATA_DIR,
        )
    )
    tb3ms_1934_now_csv = (
        Path(args.tb3ms_1934_now_csv)
        if args.tb3ms_1934_now_csv
        else find_latest_file(
            settings.RAW_DATA_FILES["tb3ms_1934_now_monthly"],
            base_dir=settings.DATA_DIR,
        )
    )

    spx_df = read_index_daily(spx_csv)
    ndx_df = read_index_daily(ndx_csv)

    def _infer_default_start(
            symbols: list[str],
            spx: pd.DataFrame,
            ndx: pd.DataFrame,
    ) -> Optional[pd.Timestamp]:
     """
     Return the latest available start date across the underlyings
     required by the selected symbols.
     """
     try:
      needed: Set[str] = {
       str(settings.SIMULATION_SPECS[s]["underlying"]).upper()
       for s in symbols
      }
     except Exception:
      needed = {"SP500", "NDX"}

     candidates: list[pd.Timestamp] = []
     if "SP500" in needed and not spx.empty:
      candidates.append(pd.to_datetime(spx["Date"].min()))
     if "NDX" in needed and not ndx.empty:
      candidates.append(pd.to_datetime(ndx["Date"].min()))

     return max(candidates) if candidates else None

    start_ts = (
     pd.to_datetime(args.start)
     if args.start
     else _infer_default_start(args.symbols, spx_df, ndx_df)
    )
    if start_ts is not None:
     spx_df = spx_df[spx_df["Date"] >= start_ts].reset_index(drop=True)
     ndx_df = ndx_df[ndx_df["Date"] >= start_ts].reset_index(drop=True)

    if args.end:
     end_ts = pd.to_datetime(args.end)
     spx_df = spx_df[spx_df["Date"] <= end_ts].reset_index(drop=True)
     ndx_df = ndx_df[ndx_df["Date"] <= end_ts].reset_index(drop=True)

    paths = DividendInputPaths(
        sp500_dividend_monthly_csv=dividend_csv,
        tbill_1920_1934_monthly_csv=tbill_1920_1934_csv,
        tb3ms_1934_now_monthly_csv=tb3ms_1934_now_csv,
    )
    div_m, rf_m = load_monthly_dividend_and_riskfree(paths)

    div_spx_d = to_daily_series_from_monthly(div_m, "DivYield", spx_df["Date"])
    rf_spx_d = to_daily_series_from_monthly(rf_m, "RiskFree", spx_df["Date"])
    rf_ndx_d = to_daily_series_from_monthly(rf_m, "RiskFree", ndx_df["Date"])

    outdir = Path(args.outdir)
    saved_paths: list[Path] = []

    for symbol in args.symbols:
        spec = resolve_symbol_spec(
            symbol=symbol,
            cli_cost=args.cost,
            cli_borrow_alpha=args.borrow_alpha,
            cli_borrow_beta=args.borrow_beta,
            cli_carry_annual=args.carry_annual,
        )

        underlying = str(spec["underlying"]).upper()
        if underlying == "SP500":
            index_df = spx_df
            div_y = div_spx_d
            rf_y = rf_spx_d
        elif underlying == "NDX":
            index_df = ndx_df
            div_y = None
            rf_y = rf_ndx_d
        else:
            raise ValueError(f"Unsupported underlying: {underlying}")

        sim_df = simulate_from_spec(
            index_df=index_df,
            div_y=div_y,
            rf_y=rf_y,
            spec=spec,
            include_dividends=include_dividends,
            base_from_index=base_from_index,
        )

        outpath = save_simulated_series(
            df=sim_df,
            outdir=outdir,
            symbol=symbol,
            spec=spec,
            include_dividends=include_dividends,
        )
        saved_paths.append(outpath)

        print(
            f"[{symbol}] saved: {outpath}\n"
            f"    model={spec['model']} underlying={spec['underlying']} "
            f"L={spec['leverage']} "
            f"cost={float(spec['cost_annual']) * 100:.4f}%/yr "
            f"borrow_alpha={float(spec['borrow_alpha']):.6f} "
            f"borrow_beta={float(spec['borrow_beta']) * 100:.4f}%/yr "
            f"carry={float(spec.get('carry_annual', 0.0)) * 100:.4f}%/yr "
            f"include_dividends={include_dividends} "
            f"base_from_index={base_from_index}"
        )

    if not saved_paths:
        print("No files were generated.")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))