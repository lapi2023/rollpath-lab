# src/commands/optimize_cost_by_mae.py
"""
Optimize leveraged ETF simulation parameters against live ETF history.

Supported models
----------------
- dividend_carry
    Used for SPXL / SSO
    Optimizable parameters typically:
    - borrow_alpha
    - borrow_beta
    - cost_annual

- constant_effective_carry
    Used for TQQQ / QLD
    Optimizable parameters typically:
    - borrow_alpha
    - borrow_beta
    - cost_annual
    - carry_annual

The optimizer compares the simulated series against the actual ETF series using
Adjusted Close when available. This means the comparison is based on a
total-return-like live ETF path.

By default, all optimization settings (symbols, bounds, initial values, loss
type, and whether to save the best simulated file) are controlled via
`src/settings.py`.
"""

from __future__ import annotations

import argparse
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd

from src import settings
from src.commands.generate_leveraged_etf import (
    find_latest_file,
    read_index_daily,
    save_simulated_series,
    simulate_from_spec,
)
from src.dividend_loader import (
    DividendInputPaths,
    load_monthly_dividend_and_riskfree,
    to_daily_series_from_monthly,
)
from src.utils import read_price_csv_two_col


# ---------------------------------------------------------------------------
# I/O helpers
# ---------------------------------------------------------------------------
@dataclass
class PreparedInputs:
    spx_df: pd.DataFrame
    ndx_df: pd.DataFrame
    div_spx_d: pd.Series
    rf_spx_d: pd.Series
    rf_ndx_d: pd.Series


def build_monthly_input_paths(
    dividend_csv: Optional[str],
    tbill_1920_1934_csv: Optional[str],
    tb3ms_1934_now_csv: Optional[str],
) -> DividendInputPaths:
    """
    Resolve raw monthly input files from CLI or settings patterns.
    """
    dividend_path = (
        Path(dividend_csv)
        if dividend_csv
        else find_latest_file(
            settings.RAW_DATA_FILES["sp500_dividend_monthly"],
            base_dir=settings.DATA_DIR,
        )
    )
    tbill_path = (
        Path(tbill_1920_1934_csv)
        if tbill_1920_1934_csv
        else find_latest_file(
            settings.RAW_DATA_FILES["tbill_1920_1934_monthly"],
            base_dir=settings.DATA_DIR,
        )
    )
    tb3ms_path = (
        Path(tb3ms_1934_now_csv)
        if tb3ms_1934_now_csv
        else find_latest_file(
            settings.RAW_DATA_FILES["tb3ms_1934_now_monthly"],
            base_dir=settings.DATA_DIR,
        )
    )

    return DividendInputPaths(
        sp500_dividend_monthly_csv=dividend_path,
        tbill_1920_1934_monthly_csv=tbill_path,
        tb3ms_1934_now_monthly_csv=tb3ms_path,
    )



def prepare_inputs(
    spx_csv: Optional[str],
    ndx_csv: Optional[str],
    dividend_csv: Optional[str],
    tbill_1920_1934_csv: Optional[str],
    tb3ms_1934_now_csv: Optional[str],
) -> PreparedInputs:
    """
    Load and prepare all index and monthly carry inputs needed by the optimizer.
    """
    spx_path = (
        Path(spx_csv)
        if spx_csv
        else find_latest_file(
            settings.INDEX_FILE_GLOBS["SP500"],
            base_dir=settings.DATA_DIR,
            exclude_substrings=settings.EXCLUDE_FILENAME_SUBSTRINGS,
        )
    )
    ndx_path = (
        Path(ndx_csv)
        if ndx_csv
        else find_latest_file(
            settings.INDEX_FILE_GLOBS["NDX"],
            base_dir=settings.DATA_DIR,
            exclude_substrings=settings.EXCLUDE_FILENAME_SUBSTRINGS,
        )
    )

    spx_df = read_index_daily(spx_path)
    ndx_df = read_index_daily(ndx_path)

    monthly_paths = build_monthly_input_paths(
        dividend_csv=dividend_csv,
        tbill_1920_1934_csv=tbill_1920_1934_csv,
        tb3ms_1934_now_csv=tb3ms_1934_now_csv,
    )
    div_m, rf_m = load_monthly_dividend_and_riskfree(monthly_paths)

    div_spx_d = to_daily_series_from_monthly(div_m, "DivYield", spx_df["Date"])
    rf_spx_d = to_daily_series_from_monthly(rf_m, "RiskFree", spx_df["Date"])
    rf_ndx_d = to_daily_series_from_monthly(rf_m, "RiskFree", ndx_df["Date"])

    return PreparedInputs(
        spx_df=spx_df,
        ndx_df=ndx_df,
        div_spx_d=div_spx_d,
        rf_spx_d=rf_spx_d,
        rf_ndx_d=rf_ndx_d,
    )


# ---------------------------------------------------------------------------
# Loss functions
# ---------------------------------------------------------------------------
def mae_loss(sim_norm: np.ndarray, act_norm: np.ndarray) -> float:
    return float(np.mean(np.abs(sim_norm / act_norm - 1.0) * 100.0))


def huber_loss(sim_norm: np.ndarray, act_norm: np.ndarray, delta: float = 1.0) -> float:
    r = (sim_norm / act_norm - 1.0) * 100.0
    abs_r = np.abs(r)
    quad = np.minimum(abs_r, delta)
    lin = abs_r - quad
    return float(np.mean(0.5 * quad**2 + delta * lin))


def trimmed_mae_loss(sim_norm: np.ndarray, act_norm: np.ndarray, trim: float = 0.02) -> float:
    r = np.sort(np.abs((sim_norm / act_norm - 1.0) * 100.0))
    n = len(r)
    if n == 0:
        return float("nan")
    k = int(n * trim)
    if 2 * k >= n:
        return float(np.mean(r))
    return float(np.mean(r[k:n - k]))


def compute_loss(
    sim_norm: np.ndarray,
    act_norm: np.ndarray,
    loss_name: str,
    huber_delta: float,
    trim: float,
) -> float:
    if loss_name == "mae":
        return mae_loss(sim_norm, act_norm)
    if loss_name == "huber":
        return huber_loss(sim_norm, act_norm, delta=huber_delta)
    if loss_name == "trimmed":
        return trimmed_mae_loss(sim_norm, act_norm, trim=trim)
    raise ValueError(f"Unsupported loss type: {loss_name}")


# ---------------------------------------------------------------------------
# Optimization primitives
# ---------------------------------------------------------------------------
def golden_section_minimize(
    f,
    lo: float,
    hi: float,
    tol: float = 1e-7,
    max_iter: int = 200,
) -> Tuple[float, float]:
    """
    One-dimensional golden-section minimization.
    """
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

    x = (a + b) / 2.0
    return x, f(x)



def coordinate_descent_generic(
    objective,
    optimize_params: list[str],
    bounds: Dict[str, Tuple[float, float]],
    initial: Dict[str, float],
    tol: float,
    max_outer: int,
) -> Tuple[Dict[str, float], float]:
    """
    Coordinate descent across any supported set of parameters.
    """
    params = {k: float(v) for k, v in initial.items()}

    for name in optimize_params:
        lo, hi = bounds[name]
        params[name] = float(np.clip(params[name], lo, hi))

    best_obj = objective(params)

    for _ in range(max_outer):
        prev_obj = best_obj

        for name in optimize_params:
            lo, hi = bounds[name]

            def one_dim_objective(x: float, param_name: str = name) -> float:
                trial = params.copy()
                trial[param_name] = float(x)
                return objective(trial)

            x_opt, obj_opt = golden_section_minimize(
                one_dim_objective,
                lo,
                hi,
                tol=1e-7,
                max_iter=200,
            )
            params[name] = float(x_opt)
            best_obj = float(obj_opt)

        if abs(prev_obj - best_obj) < tol:
            break

    return params, best_obj


# ---------------------------------------------------------------------------
# Alignment and simulation
# ---------------------------------------------------------------------------
def align_and_normalize(actual_df: pd.DataFrame, sim_df: pd.DataFrame) -> pd.DataFrame:
    """
    Inner-join on dates and normalize both series to 1.0 at the first overlap.
    """
    a = actual_df.rename(columns={"Price": "actual"})
    s = sim_df.rename(columns={"Price": "sim"})
    merged = pd.merge(a, s, on="Date", how="inner")

    if merged.empty:
        raise ValueError("No overlapping dates were found after alignment.")

    merged = merged.sort_values("Date").reset_index(drop=True)
    merged["norm_actual"] = merged["actual"] / float(merged["actual"].iloc[0])
    merged["norm_sim"] = merged["sim"] / float(merged["sim"].iloc[0])
    return merged



def get_actual_series_for_symbol(
    symbol: str,
    start: Optional[pd.Timestamp],
    end: Optional[pd.Timestamp],
) -> Tuple[Path, pd.DataFrame]:
    """
    Load the latest actual ETF file for a symbol, excluding simulated files.
    """
    actual_glob = settings.OPTIMIZATION_SPECS[symbol]["actual_glob"]
    path = find_latest_file(
        actual_glob,
        base_dir=settings.DATA_DIR,
        exclude_substrings=settings.EXCLUDE_FILENAME_SUBSTRINGS,
    )
    df = read_price_csv_two_col(path)

    if start is not None:
        df = df[df["Date"] >= start]
    if end is not None:
        df = df[df["Date"] <= end]

    df = df.reset_index(drop=True)

    if df.empty:
        raise ValueError(f"Actual series for {symbol} is empty after applying the date window.")

    return path, df



def build_symbol_spec_from_params(symbol: str, params: Dict[str, float]) -> Dict[str, float | int | str]:
    """
    Merge optimized parameter values into the base simulation spec.
    """
    spec = deepcopy(settings.SIMULATION_SPECS[symbol])
    for key, value in params.items():
        spec[key] = float(value)
    return spec



def simulate_for_symbol(
    symbol: str,
    prepared: PreparedInputs,
    params: Dict[str, float],
    include_dividends: bool,
    start: Optional[pd.Timestamp],
    end: Optional[pd.Timestamp],
) -> pd.DataFrame:
    """
    Simulate a series for one symbol using prepared inputs and a parameter set.
    """
    spec = build_symbol_spec_from_params(symbol, params)
    underlying = str(spec["underlying"]).upper()

    if underlying == "SP500":
        index_df = prepared.spx_df
        div_y = prepared.div_spx_d
        rf_y = prepared.rf_spx_d
    elif underlying == "NDX":
        index_df = prepared.ndx_df
        div_y = None
        rf_y = prepared.rf_ndx_d
    else:
        raise ValueError(f"Unsupported underlying: {underlying}")

    if start is not None:
        index_df = index_df[index_df["Date"] >= start]
    if end is not None:
        index_df = index_df[index_df["Date"] <= end]
    index_df = index_df.reset_index(drop=True)

    if underlying == "SP500":
        div_y = div_y.reindex(index_df.index)
        rf_y = rf_y.reindex(index_df.index)
    else:
        rf_y = rf_y.reindex(index_df.index)

    sim_df = simulate_from_spec(
        index_df=index_df,
        div_y=div_y,
        rf_y=rf_y,
        spec=spec,
        include_dividends=include_dividends,
        base_from_index=True,
    )
    return sim_df


# ---------------------------------------------------------------------------
# Objective construction
# ---------------------------------------------------------------------------
def build_objective(
    symbol: str,
    actual_df: pd.DataFrame,
    prepared: PreparedInputs,
    include_dividends: bool,
    start: Optional[pd.Timestamp],
    end: Optional[pd.Timestamp],
    loss_name: str,
    huber_delta: float,
    trim: float,
):
    """
    Build the objective function used during optimization.
    """

    def objective(params: Dict[str, float]) -> float:
        sim_df = simulate_for_symbol(
            symbol=symbol,
            prepared=prepared,
            params=params,
            include_dividends=include_dividends,
            start=start,
            end=end,
        )
        merged = align_and_normalize(actual_df, sim_df)
        return compute_loss(
            sim_norm=merged["norm_sim"].to_numpy(),
            act_norm=merged["norm_actual"].to_numpy(),
            loss_name=loss_name,
            huber_delta=huber_delta,
            trim=trim,
        )

    return objective


# ---------------------------------------------------------------------------
# End-to-end optimization
# ---------------------------------------------------------------------------
def optimize_for_symbol(
    symbol: str,
    prepared: PreparedInputs,
    include_dividends: bool,
    loss_name: str,
    huber_delta: float,
    trim: float,
    start: Optional[pd.Timestamp],
    end: Optional[pd.Timestamp],
    save_best_sim: bool,
) -> Tuple[Dict[str, float], float, pd.DataFrame]:
    """
    Optimize one symbol using settings-driven parameter specs.

    If start is not provided, this function automatically uses the latest
    available start date across:
    - the actual ETF history, and
    - the required underlying index history.
    """
    _, actual_df = get_actual_series_for_symbol(symbol, start=None, end=end)

    opt_spec = settings.OPTIMIZATION_SPECS[symbol]
    optimize_params = list(opt_spec["optimize_params"])
    bounds = deepcopy(opt_spec["bounds"])
    initial = deepcopy(opt_spec["initial"])

    # Apply hard floors from settings to prevent unrealistic solutions.
    hard_floors = getattr(settings, "OPTIMIZATION_HARD_FLOORS", {}).get(symbol, {})
    for param_name, floor_value in hard_floors.items():
        if param_name in bounds:
            lo, hi = bounds[param_name]
            bounds[param_name] = (max(lo, float(floor_value)), hi)
        if param_name in initial:
            initial[param_name] = max(
                float(initial[param_name]),
                float(floor_value),
            )

    underlying = str(settings.SIMULATION_SPECS[symbol]["underlying"]).upper()
    if underlying == "SP500":
        index_df = prepared.spx_df
    elif underlying == "NDX":
        index_df = prepared.ndx_df
    else:
        raise ValueError(f"Unsupported underlying: {underlying}")

    default_start: Optional[pd.Timestamp] = None
    candidates: list[pd.Timestamp] = []
    if not actual_df.empty:
        candidates.append(pd.to_datetime(actual_df["Date"].min()))
    if not index_df.empty:
        candidates.append(pd.to_datetime(index_df["Date"].min()))
    if candidates:
        default_start = max(candidates)

    start_eff = start if start is not None else default_start

    if start_eff is not None:
        actual_df = actual_df[actual_df["Date"] >= start_eff].reset_index(drop=True)

    if actual_df.empty:
        raise ValueError(
            f"Actual series for {symbol} is empty after applying the effective date window."
        )

    objective = build_objective(
        symbol=symbol,
        actual_df=actual_df,
        prepared=prepared,
        include_dividends=include_dividends,
        start=start_eff,
        end=end,
        loss_name=loss_name,
        huber_delta=huber_delta,
        trim=trim,
    )

    best_params, best_obj = coordinate_descent_generic(
        objective=objective,
        optimize_params=optimize_params,
        bounds=bounds,
        initial=initial,
        tol=settings.OPTIMIZATION_TOL,
        max_outer=settings.OPTIMIZATION_MAX_OUTER,
    )

    best_sim = simulate_for_symbol(
        symbol=symbol,
        prepared=prepared,
        params=best_params,
        include_dividends=include_dividends,
        start=start_eff,
        end=end,
    )
    merged = align_and_normalize(actual_df, best_sim)

    if save_best_sim:
        spec_to_save = build_symbol_spec_from_params(symbol, best_params)
        outpath = save_simulated_series(
            df=best_sim,
            outdir=settings.DATA_DIR,
            symbol=symbol,
            spec=spec_to_save,
            include_dividends=include_dividends,
        )
        print(f"Saved best simulated series: {outpath}")

    return best_params, best_obj, merged

# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def parse_args(argv: Optional[list[str]] = None) -> argparse.Namespace:
    """
    Parse command-line arguments.
    """
    ap = argparse.ArgumentParser(
        description="Optimize leveraged ETF simulation parameters against live ETF history."
    )
    ap.add_argument(
        "--symbols",
        nargs="+",
        default=settings.OPTIMIZATION_DEFAULT_SYMBOLS,
        choices=sorted(settings.OPTIMIZATION_SPECS.keys()),
        help="Symbols to optimize.",
    )
    ap.add_argument(
        "--loss",
        type=str,
        default=settings.OPTIMIZATION_DEFAULT_LOSS,
        choices=["mae", "huber", "trimmed"],
    )
    ap.add_argument(
        "--huber-delta",
        type=float,
        default=settings.OPTIMIZATION_DEFAULT_HUBER_DELTA,
        dest="huber_delta",
    )
    ap.add_argument(
        "--trim",
        type=float,
        default=settings.OPTIMIZATION_DEFAULT_TRIM,
    )
    ap.add_argument("--start", type=str, default=None)
    ap.add_argument("--end", type=str, default=None)

    ap.add_argument("--include-dividends", action="store_true")
    ap.add_argument("--no-dividends", action="store_true")

    ap.add_argument("--spx-csv", type=str, default=None)
    ap.add_argument("--ndx-csv", type=str, default=None)
    ap.add_argument("--dividend-csv", type=str, default=None)
    ap.add_argument("--tbill-1920-1934-csv", type=str, default=None)
    ap.add_argument("--tb3ms-1934-now-csv", type=str, default=None)

    ap.add_argument(
        "--save-best-sim",
        action="store_true",
        default=settings.OPTIMIZATION_SAVE_BEST_SIM,
        help="Save the best fitted simulated series into data/.",
    )
    return ap.parse_args(argv)



def main(argv: Optional[list[str]] = None) -> int:
    """
    Run the optimization workflow.
    """
    args = parse_args(argv)

    include_dividends = settings.INCLUDE_DIVIDENDS
    if args.include_dividends:
        include_dividends = True
    if args.no_dividends:
        include_dividends = False

    start_ts = pd.to_datetime(args.start) if args.start else None
    end_ts = pd.to_datetime(args.end) if args.end else None

    prepared = prepare_inputs(
        spx_csv=args.spx_csv,
        ndx_csv=args.ndx_csv,
        dividend_csv=args.dividend_csv,
        tbill_1920_1934_csv=args.tbill_1920_1934_csv,
        tb3ms_1934_now_csv=args.tb3ms_1934_now_csv,
    )

    print("=== Leveraged ETF parameter optimization ===")
    print(f"loss={args.loss} include_dividends={include_dividends}")

    results: Dict[str, Tuple[Dict[str, float], float]] = {}

    for symbol in args.symbols:
        print(f"--- {symbol} ---")
        best_params, best_obj, _ = optimize_for_symbol(
            symbol=symbol,
            prepared=prepared,
            include_dividends=include_dividends,
            loss_name=args.loss,
            huber_delta=args.huber_delta,
            trim=args.trim,
            start=start_ts,
            end=end_ts,
            save_best_sim=args.save_best_sim,
        )

        results[symbol] = (best_params, best_obj)

        ordered_parts = []
        for key in settings.OPTIMIZATION_SPECS[symbol]["optimize_params"]:
            value = best_params[key]
            if key in ("borrow_beta", "cost_annual", "carry_annual"):
                ordered_parts.append(f"{key}={value * 100:.6f}%/yr")
            else:
                ordered_parts.append(f"{key}={value:.6f}")

        print(f"{symbol}: " + ", ".join(ordered_parts) + f" (loss={best_obj:.6f})")

    print("=== Recommended parameters ===")
    for symbol, (best_params, best_obj) in results.items():
        ordered_parts = []
        for key in settings.OPTIMIZATION_SPECS[symbol]["optimize_params"]:
            value = best_params[key]
            if key in ("borrow_beta", "cost_annual", "carry_annual"):
                ordered_parts.append(f"{key}={value * 100:.6f}%/yr")
            else:
                ordered_parts.append(f"{key}={value:.6f}")

        print(f"{symbol}: " + ", ".join(ordered_parts) + f" (loss={best_obj:.6f})")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
