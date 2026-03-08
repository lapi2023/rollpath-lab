# src/portfolio.py (English-only)
"""
Portfolio return simulation with rebalancing and optional taxation.
- Polars in/out; inner loops use pandas/numpy without to_pandas() to avoid pyarrow.
- Average-cost basis; tax only on realized gains from rebalancing sells.
"""
from typing import Dict, Tuple, List
import numpy as np
import pandas as pd
import polars as pl

# ---- Debug tax ledger helpers ----
DEBUG_TAX_LEDGER = []          # list of dicts
DEBUG_TAX_LEDGER_PATH = None   # set by main (e.g., output_dir/debug_tax_ledger.csv)

def _append_tax_ledger(date, portfolio_name, asset, sell_amt, realized_gain, tax_paid):
    """Append one taxable rebalancing sell event to the in-memory ledger buffer."""
    try:
        DEBUG_TAX_LEDGER.append({
            'date': str(date),
            'portfolio': str(portfolio_name),
            'asset': str(asset),
            'sell_amount': float(sell_amt),
            'realized_gain': float(realized_gain),
            'tax_paid': float(tax_paid),
        })
    except Exception:
        # do not fail the main simulation even if logging fails
        pass

def _flush_tax_ledger():
    """Flush the in-memory ledger buffer to CSV (append mode, header written once)."""
    global DEBUG_TAX_LEDGER
    if not DEBUG_TAX_LEDGER_PATH or not DEBUG_TAX_LEDGER:
        return
    try:
        import csv, os
        os.makedirs(os.path.dirname(DEBUG_TAX_LEDGER_PATH), exist_ok=True)
        cols = ['date','portfolio','asset','sell_amount','realized_gain','tax_paid']
        file_exists = os.path.exists(DEBUG_TAX_LEDGER_PATH)
        mode = 'a'  # append so we can collect multiple portfolios in one file
        with open(DEBUG_TAX_LEDGER_PATH, mode, newline='', encoding='utf-8') as f:
            w = csv.DictWriter(f, fieldnames=cols)
            if not file_exists:
                w.writeheader()
            for row in DEBUG_TAX_LEDGER:
                w.writerow(row)
    except Exception:
        # never break the simulation because of logging/IO
        pass
    finally:
        DEBUG_TAX_LEDGER = []


def _make_rebalance_id(dates: pd.Series, freq: str) -> np.ndarray:
    dates = pd.to_datetime(dates)
    f = (freq or "").lower()

    if f == "none":
        return np.zeros(len(dates), dtype=int)

    if f == "every_period":
        return np.arange(len(dates), dtype=int)

    if f == "monthly":
        keys = dates.dt.to_period("M").astype(str)
        _, gid = np.unique(keys, return_inverse=True)
        return gid.astype(int)

    if f == "yearly":
        keys = dates.dt.year
        _, gid = np.unique(keys, return_inverse=True)
        return gid.astype(int)

    if f.startswith("yearly_"):
        try:
            n = int(f.split("_")[1])
        except Exception:
            n = 1
        years = dates.dt.year
        start_year = int(years.iloc[0])
        gid = ((years - start_year) // n).to_numpy()
        return gid.astype(int)

    # default falls back to no-rebalance grouping
    return np.zeros(len(dates), dtype=int)


def _simulate_with_tax(
    dates: pd.Series,
    returns_df: pd.DataFrame,
    weights: Dict[str, float],
    rebalance_freq: str,
    tax_rate: float,
    portfolio_name: str,
) -> Tuple[pd.Series, Dict[str, pd.Series]]:
    """
    Simulate portfolio given asset returns and target weights, applying taxes on
    realized gains created by rebalancing sells at group ends.

    - Average cost basis: basis tracks invested cost per asset, proportionally reduced on sells.
    - Taxation: only the gains part of sells is taxed. We solve for post-tax total (T_after) via a fixed-point iteration.
    """
    assets = list(weights.keys())
    w = np.array([weights[a] for a in assets], dtype=float)
    w = w / w.sum()

    rets = returns_df[assets].to_numpy(dtype=float)
    T, _ = rets.shape

    gid = _make_rebalance_id(dates, rebalance_freq)

    # Initialize positions and bases (start with target weights; basis equals invested value initially)
    pos = w.copy()
    basis = w.copy()

    V_prev = 1.0
    port_ret = np.zeros(T, dtype=float)
    tax_paid = np.zeros(T, dtype=float)
    realized_gains = np.zeros(T, dtype=float)

    for t in range(T):
        # Grow positions with asset returns
        pos *= (1.0 + rets[t])

        # End-of-group rebalance
        end_of_group = (t == T - 1) or (gid[t + 1] != gid[t])
        if end_of_group and rebalance_freq != "none":
            T_before = float(pos.sum())
            tax_t = 0.0
            rg_t = 0.0

            if tax_rate > 0.0:
                # Solve for T_after by fixed-point iteration:
                # tax = tax_rate * realized_gains(pos -> target(T_after))
                # T_after = T_before - tax
                T_after = T_before
                for _ in range(16):
                    target_vals = w * T_after
                    # Provisional sells to reach target under the current guess
                    sell = np.maximum(pos - target_vals, 0.0)
                    with np.errstate(divide="ignore", invalid="ignore"):
                        basis_ratio = np.divide(basis, pos, out=np.zeros_like(pos), where=pos > 0.0)
                    gains_component = sell * np.maximum(1.0 - basis_ratio, 0.0)
                    rg_t = float(np.sum(gains_component))
                    tax_t = tax_rate * rg_t
                    new_T_after = T_before - tax_t
                    if abs(new_T_after - T_after) <= 1e-12:
                        T_after = new_T_after
                        break
                    T_after = new_T_after

                # Final target using the converged T_after
                target_vals = w * T_after

                # === Debug ledger (per-asset) ===
                # Actual sells executed at this rebalance step:
                sell_exec = np.maximum(pos - target_vals, 0.0)
                if tax_rate > 0.0 and np.any(sell_exec > 0.0):
                    with np.errstate(divide="ignore", invalid="ignore"):
                        basis_ratio_final = np.divide(basis, pos, out=np.zeros_like(pos), where=pos > 0.0)
                    gains_per_asset = sell_exec * np.maximum(1.0 - basis_ratio_final, 0.0)
                    # Append only when realized gain is positive
                    curr_date = dates.iloc[t] if hasattr(dates, "iloc") else dates[t]
                    for i, a in enumerate(assets):
                        if sell_exec[i] > 0.0 and gains_per_asset[i] > 0.0:
                            _append_tax_ledger(
                                date=curr_date,
                                portfolio_name=portfolio_name,
                                asset=a,
                                sell_amt=float(sell_exec[i]),
                                realized_gain=float(gains_per_asset[i]),
                                tax_paid=float(tax_rate * gains_per_asset[i]),
                            )

                # Update basis proportionally for keeps, add buys to basis, then set positions to target
                with np.errstate(divide="ignore", invalid="ignore"):
                    keep_ratio = np.divide(target_vals, pos, out=np.zeros_like(pos), where=pos > 0.0)
                basis = basis * keep_ratio
                buy = np.maximum(target_vals - pos, 0.0)
                basis = basis + buy
                pos = target_vals.copy()

            else:
                # No tax: ordinary rebalance to T_before
                target_vals = w * T_before
                with np.errstate(divide="ignore", invalid="ignore"):
                    keep_ratio = np.divide(target_vals, pos, out=np.zeros_like(pos), where=pos > 0.0)
                basis = basis * keep_ratio
                buy = np.maximum(target_vals - pos, 0.0)
                basis = basis + buy
                pos = target_vals.copy()

            tax_paid[t] = tax_t
            realized_gains[t] = rg_t

            # Flush ledger for each completed rebalance group so the CSV grows incrementally
            _flush_tax_ledger()

        # Compute portfolio period return
        V_now = float(pos.sum())
        port_ret[t] = (V_now / V_prev) - 1.0
        V_prev = V_now

    series_dict = {
        "tax_paid": pd.Series(tax_paid, index=dates.index, name="tax_paid"),
        "realized_gains": pd.Series(realized_gains, index=dates.index, name="realized_gains"),
    }
    return pd.Series(port_ret, index=dates.index, name="__port__"), series_dict


def _to_pandas_series_no_arrow(df: pl.DataFrame, col: str) -> pd.Series:
    s = df.get_column(col).to_list()
    return pd.Series(s, name=col)


def _to_pandas_frame_no_arrow(df: pl.DataFrame, columns: List[str]) -> pd.DataFrame:
    data = {c: df.get_column(c).to_list() for c in columns}
    return pd.DataFrame(data)


def calculate_portfolio_returns(
    merged_df: pl.DataFrame,
    portfolios: Dict[str, Dict[str, float]],
    rebalance_freq: str,
    tax_rate: float = 0.0,
) -> Tuple[pl.DataFrame, Dict[str, dict]]:
    """
    Build portfolio return columns (Polars) and tax report dict per portfolio.
    """
    out_df = merged_df.select("Date")
    dates_pd = _to_pandas_series_no_arrow(merged_df.select("Date"), "Date")

    tax_reports: Dict[str, dict] = {}

    for p_name, weights in portfolios.items():
        cols = [f"Return_{k}" for k in weights.keys()]
        df_small = merged_df.select(["Date"] + cols)
        df_pd = _to_pandas_frame_no_arrow(df_small, ["Date"] + cols)
        rename_map = {f"Return_{k}": k for k in weights.keys()}
        df_pd = df_pd.rename(columns=rename_map)

        port_ret_series, series_dict = _simulate_with_tax(
            dates=df_pd["Date"],
            returns_df=df_pd[list(weights.keys())],
            weights=weights,
            rebalance_freq=rebalance_freq,
            tax_rate=float(tax_rate or 0.0),
            portfolio_name=p_name,  # for debug ledger
        )

        # Append returns column as Polars Series
        out_df = out_df.with_columns(
            pl.Series(name=p_name, values=port_ret_series.to_numpy())
        )

        # Aggregate tax report metrics per portfolio
        tax_paid = series_dict["tax_paid"]
        realized_gains = series_dict["realized_gains"]
        total_tax = float(tax_paid.sum())
        tax_events = int((tax_paid > 0).sum())
        avg_per_event = float(total_tax / tax_events) if tax_events > 0 else 0.0
        max_event = float(tax_paid.max()) if len(tax_paid) > 0 else 0.0

        tax_reports[p_name] = {
            "tax_rate": float(tax_rate or 0.0),
            "tax_paid_series": tax_paid,
            "realized_gains_series": realized_gains,
            "total_tax": total_tax,
            "tax_events": tax_events,
            "avg_tax_per_event": avg_per_event,
            "max_tax_single_event": max_event,
        }

    return out_df, tax_reports