"""
Cash series injection utilities (flat, fixed annual, or risk-free based),
extracted from the original main.py for clarity and reuse.
"""
from __future__ import annotations
from typing import Dict
import pandas as pd
from rich.console import Console


from src.utils import _infer_ppy_from_freq
try:
    from src.dividend_loader import (
        DividendInputPaths,
        load_monthly_dividend_and_riskfree,
        to_daily_series_from_monthly,
    )
    from src import settings
    _HAS_RF_LOADER = True
except Exception:
    _HAS_RF_LOADER = False
    settings = None  # type: ignore


def _portfolios_require_cash(portfolios: Dict[str, Dict[str, float]]) -> bool:
    for w in portfolios.values():
        for k in w.keys():
            if str(k).strip().upper() == "CASH":
                return True
    return False


def _inject_cash_series_flat(returns_pd: pd.DataFrame, console: Console) -> pd.DataFrame:
    df = returns_pd.copy()
    df["Return_CASH"] = 0.0
    console.print("[bold green]CASH mode:[/] flat (0% nominal return per period)")
    return df


def _inject_cash_series_fixed(returns_pd: pd.DataFrame, freq: str, annual_rate: float, console: Console) -> pd.DataFrame:
    df = returns_pd.copy()
    ppy = _infer_ppy_from_freq(freq)
    per = (1.0 + float(annual_rate)) ** (1.0 / ppy) - 1.0
    df["Return_CASH"] = per
    console.print(f"[bold green]CASH mode:[/] fixed (annual={annual_rate:.2%} → per-period={per:.6%}, ppy={ppy})")
    return df


def _inject_cash_series_rf(returns_pd: pd.DataFrame, freq: str, console: Console) -> pd.DataFrame:
    if not _HAS_RF_LOADER or settings is None:
        console.print("[bold red]CASH mode 'rf' unavailable:[/] Falling back to flat.")
        return _inject_cash_series_flat(returns_pd, console)
    try:
        paths = DividendInputPaths(
            sp500_dividend_monthly_csv=max((settings.DATA_DIR / "raw").glob("SPX Dividend Yield by Month_*.csv")),
            tbill_1920_1934_monthly_csv=max((settings.DATA_DIR / "raw").glob("Yields on Short-Term United States Securities*.csv")),
            tb3ms_1934_now_monthly_csv=max((settings.DATA_DIR / "raw").glob("3-Month Treasury Bill Secondary Market Rate, Discount Basis (TB3MS)*.csv")),
        )
    except Exception:
        console.print("[bold red]CASH mode 'rf' inputs missing; falling back to flat.[/]")
        return _inject_cash_series_flat(returns_pd, console)

    try:
        _, rf_m = load_monthly_dividend_and_riskfree(paths)
    except Exception:
        console.print("[bold red]RF monthly load failed. Falling back to flat.[/]")
        return _inject_cash_series_flat(returns_pd, console)

    df = returns_pd.copy()
    dates = pd.to_datetime(df["Date"])
    f = (freq or "").lower()
    if f in ("", "daily", "day"):
        rf_daily = to_daily_series_from_monthly(rf_m, "RiskFree", dates).astype(float)
        dt_days = dates.diff().dt.days.fillna(0).astype(float)
        dt_years = dt_days / 365.0
        df["Return_CASH"] = rf_daily.values * dt_years.values
        console.print("[bold green]CASH mode:[/] rf (daily): per-period r = RF_annual × Δt_years")
    elif f in ("monthly", "month"):
        ym = dates.dt.to_period("M").astype(str)
        m = rf_m.set_index("YearMonth")["RiskFree"]
        rf_ann = ym.map(m).astype(float).fillna(method="ffill").fillna(method="bfill")
        df["Return_CASH"] = rf_ann.values * (1.0 / 12.0)
        console.print("[bold green]CASH mode:[/] rf (monthly)")
    elif f in ("yearly", "year"):
        ym = dates.dt.to_period("M").astype(str)
        m = rf_m.set_index("YearMonth")["RiskFree"]
        rf_ann = ym.map(m).astype(float).fillna(method="ffill").fillna(method="bfill")
        df["Return_CASH"] = rf_ann.values * 1.0
        console.print("[bold green]CASH mode:[/] rf (yearly)")
    else:
        console.print(f"[bold yellow]Unknown freq='{freq}', defaulting CASH to flat.[/]")
        return _inject_cash_series_flat(returns_pd, console)
    return df


def inject_cash_if_needed(
    returns_pd: pd.DataFrame,
    portfolios: Dict[str, Dict[str, float]],
    freq: str,
    cash_mode: str,
    cash_fixed_rate: float,
    console: Console,
) -> pd.DataFrame:
    if not _portfolios_require_cash(portfolios):
        return returns_pd
    cm = (cash_mode or "flat").lower()
    if cm == "flat":
        return _inject_cash_series_flat(returns_pd, console)
    if cm == "fixed":
        return _inject_cash_series_fixed(returns_pd, freq=freq, annual_rate=cash_fixed_rate, console=console)
    if cm == "rf":
        return _inject_cash_series_rf(returns_pd, freq=freq, console=console)
    console.print(f"[bold yellow]Unknown --cash-mode='{cash_mode}', defaulting to 'flat'.[/]")
    return _inject_cash_series_flat(returns_pd, console)
