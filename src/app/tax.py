
"""Tax-related helpers (safe strings for Python 3.11)."""
from __future__ import annotations
import numpy as np
import pandas as pd
from rich.console import Console


def print_tax_assumptions(console: Console, tax_rate: float, rebalance: str) -> None:
    console.print("[bold yellow]Tax Model Assumptions:[/]")
    console.print(" • Tax rate on realized rebalancing gains: [bold]" + f"{tax_rate:.2%}" + "[/]")
    console.print(" • Applied only when rebalancing occurs (no tax for 'rebalance = none').")
    console.print(" • Average-cost basis within each asset (proportional basis).")
    console.print(" • Taxes are paid out of the portfolio at the rebalance time.")
    console.print(" • No short/long distinction, no dividend/interest tax, no loss harvesting.")
    console.print("")


def rolling_tax_window_sum(tax_series: pd.Series, window: int) -> np.ndarray:
    x = tax_series.to_numpy(dtype=float)
    N = len(x)
    if N < window:
        return np.array([], dtype=float)
    c = np.cumsum(x, dtype=float)
    s = c[window - 1:]
    s[1:] = s[1:] - c[:-window]
    return s
