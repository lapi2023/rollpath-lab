# src/commands/analyze_price_csv.py
from __future__ import annotations

"""
Analyze a two-column CSV (left: Date, right: Price), produce:
- Price line chart (PNG, linear)
- Price line chart (PNG, log-scale; skipped if any non-positive price)
- Daily / Monthly / Yearly return histograms (PNG)
- Console summary table (CAGR, annualized std, max/min daily return, max drawdown)
- Summary CSV export
Options:
 --start YYYY-MM-DD Inclusive start filter (applied before all computations)
 --end   YYYY-MM-DD Inclusive end filter
All labels, file names, and logs are in English.
"""
import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Tuple, Dict, Optional
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from rich.console import Console
from rich.table import Table

# ---- Unified reader (moved to utils) ----
from src.utils import read_price_csv_two_col
from src import settings

# ---------- Data IO ----------
def filter_by_date(df_price: pd.DataFrame, start: Optional[str], end: Optional[str]) -> pd.DataFrame:
    """
    Inclusive date filtering. If start/end is None, it is ignored.
    Raises if less than 2 rows remain after filtering.
    """
    df = df_price.copy()
    if start:
        start_dt = pd.to_datetime(start, errors="coerce")
        if pd.isna(start_dt):
            raise ValueError(f"Invalid --start date: {start}")
        df = df[df["Date"] >= start_dt]
    if end:
        end_dt = pd.to_datetime(end, errors="coerce")
        if pd.isna(end_dt):
            raise ValueError(f"Invalid --end date: {end}")
        df = df[df["Date"] <= end_dt]
    df = df.sort_values("Date").reset_index(drop=True)
    if len(df) < 2:
        raise ValueError("Not enough rows after applying start/end filters (need at least 2).")
    return df

# ---------- Return computation ----------
def compute_simple_returns(df_price: pd.DataFrame) -> Dict[str, pd.Series]:
    """
    Compute simple returns at daily, monthly, and yearly frequency.
    Daily: simple pct_change of the original daily series.
    Monthly/Yearly: resample by calendar end and compute pct_change of period-end prices.
    """
    df = df_price.set_index("Date").sort_index()
    # Daily simple returns
    r_d = df["Price"].pct_change(fill_method=None).dropna()
    # Monthly period returns (calendar month end)
    price_m = df["Price"].resample("ME").last()
    r_m = price_m.pct_change(fill_method=None).dropna()
    # Yearly period returns (calendar year end)
    price_y = df["Price"].resample("YE").last()
    r_y = price_y.pct_change(fill_method=None).dropna()
    return {"daily": r_d, "monthly": r_m, "yearly": r_y}

# ---------- Stats ----------
@dataclass
class SummaryMetrics:
    start_date: pd.Timestamp
    end_date: pd.Timestamp
    n_days: int
    years: float
    cagr: float
    ann_std: float
    max_return_d: float
    min_return_d: float
    max_drawdown: float
    price_min: float
    price_max: float

def max_drawdown_from_prices(price: pd.Series) -> float:
    """
    Compute Max Drawdown from a price series.
    Returns the max peak-to-trough decline (negative number, e.g., -0.57).
    """
    cummax = price.cummax()
    dd = price / cummax - 1.0
    return float(dd.min())

def compute_summary(df_price: pd.DataFrame, r_d: pd.Series) -> SummaryMetrics:
    """
    Compute summary metrics:
    - CAGR (based on first/last price and year fraction)
    - Annualized std of daily returns (× sqrt(252))
    - Max / Min daily return
    - Max drawdown (price path)
    - Price min/max (for reference)
    """
    start_dt = df_price["Date"].iloc[0]
    end_dt = df_price["Date"].iloc[-1]
    n_days = (end_dt - start_dt).days
    years = max(n_days / 365.25, 0.0)
    p0 = float(df_price["Price"].iloc[0])
    p1 = float(df_price["Price"].iloc[-1])
    cagr = (p1 / p0) ** (1.0 / years) - 1.0 if (years > 0 and p0 > 0) else np.nan
    ann_std = float(r_d.std() * np.sqrt(252.0)) if not r_d.empty else np.nan
    max_r = float(r_d.max()) if not r_d.empty else np.nan
    min_r = float(r_d.min()) if not r_d.empty else np.nan
    dd = max_drawdown_from_prices(df_price["Price"])
    pmin = float(df_price["Price"].min())
    pmax = float(df_price["Price"].max())
    return SummaryMetrics(
        start_date=start_dt,
        end_date=end_dt,
        n_days=n_days,
        years=years,
        cagr=float(cagr),
        ann_std=ann_std,
        max_return_d=max_r,
        min_return_d=min_r,
        max_drawdown=float(dd),
        price_min=pmin,
        price_max=pmax,
    )

# ---------- Plotting ----------
def plot_price_line(df_price: pd.DataFrame, title: str, out: Path, dpi: int = 150) -> Path:
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(df_price["Date"], df_price["Price"], color="#1f77b4", lw=1.2, label="Price")
    ax.set_title(title or "Price History", loc="left")
    ax.set_xlabel("Date")
    ax.set_ylabel("Price")
    ax.grid(True, lw=0.3, alpha=0.5)
    ax.legend(loc="upper left")
    fig.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=dpi)
    plt.close(fig)
    return out

def plot_price_line_log(
    df_price: pd.DataFrame,
    title: str,
    out: Path,
    dpi: int = 150,
    console: Optional[Console] = None,
) -> Optional[Path]:
    """
    Plot price on log scale. If any price <= 0, the log plot is skipped.
    Returns the saved path, or None when skipped.
    """
    if (df_price["Price"] <= 0).any():
        if console:
            console.print("[bold yellow]Skipped log-scale plot:[/] non-positive price detected (<= 0).")
        return None
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(df_price["Date"], df_price["Price"], color="#1f77b4", lw=1.2, label="Price")
    ax.set_yscale("log")
    ax.set_title((title or "Price History") + " [log scale]", loc="left")
    ax.set_xlabel("Date")
    ax.set_ylabel("Price (log scale)")
    ax.grid(True, lw=0.3, alpha=0.5, which="both")
    ax.legend(loc="upper left")
    fig.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=dpi)
    plt.close(fig)
    return out

def _hist(ax, data: pd.Series, bins: int, title: str):
    if data.empty:
        ax.text(0.5, 0.5, "No data", ha="center", va="center")
        ax.set_title(title, loc="left")
        return
    b = int(bins)
    ax.hist(data.dropna().to_numpy(), bins=b, color="#2ca02c", alpha=0.75, edgecolor="white")
    ax.set_title(title, loc="left")
    ax.set_xlabel("Return")
    ax.set_ylabel("Frequency")
    ax.grid(True, lw=0.3, alpha=0.5)

def plot_return_histograms(returns: Dict[str, pd.Series], title_prefix: str, outdir: Path, bins: int = 50, dpi: int = 150) -> Dict[str, Path]:
    outdir.mkdir(parents=True, exist_ok=True)
    paths = {}
    b = int(bins)
    # Daily
    fig, ax = plt.subplots(figsize=(7, 4))
    _hist(ax, returns["daily"], b, f"{title_prefix} — Daily Returns (bins={b})")
    fig.tight_layout()
    p = outdir / "hist_daily.png"
    fig.savefig(p, dpi=dpi); plt.close(fig)
    paths["daily"] = p
    # Monthly
    fig, ax = plt.subplots(figsize=(7, 4))
    _hist(ax, returns["monthly"], b, f"{title_prefix} — Monthly Returns (bins={b})")
    fig.tight_layout()
    p = outdir / "hist_monthly.png"
    fig.savefig(p, dpi=dpi); plt.close(fig)
    paths["monthly"] = p
    # Yearly
    fig, ax = plt.subplots(figsize=(7, 4))
    _hist(ax, returns["yearly"], b, f"{title_prefix} — Yearly Returns (bins={b})")
    fig.tight_layout()
    p = outdir / "hist_yearly.png"
    fig.savefig(p, dpi=dpi); plt.close(fig)
    paths["yearly"] = p
    return paths

# ---------- Console & CSV ----------
def print_summary_table(console: Console, s: SummaryMetrics) -> None:
    tb = Table(title="Summary Metrics", show_lines=False)
    tb.add_column("Field", style="bold")
    tb.add_column("Value")
    tb.add_row("Start Date", str(s.start_date.date()))
    tb.add_row("End Date", str(s.end_date.date()))
    tb.add_row("Span (days)", f"{s.n_days:,}")
    tb.add_row("Span (years)", f"{s.years:.3f}")
    tb.add_row("CAGR", f"{s.cagr:.4%}" if np.isfinite(s.cagr) else "NaN")
    tb.add_row("Annualized Std (from daily)", f"{s.ann_std:.4%}" if np.isfinite(s.ann_std) else "NaN")
    tb.add_row("Max Daily Return", f"{s.max_return_d:.4%}" if np.isfinite(s.max_return_d) else "NaN")
    tb.add_row("Min Daily Return", f"{s.min_return_d:.4%}" if np.isfinite(s.min_return_d) else "NaN")
    tb.add_row("Max Drawdown", f"{s.max_drawdown:.4%}" if np.isfinite(s.max_drawdown) else "NaN")
    tb.add_row("Price Min", f"{s.price_min:,.6f}")
    tb.add_row("Price Max", f"{s.price_max:,.6f}")
    console.print(tb)

def save_summary_csv(s: SummaryMetrics, out_path: Path) -> Path:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(
        {
            "start_date": [s.start_date],
            "end_date": [s.end_date],
            "span_days": [s.n_days],
            "span_years": [s.years],
            "cagr": [s.cagr],
            "ann_std_from_daily": [s.ann_std],
            "max_daily_return": [s.max_return_d],
            "min_daily_return": [s.min_return_d],
            "max_drawdown": [s.max_drawdown],
            "price_min": [s.price_min],
            "price_max": [s.price_max],
        }
    )
    df.to_csv(out_path, index=False)
    return out_path

# ---------- CLI ----------
def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description="Analyze a (Date, Price) CSV: price plot (linear & log), return histograms, and summary metrics."
    )
    ap.add_argument("--csv", required=True, help="Path to CSV (left=Date, right=Price).")
    ap.add_argument("--outdir", default=str(Path(settings.OUTPUT_DIR) / "analyze"), help="Directory to save charts and tables.")
    ap.add_argument("--title", default="", help="Chart title prefix.")
    ap.add_argument("--bins", type=int, default=50, help="Histogram bins.")
    ap.add_argument("--dpi", type=int, default=150, help="Figure DPI.")
    ap.add_argument("--start", type=str, default=None, help="Inclusive start date, e.g., 1990-01-01.")
    ap.add_argument("--end", type=str, default=None, help="Inclusive end date, e.g., 2025-12-31.")
    args = ap.parse_args(argv)

    console = Console()
    csv_path = Path(args.csv)
    outdir = Path(args.outdir).resolve()
    outdir.mkdir(parents=True, exist_ok=True)

    console.print(f"[bold cyan]Input[/]: {csv_path}")
    console.print(f"[bold cyan]Output Dir[/]: {outdir}")
    if args.start or args.end:
        console.print(f"[bold cyan]Filter[/]: start={args.start or '-'} end={args.end or '-'}")

    # Load -> filter -> compute
    df_price = read_price_csv_two_col(csv_path)
    df_price = filter_by_date(df_price, args.start, args.end)
    rets = compute_simple_returns(df_price)
    summary = compute_summary(df_price, rets["daily"])

    # Price plot (linear + log)
    title = args.title or f"Price History ({csv_path.name})"
    plot_price_line(df_price=df_price, title=title, out=outdir / "price.png", dpi=args.dpi)
    plot_price_line_log(df_price=df_price, title=title, out=outdir / "price_log.png", dpi=args.dpi, console=console)

    # Histograms
    plot_return_histograms(
        returns=rets,
        title_prefix=args.title or "Return Distribution",
        outdir=outdir,
        bins=args.bins,
        dpi=args.dpi,
    )

    # Console + CSV
    print_summary_table(console, summary)
    saved_csv = save_summary_csv(summary, outdir / "summary_metrics.csv")
    console.print(f"[green]Saved summary:[/] {saved_csv}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())