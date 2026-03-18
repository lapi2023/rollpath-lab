# src/commands/compare_actual_vs_simulated.py
"""
Compare actual ETF price history against the latest simulated ETF series.

This module:
- finds the latest actual ETF CSV for each symbol,
- finds the latest simulated ETF CSV for each symbol,
- aligns both series on overlapping dates,
- normalizes both to 1.0 at the first overlap,
- computes divergence metrics,
- saves charts.

Supported symbols are derived from `src.settings`.

Notes
-----
- Actual CSVs are resolved using `settings.ACTUAL_ETF_FILE_GLOBS`.
- Simulated CSVs are resolved using `settings.SIMULATION_SPECS[symbol]["output_prefix"]`.
- File discovery uses the shared case-insensitive `find_latest_file()` helper from
  `generate_leveraged_etf.py`.
"""

from __future__ import annotations

import argparse
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import font_manager

from src import settings
from src.commands.generate_leveraged_etf import find_latest_file
from src.utils import read_price_csv_two_col

try:
    from src.visualizer import save_with_watermarks
except Exception:
    def save_with_watermarks(path: Path, dpi: int = 150):
        plt.savefig(path, dpi=dpi)


# ---------------------------------------------------------------------------
# Font helper
# ---------------------------------------------------------------------------
def set_cjk_font(prefer: list[str] | None = None) -> Optional[str]:
    """
    Pick an available CJK font to avoid missing glyphs (best effort).
    """
    candidates = prefer or [
        "Yu Gothic UI",
        "MS Gothic",
        "MS UI Gothic",
        "Hiragino Sans",
        "Hiragino Kaku Gothic ProN",
        "Noto Sans CJK JP",
        "IPAexGothic",
        "IPAGothic",
    ]
    available = {f.name for f in font_manager.fontManager.ttflist}
    for family in candidates:
        if family in available:
            plt.rcParams["font.family"] = family
            plt.rcParams["axes.unicode_minus"] = False
            return family

    plt.rcParams["axes.unicode_minus"] = False
    return None


# ---------------------------------------------------------------------------
# Metadata structures
# ---------------------------------------------------------------------------
@dataclass
class SimMeta:
    """
    Metadata parsed from a simulated series filename.
    """
    path: Path
    div_tag: Optional[str]
    model_tag: Optional[str]
    cost_annual: Optional[float]
    borrow_alpha: Optional[float]
    borrow_beta_annual: Optional[float]
    carry_annual: Optional[float]


@dataclass
class ActualMeta:
    """
    Metadata for an actual ETF series file.
    """
    path: Path
    tag: Optional[str]


@dataclass
class CompareResult:
    """
    Final comparison output for one symbol.
    """
    symbol: str
    actual_meta: ActualMeta
    simulated_meta: SimMeta
    mae_divergence_pct: float
    final_divergence_pct: float
    df_aligned: pd.DataFrame


# ---------------------------------------------------------------------------
# File parsing helpers
# ---------------------------------------------------------------------------
def parse_simulated_filename(path: Path) -> SimMeta:
    """
    Parse metadata from a simulated filename.

    Expected examples:
    - ^tqqq_simulated_d_TR_constant_effective_carry_0.9700%cost_a1.20_b0.10%_carry2.00%_1999-01-01_2026-03-17.csv
    - ^spxl_simulated_d_TR_dividend_carry_1.0220%cost_a1.07_b0.45%_1928-01-01_2026-03-17.csv
    """
    name = path.name

    div_tag_match = re.search(r"_(TR|PX)_", name, flags=re.IGNORECASE)
    div_tag = div_tag_match.group(1).upper() if div_tag_match else None

    model_match = re.search(
        r"_(dividend_carry|constant_effective_carry)_",
        name,
        flags=re.IGNORECASE,
    )
    model_tag = model_match.group(1) if model_match else None

    cost_match = re.search(r"_([0-9]+(?:\.[0-9]+)?)%cost", name, flags=re.IGNORECASE)
    cost_annual = float(cost_match.group(1)) / 100.0 if cost_match else None

    borrow_match = re.search(
        r"_a([0-9]+(?:\.[0-9]+)?)_b([0-9]+(?:\.[0-9]+)?)%",
        name,
        flags=re.IGNORECASE,
    )
    borrow_alpha = float(borrow_match.group(1)) if borrow_match else None
    borrow_beta_annual = float(borrow_match.group(2)) / 100.0 if borrow_match else None

    carry_match = re.search(
        r"_carry([0-9]+(?:\.[0-9]+)?)%",
        name,
        flags=re.IGNORECASE,
    )
    carry_annual = float(carry_match.group(1)) / 100.0 if carry_match else None

    return SimMeta(
        path=path,
        div_tag=div_tag,
        model_tag=model_tag,
        cost_annual=cost_annual,
        borrow_alpha=borrow_alpha,
        borrow_beta_annual=borrow_beta_annual,
        carry_annual=carry_annual,
    )


def parse_actual_filename_tag(path: Path) -> ActualMeta:
    """
    Try to extract a simple tag from the actual ETF filename.
    """
    name = path.name
    match = re.search(r"_daily_([A-Za-z0-9]+)\.csv$", name, flags=re.IGNORECASE)
    tag = match.group(1) if match else None
    if tag and tag.upper() in {"TR", "PX"}:
        tag = tag.upper()
    return ActualMeta(path=path, tag=tag)


def latest_simulated_for_symbol(symbol: str) -> SimMeta:
    """
    Return the latest simulated file for one symbol.
    """
    spec = settings.SIMULATION_SPECS[symbol]
    prefix = str(spec["output_prefix"])
    path = find_latest_file(
        f"{prefix}_*.csv",
        base_dir=settings.DATA_DIR,
    )
    return parse_simulated_filename(path)


def latest_actual_for_symbol(symbol: str) -> ActualMeta:
    """
    Return the latest actual ETF file for one symbol.
    """
    actual_glob = settings.ACTUAL_ETF_FILE_GLOBS[symbol]
    path = find_latest_file(
        actual_glob,
        base_dir=settings.DATA_DIR,
        exclude_substrings=settings.EXCLUDE_FILENAME_SUBSTRINGS,
    )
    return parse_actual_filename_tag(path)


# ---------------------------------------------------------------------------
# Comparison core
# ---------------------------------------------------------------------------
def compute_comparison(
    symbol: str,
    df_actual: pd.DataFrame,
    df_sim: pd.DataFrame,
) -> Tuple[pd.DataFrame, float, float]:
    """
    Align two series, normalize both to 1.0, and compute divergence metrics.
    """
    actual = df_actual.rename(columns={"Price": "actual"})
    simulated = df_sim.rename(columns={"Price": "simulated"})

    merged = pd.merge(actual, simulated, on="Date", how="inner")
    if merged.empty:
        raise ValueError(f"No overlapping dates between actual and simulated for {symbol}.")

    merged = merged.sort_values("Date").reset_index(drop=True)
    merged["norm_actual"] = merged["actual"] / float(merged["actual"].iloc[0])
    merged["norm_sim"] = merged["simulated"] / float(merged["simulated"].iloc[0])
    merged["divergence_pct"] = (merged["norm_sim"] / merged["norm_actual"] - 1.0) * 100.0

    mae = float(np.mean(np.abs(merged["divergence_pct"].to_numpy())))
    final_divergence = float(merged["divergence_pct"].iloc[-1])

    return merged, mae, final_divergence


# ---------------------------------------------------------------------------
# Plot labels
# ---------------------------------------------------------------------------
def format_borrow(meta: SimMeta) -> str:
    """
    Format borrow metadata for chart subtitles.
    """
    parts = []
    if meta.borrow_alpha is not None:
        parts.append(f"a={meta.borrow_alpha:.4f}")
    if meta.borrow_beta_annual is not None:
        parts.append(f"b={meta.borrow_beta_annual * 100:.4f}%/yr")
    if not parts:
        return ""
    return " Borrow: " + ", ".join(parts)


def format_carry(meta: SimMeta) -> str:
    """
    Format carry metadata for chart subtitles.
    """
    if meta.carry_annual is None:
        return ""
    return f" Carry: {meta.carry_annual * 100:.4f}%/yr"


def title_suffix(sim_meta: SimMeta, actual_meta: ActualMeta) -> str:
    """
    Build a subtitle suffix describing actual/simulated series metadata.
    """
    left = "Sim: Mode=" + (sim_meta.div_tag if sim_meta.div_tag else "Unknown")
    if sim_meta.model_tag:
        left += ", Model=" + sim_meta.model_tag
    if sim_meta.cost_annual is not None:
        left += ", Cost=" + f"{sim_meta.cost_annual * 100:.4f}%/yr"
    left += format_borrow(sim_meta)
    left += format_carry(sim_meta)

    right = " Actual: " + (actual_meta.tag if actual_meta.tag else "Unknown")
    return left + right


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------
def plot_divergence(symbol: str, result: CompareResult, outdir: Path) -> Path:
    """
    Plot divergence (%) over time.
    """
    aligned = result.df_aligned
    mae = result.mae_divergence_pct
    final = result.final_divergence_pct

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(aligned["Date"], aligned["divergence_pct"], label="Divergence (%)")
    ax.axhline(0.0, color="black", linewidth=0.8)

    ax.set_ylabel("Divergence (%)")
    ax.set_xlabel("Date")
    ax.set_title(
        f"Actual vs Simulated {symbol} Divergence (%)\n"
        f"MAE={mae:.3f}% | Final={final:+.3f}%\n"
        f"{title_suffix(result.simulated_meta, result.actual_meta)}"
    )
    ax.legend(loc="upper left")
    ax.grid(True, linewidth=0.3, alpha=0.5)

    outpath = outdir / f"{symbol}_divergence.png"
    fig.tight_layout()
    save_with_watermarks(outpath, dpi=150)
    plt.close(fig)
    return outpath


def plot_normalized(symbol: str, result: CompareResult, outdir: Path) -> Path:
    """
    Plot normalized actual vs simulated series (both start at 1.0).
    """
    aligned = result.df_aligned
    final = result.final_divergence_pct

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(
        aligned["Date"],
        aligned["norm_actual"],
        label=f"{symbol} Actual (start=1)",
    )
    ax.plot(
        aligned["Date"],
        aligned["norm_sim"],
        label=f"{symbol} Simulated (start=1)",
    )

    x_last = aligned["Date"].iloc[-1]
    y_last = aligned["norm_sim"].iloc[-1]
    ax.annotate(
        f"Final divergence: {final:+.3f}%",
        xy=(x_last, y_last),
        xytext=(0, 20),
        textcoords="offset points",
        arrowprops=dict(arrowstyle="->", lw=0.8),
    )

    ax.set_ylabel("Index")
    ax.set_xlabel("Date")
    ax.set_title(
        f"Actual vs Simulated {symbol} Normalized Price (Start=1)\n"
        f"{title_suffix(result.simulated_meta, result.actual_meta)}"
    )
    ax.legend(loc="upper left")
    ax.grid(True, linewidth=0.3, alpha=0.5)

    outpath = outdir / f"{symbol}_normalized.png"
    fig.tight_layout()
    save_with_watermarks(outpath, dpi=150)
    plt.close(fig)
    return outpath


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------
def run_for_symbol(
    symbol: str,
    outdir: Path,
    start: Optional[pd.Timestamp],
    end: Optional[pd.Timestamp],
) -> CompareResult:
    """
    Run the full comparison pipeline for one symbol and save charts.

    If start is not provided, this function automatically uses the latest
    available start date across the actual and simulated series.
    """
    actual_meta = latest_actual_for_symbol(symbol)
    simulated_meta = latest_simulated_for_symbol(symbol)

    df_actual = read_price_csv(actual_meta.path)
    df_sim = read_price_csv(simulated_meta.path)

    if end is not None:
        df_actual = read_price_csv_two_col(actual_meta.path)
        df_sim = read_price_csv_two_col(simulated_meta.path)

    start_eff = start
    if start_eff is None:
        candidates: list[pd.Timestamp] = []
        if not df_actual.empty:
            candidates.append(pd.to_datetime(df_actual["Date"].min()))
        if not df_sim.empty:
            candidates.append(pd.to_datetime(df_sim["Date"].min()))
        start_eff = max(candidates) if candidates else None

    if start_eff is not None:
        df_actual = df_actual[df_actual["Date"] >= start_eff].reset_index(drop=True)
        df_sim = df_sim[df_sim["Date"] >= start_eff].reset_index(drop=True)

    df_aligned, mae, final = compute_comparison(symbol, df_actual, df_sim)

    result = CompareResult(
        symbol=symbol,
        actual_meta=actual_meta,
        simulated_meta=simulated_meta,
        mae_divergence_pct=mae,
        final_divergence_pct=final,
        df_aligned=df_aligned,
    )

    outdir.mkdir(parents=True, exist_ok=True)
    divergence_path = plot_divergence(symbol, result, outdir)
    normalized_path = plot_normalized(symbol, result, outdir)

    print("=== Comparison summary ===")
    print(
        f"{symbol} actual range: {df_actual['Date'].min().date()} ~ {df_actual['Date'].max().date()} "
        f"(rows={len(df_actual)})"
    )
    print(
        f"{symbol} simulated range: {df_sim['Date'].min().date()} ~ {df_sim['Date'].max().date()} "
        f"(rows={len(df_sim)})"
    )
    print(f"{symbol} final divergence: {final:+.3f}% | MAE: {mae:.3f}%")
    print(f"Saved plots: {divergence_path.name}, {normalized_path.name}\n")

    return result

# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """
    Parse command-line arguments.
    """
    default_outdir = Path(settings.OUTPUT_DIR / "compare").resolve()
    ap = argparse.ArgumentParser(
        description="Compare actual ETF series against simulated ETF series."
    )
    ap.add_argument(
        "--outdir",
        type=str,
        default=str(default_outdir),
        help="Directory where comparison charts will be saved.",
    )
    ap.add_argument(
        "--symbols",
        nargs="+",
        default=settings.SIMULATION_DEFAULT_SYMBOLS,
        choices=sorted(settings.SIMULATION_SPECS.keys()),
        help="Symbols to compare.",
    )
    ap.add_argument(
        "--start",
        type=str,
        default=None,
        help="Inclusive start date, e.g. 2010-01-01.",
    )
    ap.add_argument(
        "--end",
        type=str,
        default=None,
        help="Inclusive end date, e.g. 2026-12-31.",
    )
    return ap.parse_args(argv)

def main(argv: list[str] | None = None) -> int:
    """
    CLI entry point.
    """
    args = parse_args(argv)

    chosen_font = set_cjk_font()
    if chosen_font:
        print(f"[info] Using CJK font: {chosen_font}")
    else:
        print("[warn] No CJK font found. You may see missing-glyph warnings.")

    outdir = Path(args.outdir)
    start_ts = pd.to_datetime(args.start) if args.start else None
    end_ts = pd.to_datetime(args.end) if args.end else None

    print(f"Saving comparison charts to: {outdir}")
    for symbol in args.symbols:
        run_for_symbol(symbol, outdir, start=start_ts, end=end_ts)

    return 0

if __name__ == "__main__":
    raise SystemExit(main())