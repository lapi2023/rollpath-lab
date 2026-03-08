
# src/commands/compare_actual_vs_simulated.py
from __future__ import annotations
import argparse
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import font_manager

from src.settings import OUTPUT_DIR

try:
    from src.visualizer import save_with_watermarks  # project helper (if available)
except Exception:
    def save_with_watermarks(path: Path, dpi: int = 150):
        plt.savefig(path, dpi=dpi)

try:
    from src.settings import DATA_DIR
except Exception:
    DATA_DIR = Path('data')


def set_cjk_font(prefer: list[str] | None = None) -> Optional[str]:
    """Pick an available CJK font to avoid missing glyphs (best-effort)."""
    candidates = prefer or [
        "Yu Gothic UI", "MS Gothic", "MS UI Gothic",
        "Hiragino Sans", "Hiragino Kaku Gothic ProN",
        "Noto Sans CJK JP", "IPAexGothic", "IPAGothic",
    ]
    available = {f.name for f in font_manager.fontManager.ttflist}
    for fam in candidates:
        if fam in available:
            plt.rcParams["font.family"] = fam
            plt.rcParams["axes.unicode_minus"] = False
            return fam
    plt.rcParams["axes.unicode_minus"] = False
    return None


def _find_latest_in_data(glob_pattern: str) -> Path:
    base = Path(DATA_DIR)
    cands = list(base.glob(glob_pattern))
    if not cands:
        raise FileNotFoundError(f"No files matched under data/: {glob_pattern}")
    return max(cands, key=lambda p: p.stat().st_mtime)


@dataclass
class SimMeta:
    path: Path
    div_tag: Optional[str]
    cost_annual: float
    borrow_alpha: Optional[float] = None
    borrow_beta_annual: Optional[float] = None


def _parse_simulated_filename(p: Path) -> SimMeta:
    name = p.name
    m_tag = re.search(r"_(TR|PX)_", name)
    div_tag = m_tag.group(1) if m_tag else None
    m_cost = re.search(r"_([0-9]+(?:\.[0-9]+)?)%cost", name)
    if not m_cost:
        raise ValueError(f"Cost pattern not found in: {name}")
    cost_pct = float(m_cost.group(1))
    m_borrow = re.search(r"_a([0-9]+(?:\.[0-9]+)?)_b([0-9]+(?:\.[0-9]+)?)%", name)
    alpha = float(m_borrow.group(1)) if m_borrow else None
    beta_frac = float(m_borrow.group(2)) / 100.0 if m_borrow else None
    return SimMeta(path=p, div_tag=div_tag, cost_annual=cost_pct / 100.0,
                   borrow_alpha=alpha, borrow_beta_annual=beta_frac)


def _latest_simulated(prefix: str) -> SimMeta:
    latest = _find_latest_in_data(f"{prefix}_*%cost*.csv")
    return _parse_simulated_filename(latest)


@dataclass
class ActualMeta:
    path: Path
    tag: Optional[str]


def _parse_actual_filename_tag(p: Path) -> ActualMeta:
    name = p.name
    m = re.search(r"_daily_([A-Za-z0-9]+)\.csv$", name)
    tag = m.group(1) if m else None
    if tag and tag.upper() in {"TR", "PX"}:
        tag = tag.upper()
    return ActualMeta(path=p, tag=tag)


def _read_price_csv(p: Path) -> pd.DataFrame:
    df = pd.read_csv(p)
    if "Date" not in df.columns:
        raise ValueError(f"'Date' column not found in {p.name}.")
    price_col = None
    for cand in ("Adj Close", "Price", "Close"):
        if cand in df.columns:
            price_col = cand
            break
    if price_col is None:
        lc = {c.lower(): c for c in df.columns}
        for cand in ("adj close", "price", "close"):
            if cand in lc:
                price_col = lc[cand]
                break
    if price_col is None:
        raise ValueError(f"Price-like column not found in {p.name}.")
    dt = pd.to_datetime(df["Date"], errors="coerce", utc=True)
    df["Date"] = dt.dt.tz_convert(None).dt.floor("D")
    df[price_col] = pd.to_numeric(df[price_col], errors="coerce")
    df = df.dropna(subset=["Date", price_col]).sort_values("Date").reset_index(drop=True)
    return df[["Date", price_col]].rename(columns={price_col: "Price"})


@dataclass
class CompareResult:
    symbol: str
    actual_meta: ActualMeta
    simulated_meta: SimMeta
    mae_divergence_pct: float
    final_divergence_pct: float
    df_aligned: pd.DataFrame


def _compute_comparison(symbol: str, df_actual: pd.DataFrame, df_sim: pd.DataFrame) -> Tuple[pd.DataFrame, float, float]:
    a = df_actual.rename(columns={"Price": "actual"})
    s = df_sim.rename(columns={"Price": "simulated"})
    m = pd.merge(a, s, on="Date", how="inner")
    if m.empty:
        raise ValueError(f"No overlapping dates between actual and simulated for {symbol}.")
    m["norm_actual"] = m["actual"] / float(m["actual"].iloc[0])
    m["norm_sim"] = m["simulated"] / float(m["simulated"].iloc[0])
    m["divergence_pct"] = (m["norm_sim"] / m["norm_actual"] - 1.0) * 100.0
    mae = float(np.mean(np.abs(m["divergence_pct"].to_numpy())))
    final = float(m["divergence_pct"].iloc[-1])
    return m, mae, final


def _fmt_borrow(meta: SimMeta) -> str:
    parts = []
    if meta.borrow_alpha is not None:
        parts.append("a=" + f"{meta.borrow_alpha:.2f}")
    if meta.borrow_beta_annual is not None:
        parts.append("b=" + f"{meta.borrow_beta_annual * 100:.2f}%/yr")
    if not parts:
        return ""
    return " Borrow: " + ", ".join(parts)


def _title_suffix(sim_meta: SimMeta, actual_meta: ActualMeta) -> str:
    left = "Sim: Mode=" + (sim_meta.div_tag if sim_meta.div_tag in {"TR", "PX"} else "Unknown")
    left += ", Cost=" + f"{sim_meta.cost_annual * 100:.2f}%"
    right = " Actual: " + (actual_meta.tag if actual_meta.tag else "Unknown")
    return left + _fmt_borrow(sim_meta) + right


def _plot_divergence(symbol: str, res: CompareResult, outdir: Path) -> Path:
    m = res.df_aligned
    mae = res.mae_divergence_pct
    final = res.final_divergence_pct
    sim_meta = res.simulated_meta
    actual_meta = res.actual_meta

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(m["Date"], m["divergence_pct"], label="Divergence (%)")
    ax.axhline(0.0, color="black", linewidth=0.8)
    ax.set_ylabel("Divergence (%)")
    ax.set_xlabel("Date")
    title_main = "Actual vs Simulated " + symbol + " Divergence (%) "
    title_stats = "MAE=" + f"{mae:.2f}%" + ", Final=" + f"{final:+.2f}%"
    title = title_main + title_stats + "\n" + _title_suffix(sim_meta, actual_meta)
    ax.set_title(title)
    ax.legend(loc="upper left")
    ax.grid(True, linewidth=0.3, alpha=0.5)
    outpath = outdir / f"{symbol}_divergence.png"
    fig.tight_layout()
    save_with_watermarks(outpath, dpi=150)
    return outpath


def _plot_normalized(symbol: str, res: CompareResult, outdir: Path) -> Path:
    m = res.df_aligned
    final = res.final_divergence_pct
    sim_meta = res.simulated_meta
    actual_meta = res.actual_meta

    fig, ax = plt.subplots(figsize=(10, 5))
    label_actual = symbol + " Actual (" + (actual_meta.tag if actual_meta.tag else "Unknown") + ", start=1)"
    label_sim = symbol + " Simulated (" + (sim_meta.div_tag if sim_meta.div_tag else "Unknown") + ", start=1)"
    ax.plot(m["Date"], m["norm_actual"], label=label_actual)
    ax.plot(m["Date"], m["norm_sim"], label=label_sim)
    ax.set_ylabel("Index")
    ax.set_xlabel("Date")
    x = m["Date"].iloc[-1]
    y = m["norm_sim"].iloc[-1]
    ann = "Final divergence: " + f"{final:+.2f}%"
    ax.annotate(ann, xy=(x, y), xytext=(0, 20), textcoords="offset points",
                arrowprops=dict(arrowstyle="->", lw=0.8))
    title = "Actual vs Simulated " + symbol + " Normalized Price (Start=1)\n"  + _title_suffix(sim_meta, actual_meta)
    ax.set_title(title)
    ax.legend(loc="upper left")
    ax.grid(True, linewidth=0.3, alpha=0.5)
    outpath = outdir / f"{symbol}_normalized.png"
    fig.tight_layout()
    save_with_watermarks(outpath, dpi=150)
    return outpath


def _run_for_symbol(symbol: str, actual_glob: str, sim_prefix: str, outdir: Path) -> CompareResult:
    actual_path = _find_latest_in_data(actual_glob)
    actual_meta = _parse_actual_filename_tag(actual_path)
    sim_meta = _latest_simulated(sim_prefix)
    df_actual = _read_price_csv(actual_path)
    df_sim = _read_price_csv(sim_meta.path)
    df_aligned, mae, final = _compute_comparison(symbol, df_actual, df_sim)
    res = CompareResult(symbol=symbol, actual_meta=actual_meta, simulated_meta=sim_meta,
                        mae_divergence_pct=mae, final_divergence_pct=final, df_aligned=df_aligned)
    outdir.mkdir(parents=True, exist_ok=True)
    div_path = _plot_divergence(symbol, res, outdir)
    norm_path = _plot_normalized(symbol, res, outdir)
    print("=== Comparison summary ===")
    print(symbol + " actual range: " + str(df_actual['Date'].min().date()) + " ~ " + str(df_actual['Date'].max().date()) + " (rows=" + str(len(df_actual)) + ")")
    print(symbol + " sim range: " + str(df_sim['Date'].min().date()) + " ~ " + str(df_sim['Date'].max().date()) + " (rows=" + str(len(df_sim)) + ")")
    print(symbol + " Final divergence: " + f"{final:+.3f}%" + " MAE: " + f"{mae:.3f}%")
    print("Saved plots: " + div_path.name + ", " + norm_path.name + "\n")
    return res


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    default_outdir = Path(OUTPUT_DIR / Path('compare')).resolve()
    ap.add_argument("--outdir", type=str, default=str(default_outdir), help="Directory to save charts")
    ap.add_argument("--symbols", type=str, nargs="+", default=["SPXL", "SSO"], choices=["SPXL", "SSO"])
    args = ap.parse_args(argv)

    chosen = set_cjk_font()
    if chosen:
        print("[info] Using CJK font: " + chosen)
    else:
        print("[warn] No CJK font found. You may see missing-glyph warnings.")

    outdir = Path(args.outdir)
    if "SPXL" in args.symbols:
        _run_for_symbol("SPXL", "SPXL_*_daily_*.csv", "^spxl_simulated_d", outdir)
    if "SSO" in args.symbols:
        _run_for_symbol("SSO", "SSO_*_daily_*.csv", "^sso_simulated_d", outdir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
