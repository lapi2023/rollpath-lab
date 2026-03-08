# rollpath-lab

**rollpath-lab** is a research‑grade toolkit for **rolling-window portfolio analytics** and **leveraged ETF simulation**. It supports **DCA (dollar-cost averaging) and Lump‑Sum** workflows, **tax-aware rebalancing**, GPU‑accelerated rolling metrics, and end‑to‑end visualization & reporting.

<p align="center">
  <img alt="license" src="https://img.shields.io/badge/license-MIT-blue.svg" />
  <img alt="python" src="https://img.shields.io/badge/python-3.10%2B-orange" />
  <img alt="gpu" src="https://img.shields.io/badge/GPU-CuPy%20optional-00B894" />
</p>

---

## Key Features

- **Rolling-window analytics**: return/CAGR/risk/Sharpe, path min/max, drawups/drawdowns; median/mean “typical VALUE paths”, and representative percentile paths (Min/P25/Med/P75/Max).
- **DCA & Lump‑Sum**: value-path aware extrema for DCA (contributions at period start, none at the last step), median IRR from the DCA cash‑flow schedule.
- **Tax-aware rebalancing**: average‑cost basis; taxes on **realized gains** only at rebalance boundaries; fixed‑point solver for after‑tax reweighting; CSV ledger.
- **Leveraged ETF simulation** (2x/3x): daily model with dividend carry, risk‑free borrow, fixed annual cost; parameter search and actual‑vs‑simulated comparison.
- **GPU acceleration** (optional): CuPy-backed rolling metrics; automatic engine selection (`auto → cupy or numpy`).
- **Reproducible CLI**: `analyze`, `gen-spx-tr`, `gen-leveraged-etf`, `compare`, `optimize`.
- **Rich visualization**: distributions/boxplots, win‑rate matrices, DCA stacked value paths, and tax summaries exported to `output/`.

---

## Repository Layout

```
rollpath-lab/
├─ main.py                   # CLI entrypoint: analyze / compare / gen-*/ optimize           ← run this
├─ src/
│  ├─ settings.py            # global config: paths, periods, portfolios, tax, perf
│  ├─ app/                   # orchestration (CLI, run loop, cash, paths, tax print)
│  ├─ commands/              # subcommands (generate SPX TR, leveraged ETFs, compare, optimize)
│  ├─ data_loader.py         # strict loader for SERIES_SPECS (patterns & CSV robustness)
│  ├─ metrics*.py            # rolling metrics (NumPy / CuPy acceleration)
│  ├─ visualizer.py          # charts & CSV exports
│  └─ ...
├─ data/                     # all inputs/outputs live here
│  ├─ raw/                   # raw monthly risk‑free & dividend CSVs, plus source index
│  │  └─ source.csv          # mapping: filename → canonical URL (see below)
│  ├─ ^spx_d_*.csv           # daily SPX price (downloaded)
│  ├─ ^SPX_*_daily_TR.csv    # generated SPX Total Return (output of gen-spx-tr)
│  ├─ ^sso_simulated_*.csv   # generated 2x series (output of gen-leveraged-etf)
│  └─ ^spxl_simulated_*.csv  # generated 3x series (output of gen-leveraged-etf)
└─ output/                   # time-stamped export root with charts & tables
```

Most defaults (data patterns, portfolios, tax rate, rolling periods, DCA interval, performance knobs) are centralized in `src/settings.py` and respected by the CLI.

---

## Quick Start

### 1) Environment

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate
# macOS/Linux:
source .venv/bin/activate

pip install -U pip
pip install numpy pandas polars rich seaborn matplotlib yfinance psutil
# Optional (GPU acceleration):
pip install cupy-cuda12x  # choose the build matching your CUDA
```

> CuPy is optional. When `--engine auto`, the runtime picks `cupy` if available, otherwise `numpy`.

### 2) Data Directory

Create the folders:

```bash
mkdir -p data/raw
```

Put raw monthly macro series and your source index CSVs under `data/` and `data/raw/` as described below. The loader **strictly follows** filenames/patterns in settings (no guessing).

---

## `data/raw/source.csv` — What to Download & Where

Place a CSV named `data/raw/source.csv` listing canonical sources (filename, URL). For convenience, here is a ready‑to‑use template that matches the project’s expected filenames and patterns:

```csv
<source.csv>,<url>
^spx_d.csv,https://stooq.com/q/d/?s=%5Espx&c=0
Yields on Short-Term United States Securities, Three-Six Month Treasury Notes and Certificates, Three Month Treasury Bills for United States 1920-1934.csv,https://fred.stlouisfed.org/series/M1329AUSM193NNBR
data.csv,https://datahub.io/core/s-and-p-500
3-Month Treasury Bill Secondary Market Rate, Discount Basis (TB3MS) 1934-2026.csv,https://fred.stlouisfed.org/series/TB3MS
spxl_us_d.csv,https://stooq.com/q/d/?s=spxl.us&i=d
SPXL_2008-11-06_2026-02-28_daily.csv,https://finance.yahoo.com/quote/SPXL/history/?period1=1225929600&period2=1772259606
S&P 500 Dividend Yield by Month,https://www.multpl.com/s-p-500-dividend-yield/table/by-month
```

### Download & Save

- **Daily SPX price**  
  Download `^spx_d.csv` to `data/`. If your file does not include the `_to_YYYYMMDD` token, pass it explicitly via `--spx` when generating TR (see below).

- **Monthly risk‑free (two files stitched)**  
  Save both CSVs to `data/raw/`:
  1) `Yields on Short-Term United States Securities, Three-Six Month Treasury Notes and Certificates, Three Month Treasury Bills for United States 1920-1934.csv`
  2) `3-Month Treasury Bill Secondary Market Rate, Discount Basis (TB3MS) 1934-2026.csv`  
  These are stitched into a single monthly risk‑free rate inside the project.

- **Monthly S&P 500 dividend yield**  
  Save a file named like `S&P 500 Dividend Yield by Month_*.csv` to `data/raw/`. The parser handles mixed legacy date formats and century boundaries.

- **Actual SPXL/SSO daily** (for comparison/optimization)  
  - `SPXL_2008-11-06_2026-02-28_daily.csv` → `data/`  
  - For SSO, a similar `SSO_*_daily.csv` in `data/`.  
  The comparator auto-detects actual vs simulated by filename patterns.

> Tip: You can also fetch via `src/get_data.py` (yfinance) to create daily actual series quickly.

---

## Generating Inputs

### A) SPX Total Return (daily)

From your downloaded `data/^spx_d*.csv` and `data/raw/S&P 500 Dividend Yield by Month_*.csv`, build the daily **Total Return** series (dividend‑reinvested). This normalizes mixed date formats and rescales the final TR price to match the raw SPX price on the last day for continuity.

```bash
# Option 1: explicit input if your file is simply '^spx_d.csv'
python -m src.commands.generate_spx_total_return \
  --spx "data/^spx_d.csv" \
  --dividend "data/raw/S&P 500 Dividend Yield by Month_*.csv" \
  --outdir data

# Option 2: use the default wildcard pattern (requires files named with '_to_YYYYMMDD')
python -m src.commands.generate_spx_total_return \
  --spx "data/^spx_d_*_to_*.csv" \
  --dividend "data/raw/S&P 500 Dividend Yield by Month_*.csv" \
  --outdir data
```

**Output**: `data/^SPX_<start>_<end>_daily_TR.csv` (used downstream).

### B) Simulated Leveraged ETFs (2x/3x)

Build 2x (SSO‑like) and 3x (SPXL‑like) daily series using TR carry, risk‑free borrow, and fixed annual cost. You can tune alpha/beta/cost or rely on defaults from `settings.py`.

```bash
# via main subcommand
python main.py gen-leveraged-etf

# or directly with explicit inputs & window
python -m src.commands.generate_leveraged_etf \
  --spx-csv "data/^SPX_*_daily_TR.csv" \
  --dividend-csv "data/raw/S&P 500 Dividend Yield by Month_*.csv" \
  --tbill-1920-1934-csv "data/raw/Yields on Short-Term United States Securities, Three-Six Month Treasury Notes and Certificates, Three Month Treasury Bills for United States 1920-1934.csv" \
  --tb3ms-1934-now-csv "data/raw/3-Month Treasury Bill Secondary Market Rate, Discount Basis (TB3MS) 1934-2026.csv" \
  --start 1928-01-01 --end 2026-02-28 \
  --outdir data
```

**Outputs**:  
- `data/^sso_simulated_d_TR_<cost>_<aX_bY%>_<start>_<end>.csv`  
- `data/^spxl_simulated_d_TR_<cost>_<aX_bY%>_<start>_<end>.csv`

---

## Rolling Analysis (DCA / Lump‑Sum)

### Default configuration

- Rolling windows: 10, 20, …, 70 **years** (period unit & frequency can be changed).
- Style: **DCA** by default (`INITIAL_CAPITAL`, `DCA_AMOUNT`, `DCA_INTERVAL='monthly'`).
- Portfolios: S&P 500 TR, simulated 2x/3x (see `SERIES_SPECS`), configurable in `settings.py`.
- Tax model: annual rate (e.g., 20.315%) applied **only** on realized gains at rebalance points.

Run:

```bash
# use defaults in src/settings.py
python main.py analyze

# common overrides
python main.py analyze \
  --style dca --dca-interval monthly --amount 100000 --initial 0 \
  --vals 10 20 30 --unit years \
  --rebalance yearly --tax-rate 0.20315 \
  --engine auto --workers 8 --batch-size 256 --plots
```

**Outputs** land under a time-stamped `output/<ports>__<date>_.../` with:  
- return distributions & boxplots;  
- win‑rate matrix;  
- **DCA value tables** (Final Value stats, simple CAGR, median IRR);  
- typical VALUE paths (mean/median) + CSV; representative stacked value paths + CSV;  
- tax summary & time series (when applicable).

---

## Actual vs Simulated: Divergence & Charts

Compare normalized paths and divergence (%) between actual ETF and your simulated series:

```bash
# via main subcommand
python main.py compare

# or directly
python -m src.commands.compare_actual_vs_simulated --symbols SPXL SSO --outdir output/compare
```

It produces divergence plots (MAE, final divergence) and normalized overlays, with CJK‑safe font selection on most systems.

---

## Parameter Optimization (alpha, beta, cost)

Search (golden‑section per coordinate) to minimize divergence (MAE/Huber/trimmed) against actual ETF history:

```bash
# quick: use defaults (invoked from main.py)
python main.py optimize

# detailed control
python -m src.commands.optimize_cost_by_mae \
  --symbols SPXL SSO \
  --loss mae --alpha-lo 1.0 --alpha-hi 1.5 --beta-lo 0.0 --beta-hi 0.03 \
  --cost-lo 0.0087 --cost-hi 0.05 \
  --start 2008-11-06 --end 2026-02-28 \
  --include-dividends \
  --save-best-sim
```

This writes the best simulated series back to `data/` with parameter tags embedded in filenames.

---

## Configuration Highlights

All defaults live in `src/settings.py`: data patterns (strict resolver), portfolio maps, periods/units/frequency, DCA amounts & interval, tax rate, and performance knobs (engine/workers/max RAM batch) used by the CLI.

- **Strict series resolution**: exact filename or wildcard patterns; newest match by mtime; robust date/price column handling (headerless 2‑col CSV supported).
- **Cash series (if portfolio includes `CASH`)**: flat / fixed annual / risk‑free modes.
- **Performance**: engine chooser (`auto|numpy|cupy`), BLAS thread pinning, batch/sample heuristics based on RAM; per‑window and overall metrics printed at the end of runs.

---

## Tax Model (Summary)

- Average‑cost basis within assets; tax only on realized gains from rebalancing sells.
- Solve for after‑tax portfolio value self‑consistently so weights match targets post‑tax.
- No tax if `rebalance=none`. Exports a debug ledger and roll‑up summaries.

---

## Troubleshooting

- **No files matched pattern**: check `data/` and `data/raw/` filenames against `SERIES_SPECS` and the command defaults—resolution is strict by design.
- **Missing glyphs in plots**: the visualizer selects CJK‑capable fonts; on headless servers you may need to install a CJK font (e.g., Noto Sans CJK).
- **CUDA not used**: `--engine auto` falls back to NumPy if no visible GPU/CuPy build; install the correct `cupy-cuda*` wheel and set `CUDA_VISIBLE_DEVICES`.

---

## License

MIT, contact at lapi@foxmail.com

---

## Acknowledgments

Data sources are user‑provided; ensure you review and comply with each provider’s terms of use.
