from __future__ import annotations
import argparse
from src import settings

def _get(name: str, default):
    return getattr(settings, name, default)

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Portfolio Analyzer — performance-enabled")
    # original analysis parameters (kept)
    parser.add_argument("--freq", type=str, default=settings.DATA_FREQ)
    parser.add_argument("--rebalance", type=str, default=settings.REBALANCE_FREQ)
    parser.add_argument("--missing", type=str, default=settings.MISSING_DATA_STRATEGY)
    parser.add_argument("--vals", type=int, nargs="+", default=settings.INVESTMENT_PERIOD_VALUES)
    parser.add_argument("--unit", type=str, default=settings.INVESTMENT_PERIOD_UNIT)
    parser.add_argument("--style", type=str, default=settings.INVESTMENT_STYLE, choices=["lump_sum", "dca"]) 
    parser.add_argument("--initial", type=float, default=settings.INITIAL_CAPITAL)
    parser.add_argument("--amount", type=float, default=settings.DCA_AMOUNT)
    parser.add_argument("--dca-interval", type=str, default=getattr(settings, "DCA_INTERVAL", "every_period"), choices=["every_period", "weekly", "monthly", "quarterly", "yearly"]) 
    parser.add_argument("--tax-rate", type=float, default=getattr(settings, "TAX_RATE", 0.0))
    parser.add_argument("--cash-mode", type=str, default=getattr(settings, "CASH_MODE", "flat"), choices=["flat", "fixed", "rf"]) 
    parser.add_argument("--cash-fixed-rate", type=float, default=getattr(settings, "CASH_FIXED_RATE", 0.0))
    parser.add_argument("--debug-tax-ledger", action="store_true", default=True, help="Save taxable rebalancing sells to output_dir/debug_tax_ledger.csv")

    # performance flags: defaults pull from settings with safe fallback
    parser.add_argument("--engine", type=str, default=_get('PERF_ENGINE', 'auto'), choices=["auto", "numpy", "cupy", "numba"], help="Math backend for rolling metrics (auto -> cupy if available else numpy)")
    parser.add_argument("--workers", type=int, default=_get('PERF_WORKERS', 8), help="Max worker processes for parallel rolling metrics (default from settings or 8)")
    parser.add_argument("--max-ram-gb", type=float, default=_get('PERF_MAX_RAM_GB', 0.0), help="Soft RAM cap (GB) to adapt batch size and sampling caps; 0=disabled")
    parser.add_argument("--batch-size", type=int, default=_get('PERF_BATCH_SIZE', 256), help="Batch size for streamed path-extrema computation")
    # paired flags to allow overriding settings from CLI both ways
    parser.add_argument("--no-plots", dest="no_plots", action="store_true", help="Skip chart rendering to maximize throughput")
    parser.add_argument("--plots", dest="no_plots", action="store_false", help="Force plotting even if disabled in settings")
    parser.set_defaults(no_plots=bool(_get('PERF_NO_PLOTS', False)))
    return parser
