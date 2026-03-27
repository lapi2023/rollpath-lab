# main.py
"""
Project command dispatcher with settings-backed defaults.

This entry point lets you run every feature under src/commands from one place.
If required arguments are omitted, sensible defaults are injected from
src/settings.py (e.g., START_DATE/END_DATE/OUTPUT_DIR/DATA_DIR).

New in this revision
--------------------
- `analyze-portfolio` subcommand: runs the rolling DCA/Lump Sum analysis
  (console tables + charts/tables under OUTPUT_DIR) via src/app/run.py.
- Backward-friendly behavior for `analyze-price-csv`:
  when `--csv` is NOT provided, it falls back to `analyze-portfolio`.
- No-command default: running `python main.py` executes `analyze-portfolio`.

CLI precedence over settings
----------------------------
We now forward the user's CLI flags to `src/commands/analyze_portfolio`
so that CLI values override the defaults defined in src/app/cli.py/settings.py.
"""

from __future__ import annotations

import argparse
import fnmatch
import importlib
import pkgutil
import re
import sys
from pathlib import Path
from typing import Optional, Sequence

from src import settings

# High-level commands
from src.commands.compare_actual_vs_simulated import main as compare_actual_vs_simulated_main
from src.commands.generate_leveraged_etf import main as generate_leveraged_main
from src.commands.optimize_cost_by_mae import main as optimize_leveraged_main

# Other commands
from src.commands.analyze_price_csv import main as analyze_price_csv_main
from src.commands.generate_spx_total_return import main as generate_spx_total_return_main

# Portfolio analysis (DCA / Lump Sum)
from src.commands.analyze_portfolio import main as analyze_portfolio_main  # NEW

DEFAULT_QQQ_FAMILY = ["TQQQ", "QLD"]

# ------------------------------- utils --------------------------------------

def _get(name: str, default):
    return getattr(settings, name, default)

def has_flag(argv: Sequence[str], flag: str) -> bool:
    if flag in argv:
        return True
    prefix = f"{flag}="
    return any(str(a).startswith(prefix) for a in argv)

def ensure_kv(argv: list[str], flag: str, value: str) -> None:
    if not has_flag(argv, flag):
        argv.extend([flag, value])

def discover_commands() -> list[str]:
    try:
        pkg = importlib.import_module("src.commands")
        names: list[str] = []
        for modinfo in pkgutil.iter_modules(pkg.__path__):
            n = modinfo.name
            if not n.startswith("_"):
                names.append(n)
        return sorted(names)
    except Exception:
        return []

def run_dynamic_command(module_name: str, argv: Sequence[str] | None = None) -> int:
    try:
        mod = importlib.import_module(f"src.commands.{module_name}")
    except ModuleNotFoundError:
        sys.stderr.write(f"[ERROR] Module 'src.commands.{module_name}' not found.\n")
        return 1
    except Exception as e:
        sys.stderr.write(f"[ERROR] Failed to import 'src.commands.{module_name}': {e}\n")
        return 1
    if not hasattr(mod, "main"):
        sys.stderr.write(f"[ERROR] Module 'src.commands.{module_name}' has no 'main' function.\n")
        return 1
    try:
        return int(mod.main(list(argv or [])))  # type: ignore[attr-defined]
    except SystemExit as ex:
        return int(ex.code) if isinstance(ex.code, int) else 1
    except Exception as e:
        sys.stderr.write(f"[ERROR] Exception while running '{module_name}': {e}\n")
        return 1

# ----- file resolution for analyze (from settings) ---------------------------

def _iter_csv_files(root: Path) -> list[Path]:
    return [p for p in root.rglob("*.csv") if p.is_file()]

def _exclude_names(paths: list[Path]) -> list[Path]:
    excludes = tuple(_get("EXCLUDE_FILENAME_SUBSTRINGS", ()))
    if not excludes:
        return paths
    kept: list[Path] = []
    for p in paths:
        if any(sub.lower() in p.name.lower() for sub in excludes):
            continue
        kept.append(p)
    return kept

def _match_series_spec_first(paths: list[Path]) -> list[Path]:
    specs = _get("SERIES_SPECS", {})
    pat = specs.get("SP500_TR") if isinstance(specs, dict) else None
    if not pat:
        return paths

    def is_regex(s: str) -> bool:
        return ("^" in s) or ("$" in s)

    matched, rest = [], []
    if is_regex(pat):
        rx = re.compile(pat)
        for p in paths:
            (matched if rx.search(p.name) else rest).append(p)
    else:
        for p in paths:
            (matched if fnmatch.fnmatch(p.name, pat) else rest).append(p)
    return matched + rest

# ------------------------------- CLI ----------------------------------------

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Project command dispatcher.")
    sub = parser.add_subparsers(dest="command")

    # Generate (high-level)
    p_gen = sub.add_parser("generate-leveraged", help="Generate leveraged ETF series (high-level).")
    p_gen.add_argument("--symbols", nargs="+", default=DEFAULT_QQQ_FAMILY,
                       choices=sorted(settings.SIMULATION_SPECS.keys()), help="Symbols to generate.")
    p_gen.add_argument("--start", type=str, default=None)
    p_gen.add_argument("--end", type=str, default=None)
    p_gen.add_argument("--outdir", type=str, default=str(settings.DATA_DIR))
    p_gen.add_argument("--include-dividends", action="store_true")
    p_gen.add_argument("--no-dividends", action="store_true")

    p_gen_raw = sub.add_parser("generate-leveraged-raw",
                               help="Forward to src.commands.generate_leveraged_etf (defaults injected).")
    p_gen_raw.add_argument("args", nargs=argparse.REMAINDER)

    sub.add_parser("generate-qqq-family", help="Generate TQQQ and QLD with defaults.")

    # Optimize (high-level)
    p_opt = sub.add_parser("optimize-leveraged", help="Optimize leveraged ETF parameters (high-level).")
    p_opt.add_argument("--symbols", nargs="+", default=DEFAULT_QQQ_FAMILY,
                       choices=sorted(settings.OPTIMIZATION_SPECS.keys()), help="Symbols to optimize.")
    p_opt.add_argument("--loss", type=str, default=settings.OPTIMIZATION_DEFAULT_LOSS,
                       choices=["mae", "huber", "trimmed"])
    p_opt.add_argument("--huber-delta", type=float, default=settings.OPTIMIZATION_DEFAULT_HUBER_DELTA,
                       dest="huber_delta")
    p_opt.add_argument("--trim", type=float, default=settings.OPTIMIZATION_DEFAULT_TRIM)
    p_opt.add_argument("--start", type=str, default=None)
    p_opt.add_argument("--end", type=str, default=None)
    p_opt.add_argument("--include-dividends", action="store_true")
    p_opt.add_argument("--no-dividends", action="store_true")
    p_opt.add_argument("--save-best-sim", action="store_true",
                       default=settings.OPTIMIZATION_SAVE_BEST_SIM)

    p_opt_raw = sub.add_parser("optimize-leveraged-raw",
                               help="Forward to src.commands.optimize_cost_by_mae (defaults injected).")
    p_opt_raw.add_argument("args", nargs=argparse.REMAINDER)

    sub.add_parser("optimize-qqq-family", help="Optimize TQQQ and QLD with defaults.")

    # Compare (high-level)
    p_cmp = sub.add_parser("compare-leveraged", help="Compare actual vs simulated ETF series.")
    p_cmp.add_argument("--symbols", nargs="+", default=DEFAULT_QQQ_FAMILY,
                       choices=sorted(settings.SIMULATION_SPECS.keys()), help="Symbols to compare.")
    p_cmp.add_argument("--start", type=str, default=None)
    p_cmp.add_argument("--end", type=str, default=None)
    p_cmp.add_argument("--outdir", type=str, default=str((settings.OUTPUT_DIR / "compare").resolve()))

    p_cmp_raw = sub.add_parser("compare-leveraged-raw",
                               help="Forward to src.commands.compare_actual_vs_simulated (defaults injected).")
    p_cmp_raw.add_argument("args", nargs=argparse.REMAINDER)

    sub.add_parser("compare-qqq-family", help="Compare TQQQ vs QLD with defaults.")

    # Analyze CSV (when --csv provided) OR fallback to portfolio analysis
    p_az = sub.add_parser("analyze-price-csv",
                          help="Analyze a (Date, Price) CSV (price charts + return histograms). "
                               "If --csv is omitted, falls back to portfolio analysis.")
    # CSV-analysis flags (forwarded to src.commands.analyze_price_csv)
    p_az.add_argument("--csv", type=str, default=None, help="Path to CSV (left=Date, right=Price).")
    p_az.add_argument("--outdir", type=str, default=None, help="Directory to save charts/tables.")
    p_az.add_argument("--title", type=str, default=None)
    p_az.add_argument("--bins", type=int, default=None)
    p_az.add_argument("--dpi", type=int, default=None)
    p_az.add_argument("--start", type=str, default=None)
    p_az.add_argument("--end", type=str, default=None)

    # Explicit portfolio analysis command (DCA / Lump Sum)
    sub.add_parser("analyze-portfolio",
                   help="Run the rolling portfolio analysis (DCA or Lump Sum) via src/app/run.py.")

    # SPX Total Return (raw; defaults injected)
    p_spx = sub.add_parser("generate-spx-total-return",
                           help="Forward to src.commands.generate_spx_total_return (defaults injected).")
    p_spx.add_argument("args", nargs=argparse.REMAINDER)

    # Helpers
    sub.add_parser("list-commands", help="List src.commands modules.")
    p_run = sub.add_parser("run-command", help="Run any module under src.commands dynamically.")
    p_run.add_argument("module")
    p_run.add_argument("args", nargs=argparse.REMAINDER)

    return parser

# ------------------------------ dispatcher ----------------------------------

def normalize_remainder_args(args: list[str]) -> list[str]:
    return args[1:] if args and args[0] == "--" else args

def inject_start_end_defaults(argv: list[str]) -> None:
    if not has_flag(argv, "--start") and _get("START_DATE", None):
        ensure_kv(argv, "--start", str(_get("START_DATE", "")))
    if not has_flag(argv, "--end") and _get("END_DATE", None):
        ensure_kv(argv, "--end", str(_get("END_DATE", "")))

def main(argv: Optional[list[str]] = None) -> int:
    parser = build_parser()
    ns, unknown = parser.parse_known_args(argv)

    # ===== Default behavior (no command): run the portfolio analysis =====
    if ns.command is None:
        # CHANGED: forward root-level flags to analyze_portfolio
        return analyze_portfolio_main(unknown)  # CLI > settings
        # =====================================================================

    # canned
    if ns.command == "generate-qqq-family":
        return generate_leveraged_main(["--symbols", *DEFAULT_QQQ_FAMILY])
    if ns.command == "optimize-qqq-family":
        return optimize_leveraged_main(["--symbols", *DEFAULT_QQQ_FAMILY, "--save-best-sim"])
    if ns.command == "compare-qqq-family":
        return compare_actual_vs_simulated_main(["--symbols", *DEFAULT_QQQ_FAMILY])

    # high-level
    if ns.command == "generate-leveraged":
        if unknown:
            parser.error(f"unrecognized arguments: {' '.join(unknown)}")
        argv2 = ["--symbols", *ns.symbols]
        ensure_kv(argv2, "--start", str(ns.start or _get("START_DATE", "")))
        ensure_kv(argv2, "--end", str(ns.end or _get("END_DATE", "")))
        ensure_kv(argv2, "--outdir", str(ns.outdir))
        if ns.include_dividends: argv2.append("--include-dividends")
        if ns.no_dividends: argv2.append("--no-dividends")
        return generate_leveraged_main(argv2)

    if ns.command == "optimize-leveraged":
        if unknown:
            parser.error(f"unrecognized arguments: {' '.join(unknown)}")
        argv2 = ["--symbols", *ns.symbols, "--loss", ns.loss,
                 "--huber-delta", str(ns.huber_delta), "--trim", str(ns.trim)]
        ensure_kv(argv2, "--start", str(ns.start or _get("START_DATE", "")))
        ensure_kv(argv2, "--end", str(ns.end or _get("END_DATE", "")))
        if ns.include_dividends: argv2.append("--include-dividends")
        if ns.no_dividends: argv2.append("--no-dividends")
        if ns.save_best_sim: argv2.append("--save-best-sim")
        return optimize_leveraged_main(argv2)

    if ns.command == "compare-leveraged":
        if unknown:
            parser.error(f"unrecognized arguments: {' '.join(unknown)}")
        argv2 = ["--symbols", *ns.symbols]
        ensure_kv(argv2, "--start", str(ns.start or _get("START_DATE", "")))
        ensure_kv(argv2, "--end", str(ns.end or _get("END_DATE", "")))
        ensure_kv(argv2, "--outdir", str(ns.outdir or (settings.OUTPUT_DIR / "compare").resolve()))
        return compare_actual_vs_simulated_main(argv2)

    # raw with defaults
    if ns.command == "generate-leveraged-raw":
        argv2 = normalize_remainder_args(ns.args)
        inject_start_end_defaults(argv2)
        return generate_leveraged_main(argv2)

    if ns.command == "optimize-leveraged-raw":
        argv2 = normalize_remainder_args(ns.args)
        inject_start_end_defaults(argv2)
        return optimize_leveraged_main(argv2)

    if ns.command == "compare-leveraged-raw":
        argv2 = normalize_remainder_args(ns.args)
        inject_start_end_defaults(argv2)
        if not has_flag(argv2, "--outdir"):
            ensure_kv(argv2, "--outdir", str((settings.OUTPUT_DIR / "compare").resolve()))
        return compare_actual_vs_simulated_main(argv2)

    # === Analyze CSV or Portfolio fallback ===
    if ns.command == "analyze-price-csv":
        # If --csv is provided (or resolvable), run CSV-analysis; else run portfolio analysis.
        if ns.csv:
            argv2: list[str] = []
            # start/end/outdir defaults from settings if omitted
            ensure_kv(argv2, "--start", str(ns.start or _get("START_DATE", "")))
            ensure_kv(argv2, "--end", str(ns.end or _get("END_DATE", "")))
            ensure_kv(argv2, "--outdir", str(ns.outdir or (settings.OUTPUT_DIR / "analyze")))
            ensure_kv(argv2, "--csv", str(ns.csv))
            if ns.title is not None: ensure_kv(argv2, "--title", str(ns.title))
            if ns.bins is not None: ensure_kv(argv2, "--bins", str(ns.bins))
            if ns.dpi is not None: ensure_kv(argv2, "--dpi", str(ns.dpi))
            return analyze_price_csv_main(argv2)
        else:
            # CHANGED: No --csv → treat as portfolio analysis AND forward user flags
            return analyze_portfolio_main(unknown)  # CLI > settings

    # === Explicit portfolio analysis ===
    if ns.command == "analyze-portfolio":
        # CHANGED: forward subcommand flags to analyzer (CLI > settings)
        return analyze_portfolio_main(unknown)  #
    if ns.command == "generate-spx-total-return":
        argv2 = normalize_remainder_args(ns.args)
        inject_start_end_defaults(argv2)
        if not has_flag(argv2, "--out"):
            ensure_kv(argv2, "--out", str(Path(settings.DATA_DIR) / "SPX_TR_generated.csv"))
        return generate_spx_total_return_main(argv2)

    if ns.command == "list-commands":
        print("\n".join(discover_commands()) or "(no commands discovered)")
        return 0

    if ns.command == "run-command":
        return run_dynamic_command(ns.module, normalize_remainder_args(ns.args))

    parser.print_help()
    return 1


if __name__ == "__main__":
    from pathlib import Path
    try:
        Path(settings.OUTPUT_DIR).mkdir(parents=True, exist_ok=True)
    except Exception:
        pass
    sys.exit(main())