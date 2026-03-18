# src/main.py
"""
Project command dispatcher.

This entry point lets you run:
- leveraged ETF simulation generation
- leveraged ETF parameter optimization
- actual-vs-simulated comparison

from one place.

Examples
--------
Generate TQQQ and QLD simulated series:
    python -m src.main generate-qqq-family

Optimize TQQQ and QLD parameters:
    python -m src.main optimize-qqq-family

Compare actual vs simulated for TQQQ and QLD:
    python -m src.main compare-qqq-family

Generate any supported leveraged symbols:
    python -m src.main generate-leveraged --symbols TQQQ QLD SPXL SSO

Optimize any supported leveraged symbols:
    python -m src.main optimize-leveraged --symbols TQQQ QLD --save-best-sim

Compare any supported leveraged symbols:
    python -m src.main compare-leveraged --symbols TQQQ QLD SPXL SSO

Advanced passthrough usage:
    python -m src.main generate-leveraged-raw -- --symbols TQQQ QLD --start 1999-01-01
    python -m src.main optimize-leveraged-raw -- --symbols TQQQ QLD --loss huber
    python -m src.main compare-leveraged-raw -- --symbols TQQQ QLD --outdir output/compare
"""

from __future__ import annotations

import argparse
import sys
from typing import Optional

from src import settings
from src.commands.compare_actual_vs_simulated import main as compare_actual_vs_simulated_main
from src.commands.generate_leveraged_etf import main as generate_leveraged_main
from src.commands.optimize_cost_by_mae import main as optimize_leveraged_main


DEFAULT_QQQ_FAMILY = ["TQQQ", "QLD"]


def build_parser() -> argparse.ArgumentParser:
    """
    Build the root CLI parser.
    """
    parser = argparse.ArgumentParser(
        description="Project command dispatcher.",
    )
    subparsers = parser.add_subparsers(dest="command")

    # -----------------------------------------------------------------------
    # Generate
    # -----------------------------------------------------------------------
    p_generate = subparsers.add_parser(
        "generate-leveraged",
        help="Generate simulated leveraged ETF series using high-level options.",
    )
    p_generate.add_argument(
        "--symbols",
        nargs="+",
        default=DEFAULT_QQQ_FAMILY,
        choices=sorted(settings.SIMULATION_SPECS.keys()),
        help="Symbols to generate.",
    )
    p_generate.add_argument("--start", type=str, default=None)
    p_generate.add_argument("--end", type=str, default=None)
    p_generate.add_argument("--outdir", type=str, default=str(settings.DATA_DIR))
    p_generate.add_argument("--include-dividends", action="store_true")
    p_generate.add_argument("--no-dividends", action="store_true")

    p_generate_raw = subparsers.add_parser(
        "generate-leveraged-raw",
        help="Pass raw arguments directly to src.commands.generate_leveraged_etf.",
    )
    p_generate_raw.add_argument(
        "args",
        nargs=argparse.REMAINDER,
        help="Arguments forwarded to generate_leveraged_etf.py",
    )

    subparsers.add_parser(
        "generate-qqq-family",
        help="Generate TQQQ and QLD simulated series with default settings.",
    )

    # -----------------------------------------------------------------------
    # Optimize
    # -----------------------------------------------------------------------
    p_optimize = subparsers.add_parser(
        "optimize-leveraged",
        help="Optimize leveraged ETF parameters using high-level options.",
    )
    p_optimize.add_argument(
        "--symbols",
        nargs="+",
        default=DEFAULT_QQQ_FAMILY,
        choices=sorted(settings.OPTIMIZATION_SPECS.keys()),
        help="Symbols to optimize.",
    )
    p_optimize.add_argument(
        "--loss",
        type=str,
        default=settings.OPTIMIZATION_DEFAULT_LOSS,
        choices=["mae", "huber", "trimmed"],
    )
    p_optimize.add_argument(
        "--huber-delta",
        type=float,
        default=settings.OPTIMIZATION_DEFAULT_HUBER_DELTA,
        dest="huber_delta",
    )
    p_optimize.add_argument(
        "--trim",
        type=float,
        default=settings.OPTIMIZATION_DEFAULT_TRIM,
    )
    p_optimize.add_argument("--start", type=str, default=None)
    p_optimize.add_argument("--end", type=str, default=None)
    p_optimize.add_argument("--include-dividends", action="store_true")
    p_optimize.add_argument("--no-dividends", action="store_true")
    p_optimize.add_argument(
        "--save-best-sim",
        action="store_true",
        default=settings.OPTIMIZATION_SAVE_BEST_SIM,
    )

    p_optimize_raw = subparsers.add_parser(
        "optimize-leveraged-raw",
        help="Pass raw arguments directly to src.commands.optimize_cost_by_mae.",
    )
    p_optimize_raw.add_argument(
        "args",
        nargs=argparse.REMAINDER,
        help="Arguments forwarded to optimize_cost_by_mae.py",
    )

    subparsers.add_parser(
        "optimize-qqq-family",
        help="Optimize TQQQ and QLD parameters with default settings.",
    )

    # -----------------------------------------------------------------------
    # Compare
    # -----------------------------------------------------------------------
    p_compare = subparsers.add_parser(
        "compare-leveraged",
        help="Compare actual ETF series against simulated ETF series.",
    )
    p_compare.add_argument(
        "--symbols",
        nargs="+",
        default=DEFAULT_QQQ_FAMILY,
        choices=sorted(settings.SIMULATION_SPECS.keys()),
        help="Symbols to compare.",
    )
    p_compare.add_argument("--start", type=str, default=None)
    p_compare.add_argument("--end", type=str, default=None)
    p_compare.add_argument(
        "--outdir",
        type=str,
        default=str((settings.OUTPUT_DIR / "compare").resolve()),
        help="Directory where comparison charts will be saved.",
    )
    p_compare_raw = subparsers.add_parser(
        "compare-leveraged-raw",
        help="Pass raw arguments directly to src.commands.compare_actual_vs_simulated.",
    )
    p_compare_raw.add_argument(
        "args",
        nargs=argparse.REMAINDER,
        help="Arguments forwarded to compare_actual_vs_simulated.py",
    )
    subparsers.add_parser(
        "compare-qqq-family",
        help="Compare actual vs simulated for TQQQ and QLD using default settings.",
    )

    return parser


def build_generate_args(ns: argparse.Namespace) -> list[str]:
    """
    Convert parsed high-level generate options into forwarded CLI args.
    """
    argv: list[str] = ["--symbols", *ns.symbols]
    if ns.start:
        argv.extend(["--start", ns.start])
    if ns.end:
        argv.extend(["--end", ns.end])
    if ns.outdir:
        argv.extend(["--outdir", ns.outdir])
    if ns.include_dividends:
        argv.append("--include-dividends")
    if ns.no_dividends:
        argv.append("--no-dividends")
    return argv


def build_optimize_args(ns: argparse.Namespace) -> list[str]:
    """
    Convert parsed high-level optimize options into forwarded CLI args.
    """
    argv: list[str] = ["--symbols", *ns.symbols, "--loss", ns.loss]
    argv.extend(["--huber-delta", str(ns.huber_delta)])
    argv.extend(["--trim", str(ns.trim)])
    if ns.start:
        argv.extend(["--start", ns.start])
    if ns.end:
        argv.extend(["--end", ns.end])
    if ns.include_dividends:
        argv.append("--include-dividends")
    if ns.no_dividends:
        argv.append("--no-dividends")
    if ns.save_best_sim:
        argv.append("--save-best-sim")
    return argv


def build_compare_args(ns: argparse.Namespace) -> list[str]:
    """
    Convert parsed high-level compare options into forwarded CLI args.
    """
    argv: list[str] = ["--symbols", *ns.symbols]
    if ns.start:
        argv.extend(["--start", ns.start])
    if ns.end:
        argv.extend(["--end", ns.end])
    if ns.outdir:
        argv.extend(["--outdir", ns.outdir])
    return argv


def normalize_remainder_args(args: list[str]) -> list[str]:
    """
    Remove a leading '--' separator from argparse.REMAINDER if present.
    """
    if args and args[0] == "--":
        return args[1:]
    return args


def main(argv: Optional[list[str]] = None) -> int:
    """
    Dispatch to the selected workflow.
    """
    parser = build_parser()
    ns = parser.parse_args(argv)

    if ns.command == "generate-qqq-family":
        return generate_leveraged_main(["--symbols", *DEFAULT_QQQ_FAMILY])

    if ns.command == "optimize-qqq-family":
        return optimize_leveraged_main([
            "--symbols",
            *DEFAULT_QQQ_FAMILY,
            "--save-best-sim",
        ])

    if ns.command == "compare-qqq-family":
        return compare_actual_vs_simulated_main([
            "--symbols",
            *DEFAULT_QQQ_FAMILY,
        ])

    if ns.command == "generate-leveraged":
        return generate_leveraged_main(build_generate_args(ns))

    if ns.command == "optimize-leveraged":
        return optimize_leveraged_main(build_optimize_args(ns))

    if ns.command == "compare-leveraged":
        return compare_actual_vs_simulated_main(build_compare_args(ns))

    if ns.command == "generate-leveraged-raw":
        return generate_leveraged_main(normalize_remainder_args(ns.args))

    if ns.command == "optimize-leveraged-raw":
        return optimize_leveraged_main(normalize_remainder_args(ns.args))

    if ns.command == "compare-leveraged-raw":
        return compare_actual_vs_simulated_main(normalize_remainder_args(ns.args))

    parser.print_help()
    return 1


if __name__ == "__main__":
    sys.exit(main())