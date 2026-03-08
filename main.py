from __future__ import annotations

from dotenv import load_dotenv

"""Main application entry point.

This module provides a single, user-facing command-line entry point that
dispatches to subcommands or runs the default *analysis* workflow. It wraps
the CLI argument builder and the analysis runner, and exposes a small
command router so that auxiliary utilities can live in separate modules.

Subcommands
-----------
- ``analyze`` (default): Run the main Portfolio Analyzer.
- ``compare``: Compare actual vs. simulated plots for SSO/SPXL.
- ``gen-leveraged-etf``: Generate simulated SSO/SPXL price/return series.
- ``gen-spx-tr``: Generate daily S&P 500 Total Return series.
- ``optimize``: Jointly optimize (alpha, beta, cost) by minimizing divergence.

Performance Flags (passed through to ``analyze``)
-------------------------------------------------
- ``--engine {auto,numpy,cupy,numba}``
- ``--workers INT`` (default comes from settings or 8)
- ``--max-ram-gb FLOAT`` (soft cap; 0 = disabled)
- ``--batch-size INT`` (default from settings or 256)
- ``--no-plots`` / ``--plots`` (override plotting settings)

Notes
-----
This module intentionally avoids importing the heavier analysis code at
import-time. Subcommands are resolved via :mod:`importlib` only when needed,
which keeps startup time small for help/usage requests.
"""

import sys
from importlib import import_module

from src.app.cli import build_parser
from src.app.run import run_analysis

load_dotenv()  # Load environment variables from .env file if present

_ROOT_HELP = """
Usage: python main.py [subcommand] [options]

Subcommands:
  analyze            Run the main Portfolio Analyzer (default)
  compare            Actual vs Simulated plots for SSO/SPXL
  gen-leveraged-etf  Generate simulated SSO/SPXL series
  gen-spx-tr         Generate daily S&P 500 Total Return series
  optimize           Optimize (alpha, beta, cost) by MAE

Performance flags (analyze):
  --engine {auto,numpy,cupy,numba}
  --workers INT
  --max-ram-gb FLOAT (soft cap; 0 = disabled)
  --batch-size INT
  --no-plots / --plots
"""


def _dispatch_commands(argv: list[str]) -> int | None:
    """Dispatch a subcommand based on the first CLI token.

    If the first token is a recognized subcommand, this function loads the
    appropriate module and calls its entry function. If the token represents
    a help request, usage text is printed. If no token is provided, ``None``
    is returned so that the caller can fall back to the default behavior.

    Parameters
    ----------
    argv:
        The list of CLI arguments *after* the executable and script name.

    Returns
    -------
    int | None
        An integer process exit code if a subcommand ran, otherwise
        ``None`` when the default path (main analysis) should be executed.
    """
    if not argv:
        return None

    sub = argv[0].lower()
    cmd_map: dict[str, tuple[str | None, str | None, str]] = {
        "compare": ("src.commands.compare_actual_vs_simulated", "main", "noargv"),
        "gen-leveraged-etf": ("src.commands.generate_leveraged_etf", "main", "argv"),
        "gen-spx-tr": ("src.commands.generate_spx_total_return", "main", "argv"),
        "optimize": ("src.commands.optimize_cost_by_mae", "main", "noargv"),
        "analyze": (None, None, "analyze"),
        "help": (None, None, "help"),
        "--help": (None, None, "help"),
        "-h": (None, None, "help"),
    }

    if sub in ("-h", "--help", "help"):
        print(_ROOT_HELP)
        return 0

    if sub not in cmd_map:
        return None

    mod, func, mode = cmd_map[sub]

    if mode == "analyze":
        parser = build_parser()
        args = parser.parse_args(argv[1:])
        run_analysis(args)
        return 0

    if mod is None or func is None:
        return None

    module = import_module(mod)  # type: ignore[arg-type]
    entry = getattr(module, func)
    if mode == "argv":
        return int(entry(argv[1:]))
    else:
        result = entry()
        return int(result) if result is not None else 0


def main() -> None:
    """Program entry function used by ``python -m`` and direct execution.

    The function tries to dispatch a subcommand first. If no subcommand
    was provided or it is not recognized, the default *analyze* flow is
    executed using the standard CLI parser.

    Raises
    ------
    SystemExit
        When a dispatched subcommand returns a numeric exit code.
    """
    
    code = _dispatch_commands(sys.argv[1:])
    if code is not None:
        if isinstance(code, int):
            raise SystemExit(code)
        return

    parser = build_parser()
    args = parser.parse_args()
    run_analysis(args)


if __name__ == "__main__":
    main()
