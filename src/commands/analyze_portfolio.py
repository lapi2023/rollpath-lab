# src/commands/analyze_portfolio.py
from __future__ import annotations

"""
Run the portfolio rolling analysis (DCA or Lump Sum) using the app layer.

- CLI options are defined by src/app/cli.py (defaults come from src/settings.py).
- The actual workflow is implemented in src/app/run.py::run_analysis().
- Console prints summary tables; charts/tables are saved under settings.OUTPUT_DIR.

This command replicates the "no-command" behavior that previously lived in main.py,
but makes it available as an explicit subcommand.
"""

import sys
from typing import Optional, List

from src.app.cli import build_parser
from src.app.run import run_analysis


def main(argv: Optional[List[str]] = None) -> int:
    """
    CLI entry point.

    Parameters
    ----------
    argv : list[str] | None
        Command-line arguments (excluding program name). If None, argparse
        will parse from sys.argv.
    """
    parser = build_parser()
    args = parser.parse_args(argv)
    run_analysis(args)
    return 0


if __name__ == "__main__":
    sys.exit(main())