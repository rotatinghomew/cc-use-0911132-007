"""CLI entry: ``python3 -m market_space_twin.portfolio_risk``."""

from __future__ import annotations

import argparse
import json
import sqlite3
import sys

from market_space_twin.portfolio_risk.analysis import DEFAULT_PATHS, analyze
from market_space_twin.portfolio_risk.model import RiskInputError
from market_space_twin.simulation.twin_store import DomainError


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Compare tenant portfolio cash-flow risk across leasing plans"
    )
    parser.add_argument("--input", required=True, help="lease/cash-flow JSON input")
    parser.add_argument("--database", required=True, help="twin SQLite database")
    parser.add_argument("--output", required=True, help="risk JSON output path")
    parser.add_argument(
        "--paths",
        type=int,
        default=DEFAULT_PATHS,
        help=f"simulation paths per plan (default {DEFAULT_PATHS})",
    )
    args = parser.parse_args(argv)
    if args.paths <= 0:
        print("error: --paths must be positive", file=sys.stderr)
        return 2
    try:
        result = analyze(args.input, args.database, args.output, paths=args.paths)
    except (RiskInputError, DomainError, OSError, json.JSONDecodeError, sqlite3.Error) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    print(
        json.dumps(
            {
                "risk_run_id": result["risk_run_id"],
                "replayed": result.get("replayed", False),
                "plans": result.get("plans", []),
                "output": args.output,
            },
            ensure_ascii=False,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
