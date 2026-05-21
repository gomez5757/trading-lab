from __future__ import annotations

import argparse
import glob
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from trading_lab.monthly_risk import merge_monthly_risk_leaderboards  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description="Merge monthly SP500 risk search stage leaderboards.")
    parser.add_argument("--input-glob", default="outputs/monthly_sp500_risk/monthly_risk_leaderboard_stage_*.csv")
    parser.add_argument("--output-dir", default="outputs/monthly_sp500_risk")
    args = parser.parse_args()

    paths = sorted(glob.glob(args.input_glob))
    summary = merge_monthly_risk_leaderboards(paths, args.output_dir)
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
