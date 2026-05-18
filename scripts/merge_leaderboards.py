from __future__ import annotations

import argparse
import glob
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from trading_lab.results import merge_leaderboards, save_leaderboard


def main() -> int:
    parser = argparse.ArgumentParser(description="Merge optimization stage leaderboards.")
    parser.add_argument("--input-glob", default="outputs/optimization/stage_*.csv")
    parser.add_argument("--output", default="outputs/optimization/leaderboard.csv")
    parser.add_argument("--summary", default="outputs/optimization/summary.json")
    args = parser.parse_args()

    paths = sorted(glob.glob(args.input_glob))
    leaderboard = merge_leaderboards(paths)
    output_path = save_leaderboard(leaderboard, args.output)
    summary = {
        "input_files": paths,
        "rows": len(leaderboard),
        "output": str(output_path),
        "best": {} if leaderboard.empty else leaderboard.iloc[0].to_dict(),
    }
    summary_path = Path(args.summary)
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(summary, indent=2, default=str), encoding="utf-8")

    print(json.dumps(summary, indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
