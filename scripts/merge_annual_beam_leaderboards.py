from __future__ import annotations

import argparse
import json
from glob import glob
from pathlib import Path

import pandas as pd


def main() -> int:
    parser = argparse.ArgumentParser(description="Merge annual SP500 Beam stage leaderboards.")
    parser.add_argument("--input-glob", default="outputs/annual_sp500_beam/annual_sp500_beam_stage_*.csv")
    parser.add_argument("--output", default="outputs/annual_sp500_beam/annual_sp500_beam_leaderboard.csv")
    parser.add_argument("--summary", default="outputs/annual_sp500_beam/annual_sp500_beam_summary.json")
    args = parser.parse_args()
    paths = sorted(Path(path) for path in glob(args.input_glob))
    if not paths:
        raise FileNotFoundError(f"no stage files matched {args.input_glob}")
    frames = [pd.read_csv(path) for path in paths if path.stat().st_size > 0]
    leaderboard = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    if not leaderboard.empty:
        leaderboard = leaderboard.sort_values("annual_score", ascending=False)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    leaderboard.to_csv(output, index=False)
    best = leaderboard.iloc[0].to_dict() if not leaderboard.empty else {}
    summary = {
        "rows": int(len(leaderboard)),
        "candidates_evaluated": int(len(leaderboard)),
        "accepted": int(leaderboard["accepted"].fillna(False).astype(bool).sum()) if "accepted" in leaderboard else 0,
        "best": _json_clean(best),
        "locked_opened": False,
    }
    Path(args.summary).write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    return 0


def _json_clean(value):
    if isinstance(value, dict):
        return {str(key): _json_clean(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_json_clean(item) for item in value]
    if pd.isna(value):
        return None
    if hasattr(value, "item"):
        return value.item()
    return value


if __name__ == "__main__":
    raise SystemExit(main())
