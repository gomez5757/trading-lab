from __future__ import annotations

import argparse
import json
import sys

ROOT = __import__("pathlib").Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from trading_lab.public_data import download_public_data, download_public_feature_data


def main() -> int:
    parser = argparse.ArgumentParser(description="Download public SPY daily data.")
    parser.add_argument("--output", default="data/public/spy_daily.csv")
    parser.add_argument("--feature-panel", action="store_true", help="Download public multi-asset feature panel.")
    args = parser.parse_args()

    if args.feature_panel:
        path = download_public_feature_data(args.output)
    else:
        path = download_public_data(args.output)
    print(json.dumps({"output": str(path)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
