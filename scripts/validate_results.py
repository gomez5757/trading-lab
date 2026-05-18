from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def main() -> int:
    parser = argparse.ArgumentParser(description="Validate generated result files.")
    parser.add_argument("paths", nargs="+")
    args = parser.parse_args()

    for raw_path in args.paths:
        path = Path(raw_path)
        if not path.exists():
            raise FileNotFoundError(f"Missing result file: {path}")
        if path.suffix == ".csv" and pd.read_csv(path).empty:
            raise ValueError(f"CSV result is empty: {path}")

    print(f"Validated {len(args.paths)} result file(s).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
