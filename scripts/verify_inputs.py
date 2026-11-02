#!/usr/bin/env python3
"""Check the local original-data and frozen-product inventory by file size."""

from pathlib import Path
import csv
import json


def main():
    root = Path(__file__).resolve().parents[1]
    report = {}
    for name in ["data", "product"]:
        with (root / f"config/{name}_manifest.csv").open() as stream:
            records = list(csv.DictReader(stream))
        for row in records:
            path = root / row["path"]
            if not path.is_file():
                raise FileNotFoundError(f"{path}: restore the input-data release before checking data.")
            assert path.stat().st_size == int(row["bytes"]), path
        report[name] = dict(files=len(records), bytes=sum(int(row["bytes"]) for row in records))
    destination = root / "build/data_inventory.json"
    destination.parent.mkdir(exist_ok=True)
    destination.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
