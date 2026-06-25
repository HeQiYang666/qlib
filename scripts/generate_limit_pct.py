#!/usr/bin/env python3
"""
Generate limit_pct.day.bin for each stock — constant per-stock daily price limit percentage.

- Main board (sh600/sz000/sz002 etc.): 0.10
- STAR / ChiNext (sh688/sz300): 0.20

Usage:
    python scripts/generate_limit_pct.py [--data-dir ~/.qlib/qlib_data/cn_data]
"""
import argparse
import os
import sys
from pathlib import Path

import numpy as np


def get_limit_pct(symbol: str) -> float:
    if symbol.startswith("sh68") or symbol.startswith("sh69"):
        return 0.20
    if symbol.startswith("sz3"):
        if len(symbol) >= 5 and symbol[3] in ("0", "1", "2"):
            return 0.20
    return 0.10


def main():
    parser = argparse.ArgumentParser(description="Generate per-stock limit_pct.day.bin files")
    parser.add_argument("--data-dir", default=os.path.expanduser("~/.qlib/qlib_data/cn_data"),
                        help="Qlib data directory")
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    feature_dir = data_dir / "features"
    inst_file = data_dir / "instruments" / "all.txt"

    print(f"Data directory: {data_dir}")
    print(f"Instruments file: {inst_file}")

    instruments = []
    with open(inst_file) as f:
        for line in f:
            parts = line.strip().split("\t")
            if parts:
                instruments.append(parts[0])

    print(f"Instruments: {len(instruments)}")

    written = 0
    skipped = 0
    for sym in instruments:
        sym_dir = feature_dir / sym.lower()
        close_path = sym_dir / "close.day.bin"
        if not close_path.exists():
            skipped += 1
            continue

        close_data = np.fromfile(str(close_path), dtype="<f")
        start_pos = close_data[0]
        n_values = len(close_data) - 1

        limit_pct = get_limit_pct(sym)
        limit_values = np.full(n_values, limit_pct, dtype=np.float32)
        np.hstack([np.float32(start_pos), limit_values]).astype("<f").tofile(
            str(sym_dir / "limit_pct.day.bin")
        )
        written += 1

        if written % 1000 == 0:
            print(f"  {written}/{len(instruments)}...")

    print(f"Done! {written} stocks written, {skipped} skipped")

    # Summarize
    main_count = sum(1 for s in instruments if get_limit_pct(s) == 0.10 and (feature_dir / s.lower() / "close.day.bin").exists())
    star_count = sum(1 for s in instruments if get_limit_pct(s) == 0.20 and (feature_dir / s.lower() / "close.day.bin").exists())
    print(f"  Main board (0.10): {main_count}")
    print(f"  STAR/ChiNext (0.20): {star_count}")


if __name__ == "__main__":
    main()
