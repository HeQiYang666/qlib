#!/usr/bin/env python3
"""
Generate limit_buy.day.bin and limit_sell.day.bin for each stock.

limit_buy:  True (1.0) when daily return >= 95% of limit_pct (stock at/near limit-up, cannot buy)
limit_sell: True (1.0) when daily return <= -95% of limit_pct (stock at/near limit-down, cannot sell)

Usage:
    python scripts/generate_limit_buy_sell.py
"""
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd


def get_limit_pct(symbol: str) -> float:
    if symbol.startswith("sh68") or symbol.startswith("sh69"):
        return 0.20
    if symbol.startswith("sz3"):
        if len(symbol) >= 5 and symbol[3] in ("0", "1", "2"):
            return 0.20
    return 0.10


def main():
    data_dir = Path(os.path.expanduser("~/.qlib/qlib_data/cn_data"))
    feature_dir = data_dir / "features"
    inst_file = data_dir / "instruments" / "all.txt"

    instruments = []
    with open(inst_file) as f:
        for line in f:
            parts = line.strip().split("\t")
            if parts:
                instruments.append(parts[0])

    print(f"Total stocks: {len(instruments)}")

    written = 0
    for sym in instruments:
        sym_dir = feature_dir / sym.lower()
        close_path = sym_dir / "close.day.bin"

        if not close_path.exists():
            continue

        close_data = np.fromfile(str(close_path), dtype="<f")
        start_pos = close_data[0]
        close_arr = close_data[1:]

        limit_pct = get_limit_pct(sym)
        threshold = limit_pct * 0.95

        n = len(close_arr)
        if n < 2:
            continue

        # Forward-fill NaN so resume-day return is computed vs last valid close
        filled_close = pd.Series(close_arr).ffill().values
        daily_ret = np.zeros(n, dtype=np.float32)
        prev_close = filled_close[:-1]
        cur_close = close_arr[1:]
        with np.errstate(divide="ignore", invalid="ignore"):
            daily_ret[1:] = np.where(
                (prev_close > 0) & np.isfinite(cur_close),
                cur_close / prev_close - 1.0,
                0.0,
            )

        # limit_buy: return >= threshold (can't buy)
        limit_buy = (daily_ret >= threshold).astype(np.float32)
        np.hstack([np.float32(start_pos), limit_buy]).astype("<f").tofile(
            str(sym_dir / "limit_buy.day.bin")
        )

        # limit_sell: return <= -threshold (can't sell)
        limit_sell = (daily_ret <= -threshold).astype(np.float32)
        np.hstack([np.float32(start_pos), limit_sell]).astype("<f").tofile(
            str(sym_dir / "limit_sell.day.bin")
        )

        written += 1
        if written % 2000 == 0:
            print(f"  {written}/{len(instruments)}...")

    print(f"Done! {written} stocks written")


if __name__ == "__main__":
    main()
