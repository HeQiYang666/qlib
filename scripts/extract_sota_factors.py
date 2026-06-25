#!/usr/bin/env python3
"""
从 backtest_results 目录汇总因子及回测指标。

Usage:
    python scripts/extract_sota_factors.py                         # 扫描所有 round_*
    python scripts/extract_sota_factors.py <round_dir>             # 指定单个目录
    python scripts/extract_sota_factors.py <dir1> <dir2> --json    # 多个目录，JSON 输出
"""

import argparse
import json
from pathlib import Path

import pandas as pd

KEY_METRICS = {
    "IC": "IC",
    "ICIR": "ICIR",
    "Rank IC": "Rank IC",
    "Rank ICIR": "Rank ICIR",
    "年化超额收益（含成本）": "1day.excess_return_with_cost.annualized_return",
    "最大回撤（含成本）": "1day.excess_return_with_cost.max_drawdown",
    "信息比率（含成本）": "1day.excess_return_with_cost.information_ratio",
}


def load_metrics(csv_path: Path) -> dict[str, float]:
    df = pd.read_csv(csv_path, header=None, names=["metric", "value"])
    metrics = {}
    for _, row in df.iterrows():
        k, v = str(row["metric"]).strip(), row["value"]
        if pd.notna(k) and pd.notna(v) and k:
            try:
                metrics[k] = float(v)
            except (ValueError, TypeError):
                metrics[k] = v
    return metrics


def collect_round(rd_dir: Path) -> dict:
    info_path = rd_dir / "factors" / "factor_info.json"
    csv_path = rd_dir / "qlib_res.csv"

    factors = []
    if info_path.exists():
        factors = json.loads(info_path.read_text())

    metrics = {}
    if csv_path.exists():
        metrics = load_metrics(csv_path)

    return {"dir": rd_dir.name, "factors": factors, "metrics": metrics}


def print_table(rounds: list[dict]):
    for rd in rounds:
        print(f"\n{'='*70}")
        print(f"  {rd['dir']}")
        print(f"{'='*70}")

        if rd["factors"]:
            print(f"\n  因子 ({len(rd['factors'])} 个):")
            for f in rd["factors"]:
                print(f"    - {f['name']}: {f['description'][:80]}")

        if rd["metrics"]:
            print(f"\n  回测指标:")
            for label, key in KEY_METRICS.items():
                val = rd["metrics"].get(key)
                if val is not None:
                    print(f"    {label:20s}: {val:12.4f}" if isinstance(val, float) else f"    {label:20s}: {val}")
        else:
            print("  (无回测指标)")


def print_json(rounds: list[dict]):
    output = []
    for rd in rounds:
        entry = {
            "round": rd["dir"],
            "factors": [],
            "metrics": {},
        }
        for f in rd["factors"]:
            entry["factors"].append({
                "name": f["name"],
                "description": f["description"],
                "formulation": f["formulation"],
            })
        for label, key in KEY_METRICS.items():
            val = rd["metrics"].get(key)
            if val is not None:
                entry["metrics"][label] = val
        output.append(entry)
    print(json.dumps(output, ensure_ascii=False, indent=2))


def main():
    parser = argparse.ArgumentParser(description="汇总 backtest_results 中的因子及回测指标")
    parser.add_argument("--json", action="store_true", help="JSON 输出")
    parser.add_argument(
        "round_dirs", nargs="*", type=Path,
        help="指定一个或多个 round 目录。不指定则扫描 backtest_results/ 下所有 round_*",
    )
    args = parser.parse_args()

    rounds = []
    if args.round_dirs:
        for p in args.round_dirs:
            p = p.resolve()
            if not p.is_dir():
                print(f"跳过不存在的目录: {p}")
                continue
            rounds.append(collect_round(p))
    else:
        results_dir = Path(__file__).resolve().parent.parent / "backtest_results"
        for rd_dir in sorted(results_dir.glob("round_*")):
            rounds.append(collect_round(rd_dir))

    if not rounds:
        print("未找到任何 round 目录。")
        return

    if args.json:
        print_json(rounds)
    else:
        print_table(rounds)


if __name__ == "__main__":
    main()
