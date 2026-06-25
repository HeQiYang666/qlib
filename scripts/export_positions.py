"""导出 round 目录的每日持仓（来自回测 artifact positions_normal_1day.pkl）。

用法:
  python scripts/export_positions.py <result_dir> [-o <output_dir>]

  python scripts/export_positions.py backtest_results/.../round_6_xxx/

输出（默认写到 <result_dir>/positions_export/）:
  daily/YYYY-MM-DD.csv  每天一个：instrument, amount, price, weight, count_day（按权重降序）
  summary.csv           每天一行：持仓数 / 账户市值 / 现金 / 当日新进只数 / 当日卖出只数（看换手）
  weights.parquet       宽表：行=日期, 列=股票, 值=组合权重（当天未持有为 NaN）

说明:
  positions_normal_1day.pkl 是 PortAnaRecord 保存的逐日持仓，
  结构为 dict{交易日 -> Position}，每个 Position.position 是
  {instrument -> {amount, price, weight, count_day}} 再加 cash / now_account_value。
"""
import sys
from pathlib import Path
import pandas as pd


def find_positions(result_dir: Path) -> Path:
    cands = list(result_dir.rglob("positions_normal_1day.pkl"))
    if not cands:
        raise FileNotFoundError(f"未找到 positions_normal_1day.pkl: {result_dir}")
    return cands[0]


def convert_code(code: str) -> str:
    prefix = code[:2]
    suffix = {"sh": "SH", "sz": "SZ", "bj": "BJ"}.get(prefix, prefix.upper())
    return code[2:] + "." + suffix


def main():
    import argparse

    parser = argparse.ArgumentParser(description="导出 round 目录的每日持仓")
    parser.add_argument("result_dir", help="回测结果目录（含 positions_normal_1day.pkl）")
    parser.add_argument("--output-dir", "-o", default=None,
                        help="输出目录，默认 <result_dir>/positions_export/")
    parsed = parser.parse_args()

    result_dir = Path(parsed.result_dir).resolve()
    if not result_dir.is_dir():
        print(f"错误: result_dir 不存在: {result_dir}")
        sys.exit(1)

    out_dir = Path(parsed.output_dir) if parsed.output_dir else result_dir / "positions_export"
    daily_dir = out_dir / "daily"
    daily_dir.mkdir(parents=True, exist_ok=True)

    pos_path = find_positions(result_dir)
    # 受信任本地源：回测自身产出的 artifact，非外部输入
    pos = pd.read_pickle(pos_path)
    days = sorted(pos.keys())
    print(f"持仓文件: {pos_path}")
    print(f"交易日: {len(days)}  ({days[0].date()} ~ {days[-1].date()})")

    summary_rows = []
    weight_by_day = {}
    prev_set: set = set()

    for day in days:
        pdict = pos[day].position
        # 股票持仓的 value 是 dict（amount/price/weight/count_day）；
        # cash / now_account_value / cash_delay 等 meta 字段都是标量，按类型过滤更稳健
        stocks = {k: v for k, v in pdict.items() if isinstance(v, dict)}

        df = pd.DataFrame(stocks).T
        if not df.empty:
            df = df.sort_values("weight", ascending=False)
            df.index = df.index.map(convert_code)
            df.index.name = "instrument"
        df.to_csv(daily_dir / (day.strftime("%Y-%m-%d") + ".csv"))

        cur_set = set(stocks.keys())
        summary_rows.append({
            "date": day.strftime("%Y-%m-%d"),
            "n_holdings": len(stocks),
            "account_value": pdict.get("now_account_value", float("nan")),
            "cash": pdict.get("cash", float("nan")),
            "n_new": len(cur_set - prev_set),
            "n_exit": len(prev_set - cur_set),
        })
        prev_set = cur_set

        weight_by_day[day.strftime("%Y-%m-%d")] = {
            convert_code(k): v.get("weight") for k, v in stocks.items()
        }

    summary = pd.DataFrame(summary_rows).set_index("date")
    summary.to_csv(out_dir / "summary.csv")

    weights = pd.DataFrame.from_dict(weight_by_day, orient="index")
    weights = weights.reindex(summary.index)  # 补回空仓日（全 NaN），与 daily 天数一致
    weights.index.name = "date"
    weights = weights.sort_index(axis=1)
    weights.to_parquet(out_dir / "weights.parquet")

    print(f"\n已导出至 {out_dir}/")
    print(f"  daily/           {len(days)} 个逐日 CSV")
    print(f"  summary.csv      每日持仓数 / 市值 / 换手")
    print(f"  weights.parquet  宽表 {weights.shape[0]} 天 x {weights.shape[1]} 只股票")

    print("\nsummary 预览（首尾各 3 天）:")
    print(pd.concat([summary.head(3), summary.tail(3)]))
    if len(summary) > 1:
        # 第一天 n_new=全部持仓，算换手时跳过
        print(f"\n平均每日: 新进 {summary['n_new'][1:].mean():.1f} 只 / "
              f"卖出 {summary['n_exit'][1:].mean():.1f} 只")


if __name__ == "__main__":
    main()
