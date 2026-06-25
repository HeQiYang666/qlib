"""绘制回测收益曲线及关键指标。

用法: python scripts/plot_backtest.py <ret.pkl> [输出.png] [--start YYYY-MM-DD] [--end YYYY-MM-DD]
自动对齐 backtest_results/c20_report.pkl 到相同日期范围。
"""

import argparse
from pathlib import Path

C20_DEFAULT = Path(__file__).resolve().parent.parent / "backtest_results" / "c20_report.pkl"

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import matplotlib.ticker as mticker

# 自动匹配系统可用的 CJK 字体
for name in ["Noto Sans CJK SC", "WenQuanYi Micro Hei", "AR PL UMing CN", "SimHei"]:
    if any(f.name == name for f in fm.fontManager.ttflist):
        plt.rcParams["font.sans-serif"] = [name, "DejaVu Sans"]
        plt.rcParams["font.family"] = "sans-serif"
        break
plt.rcParams["axes.unicode_minus"] = False


def _compute_metrics(df):
    """Compute cumulative returns and metrics from a report DataFrame."""
    if not isinstance(df.index, pd.DatetimeIndex):
        df = df.copy()
        df.index = pd.to_datetime(df.index)

    cum_strategy = (1 + df["return"] - df["cost"]).cumprod()
    cum_benchmark = (1 + df["bench"]).cumprod()
    excess = cum_strategy / cum_benchmark

    n_days = len(df)
    return {
        "cum_strategy": cum_strategy,
        "cum_benchmark": cum_benchmark,
        "excess": excess,
        "ann_strategy": cum_strategy.iloc[-1] ** (252 / n_days) - 1,
        "ann_benchmark": cum_benchmark.iloc[-1] ** (252 / n_days) - 1,
        "final_strategy": cum_strategy.iloc[-1] - 1,
        "final_benchmark": cum_benchmark.iloc[-1] - 1,
        "max_dd_strategy": (cum_strategy / cum_strategy.cummax() - 1).min(),
        "max_dd_benchmark": (cum_benchmark / cum_benchmark.cummax() - 1).min(),
        "avg_turnover": df["turnover"].mean(),
    }


def plot_report(ret_path: str, output: str | None = None, compare_path: str | None = None,
                start_date: str | None = None, end_date: str | None = None):
    df = pd.read_pickle(ret_path)
    if not isinstance(df.index, pd.DatetimeIndex):
        df.index = pd.to_datetime(df.index)
    if start_date is not None:
        df = df.loc[pd.Timestamp(start_date):]
    if end_date is not None:
        df = df.loc[:pd.Timestamp(end_date)]
    m = _compute_metrics(df)

    # Auto-load C20 baseline and align to same date range
    mc = None
    dfc = None
    if compare_path:
        dfc = pd.read_pickle(compare_path)
    elif C20_DEFAULT.exists():
        dfc = pd.read_pickle(str(C20_DEFAULT))
        dfc = dfc.loc[df.index[0]:df.index[-1]]
    if dfc is not None:
        if not isinstance(dfc.index, pd.DatetimeIndex):
            dfc.index = pd.to_datetime(dfc.index)
        if len(dfc) > 0:
            mc = _compute_metrics(dfc)

    fig, ax = plt.subplots(figsize=(14, 7))

    # Main strategy (Scenario B — RD-Agent)
    ax.plot(m["cum_strategy"].index, m["cum_strategy"], linewidth=1.5, color="#2563EB",
            label="策略收益 (场景B·RD-Agent)")
    ax.plot(m["excess"].index, m["excess"], linewidth=1.2, color="#DC2626", alpha=0.8,
            label="超额收益 (场景B)")

    # Comparison strategy (Scenario C — C20 baseline with preprocessing)
    if mc:
        ax.plot(mc["cum_strategy"].index, mc["cum_strategy"], linewidth=1.5, color="#22C55E",
                label="策略收益 (场景C·C20 baseline)")
        ax.plot(mc["excess"].index, mc["excess"], linewidth=1.2, color="#F59E0B", alpha=0.8,
                label="超额收益 (场景C)")

    ax.plot(m["cum_benchmark"].index, m["cum_benchmark"], linewidth=1.2, color="#9CA3AF",
            linestyle="--", label="基准收益 (中证1000)")
    ax.axhline(y=1, color="black", linewidth=0.4, alpha=0.3)

    ax.legend(loc="upper left", fontsize=10)
    ax.set_ylabel("累计收益")
    ax.grid(True, alpha=0.2)
    ax.yaxis.set_major_formatter(mticker.PercentFormatter(1.0))

    ann_excess = m["ann_strategy"] - m["ann_benchmark"]
    final_excess = m["excess"].iloc[-1] - 1
    lines = [
        f"场景B  年化: {m['ann_strategy']:+.2%}  累计: {m['final_strategy']:+.2%}  超额: {ann_excess:+.2%}  最大回撤: {m['max_dd_strategy']:.2%}  换手: {m['avg_turnover']:.3%}",
    ]
    if mc:
        ann_excess_c = mc["ann_strategy"] - mc["ann_benchmark"]
        final_excess_c = mc["excess"].iloc[-1] - 1
        lines.append(
            f"场景C  年化: {mc['ann_strategy']:+.2%}  累计: {mc['final_strategy']:+.2%}  超额: {ann_excess_c:+.2%}  最大回撤: {mc['max_dd_strategy']:.2%}  换手: {mc['avg_turnover']:.3%}"
        )
    lines.append(f"基准  年化: {m['ann_benchmark']:+.2%}  最大回撤: {m['max_dd_benchmark']:.2%}")

    fig.tight_layout(rect=[0, 0.15, 1, 1])
    fig.text(0.5, 0.04, "\n".join(lines), ha="center", fontsize=10,
             bbox=dict(boxstyle="round", facecolor="#FAFAFA", edgecolor="#D1D5DB", alpha=0.95))

    out_path = output or Path(ret_path).with_suffix(".png")
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.show()
    print(f"已保存至 {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="绘制回测收益曲线及关键指标")
    parser.add_argument("ret_path", help="回测结果的 ret.pkl 文件路径")
    parser.add_argument("output", nargs="?", default=None, help="输出图片路径，默认与 ret.pkl 同名 .png")
    parser.add_argument("--start", dest="start_date", default=None, help="起始日期 YYYY-MM-DD")
    parser.add_argument("--end", dest="end_date", default=None, help="结束日期 YYYY-MM-DD")
    args = parser.parse_args()
    plot_report(args.ret_path, args.output, start_date=args.start_date, end_date=args.end_date)
