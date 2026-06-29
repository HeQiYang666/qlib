"""用现成的每日预测值(pred.pkl)跑一次 TopkDropoutStrategy 回测, 产出 ret.pkl。

signal = predict_forward.py 复现出的每日预测分(pred.pkl, MultiIndex datetime×instrument, 列 score)。
除 topk / n_drop 外的回测参数与 round_5 完全一致(成本/涨跌停/ST/benchmark/account/deal_price),
改下面的 TOPK / N_DROP 跑一次即得对应参数的 ret.pkl
(= qlib report_normal_1day, 与 round 产物的 ret.pkl 同格式)。
"""

import pandas as pd

import qlib
from qlib.backtest import backtest
from qlib.contrib.strategy import TopkDropoutStrategy

# ===== 配置(改这里) =====
PRED = "/home/hqy/qlib/backtest_results/predictions/pred_20240701_20260601/pred.pkl"
TOPK = 400
N_DROP = 400
END = (
    None  # 回测结束日; None=用 pred 最大日期; 设 "2025-12-31" 可截断以精确对齐 round_5
)
BENCHMARK = "sh000852"
ACCOUNT = 100000000
OPEN_COST = 0.0005  # 买入费率
CLOSE_COST = 0.0015  # 卖出费率(含印花税)
MIN_COST = 5  # 单笔最低费用(元)
OUT_DIR = (
    "/home/hqy/qlib/backtest_results"  # 输出目录; 文件名自动带 topk/n_drop/起止日期
)

qlib.init(provider_uri="~/.qlib/qlib_data/cn_data", region="cn")

# 1. 读 signal (本地可信产物). MultiIndex (datetime, instrument), 列 score, 格式 shXXXXXX
pred = pd.read_pickle(PRED)
dates = pred.index.get_level_values("datetime")
start = dates.min()
end = pd.Timestamp(END) if END else dates.max()
OUT = f"{OUT_DIR}/ret_topk{TOPK}_ndrop{N_DROP}_{start:%Y%m%d}_{end:%Y%m%d}.pkl"
print(
    f"signal: {len(pred)} rows, {dates.nunique()} days, {start.date()} ~ {end.date()}"
)

n_drop = min(N_DROP, TOPK)  # n_drop 不能超过 topk

# 2. 策略 + executor + exchange (除 topk/n_drop 外全部对齐 round_5)
strategy = TopkDropoutStrategy(
    signal=pred, topk=TOPK, n_drop=n_drop, only_tradable=True
)
executor_config = {
    "class": "SimulatorExecutor",
    "module_path": "qlib.backtest.executor",
    "kwargs": {"time_per_step": "day", "generate_portfolio_metrics": True},
}
exchange_kwargs = {
    "limit_threshold": ["$limit_buy", "$limit_sell"],
    "subscribe_fields": ["$is_st"],
    "deal_price": "close",
    "open_cost": OPEN_COST,
    "close_cost": CLOSE_COST,
    "min_cost": MIN_COST,
}

# 3. 回测
print(
    f"running backtest: topk={TOPK}, n_drop={n_drop}, {start.date()} ~ {end.date()} ..."
)
portfolio_metric_dict, _ = backtest(
    start_time=start,
    end_time=end,
    strategy=strategy,
    executor=executor_config,
    benchmark=BENCHMARK,
    account=ACCOUNT,
    exchange_kwargs=exchange_kwargs,
)
report = portfolio_metric_dict["1day"][0]  # report_normal, 与 round 的 ret.pkl 同格式

# 4. 存 ret.pkl
report.to_pickle(OUT)
print(f"saved -> {OUT}")
print(f"columns: {list(report.columns)}")

# 5. 关键指标速览
if "return" in report.columns:
    r = report["return"]
    cum = (1 + r).prod() - 1
    ann = (1 + r).prod() ** (252 / len(r)) - 1
    sharpe = r.mean() / r.std() * (252**0.5) if r.std() > 0 else float("nan")
    print(f"\n=== summary (topk={TOPK}, n_drop={n_drop}) ===")
    print(
        f"  days={len(r)}  cum_return={cum * 100:+.2f}%  annualized={ann * 100:+.2f}%  sharpe={sharpe:.2f}"
    )
    if "bench" in report.columns:
        ex = r - report["bench"]
        cum_ex = (1 + ex).prod() - 1
        print(f"  cum_excess(vs {BENCHMARK})={cum_ex * 100:+.2f}%")
    if "turnover" in report.columns:
        print(f"  avg_turnover={report['turnover'].mean() * 100:.2f}%")
