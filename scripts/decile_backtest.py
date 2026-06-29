"""按模型预测值做全市场 10 分组(decile)累计收益回测。

每个信号日 t:
  - 用 pred.pkl 里的 score 把当天全市场股票按等频分成 10 组(G1 最低分 ~ G10 最高分);
  - 每组当期收益 = 组内股票 label 的等权平均;
  - label = Ref($close,-2)/Ref($close,-1)-1, 即 t+1 收盘买入、t+2 收盘卖出的单日收益
    (与模型训练时的 LABEL0 定义一致, 对齐到信号日 t)。

不考虑涨跌停/ST/停牌/手续费; label 缺失(停牌等导致 t+1 或 t+2 无价)的样本直接丢弃。
累计收益用复利: cumprod(1 + 每期组收益) - 1。
"""

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt

import qlib
from qlib.data import D

PRED = "/home/hqy/qlib/backtest_results/predictions/pred_20240701_20260601/pred.pkl"
OUT_PNG = "/home/hqy/qlib/backtest_results/decile_cumret.png"
OUT_CSV = "/home/hqy/qlib/backtest_results/decile_cumret.csv"
N_GROUPS = 10
LABEL_EXPR = "Ref($close, -2)/Ref($close, -1) - 1"

qlib.init(provider_uri="~/.qlib/qlib_data/cn_data", region="cn")

# 1. 读 score (本地可信产物). MultiIndex (datetime, instrument), 列 'score', 已是 qlib 格式 shXXXXXX
pred = pd.read_pickle(PRED)
score = pred["score"]
dates = score.index.get_level_values("datetime")
start, end = dates.min(), dates.max()
print(
    f"score: {len(score)} rows, {dates.nunique()} days, {start.date()} ~ {end.date()}"
)

# 2. 用 qlib 算 label(未来收益). end 延后 ~1 个月保证最后一个信号日的 t+2 价格存在
label_end = end + pd.Timedelta(days=40)
print(f"computing label via qlib D.features ... (end buffered to {label_end.date()})")
lab = D.features(
    D.instruments("all"), [LABEL_EXPR], start_time=start, end_time=label_end
)
lab.columns = ["ret"]
print(
    f"label raw: {len(lab)} rows, index_names={list(lab.index.names)}, sample={lab.index[:2].tolist()}"
)
# D.features 返回 (instrument, datetime) 且 instrument 大写; 规整到与 pred 一致: (datetime, instrument)、小写
lab = lab.reset_index()
lab["instrument"] = lab["instrument"].str.lower()
lab_ret = lab.set_index(["datetime", "instrument"])["ret"]

# 3. 对齐 score 与 label
df = pd.DataFrame({"score": score, "ret": lab_ret}).dropna()
print(
    f"after join+dropna: {len(df)} rows ({len(df) / len(score):.1%} of score rows kept)"
)

# 4. 每个信号日内按 score 百分位等频分 10 组(纯向量化, 无 qcut 边界坑)
pct = df.groupby(level="datetime")["score"].rank(pct=True, method="first")
df["grp"] = np.ceil(pct * N_GROUPS).clip(1, N_GROUPS).astype(int)
print("avg stocks per group per day:")
print(df.groupby("grp").size() / dates.nunique())

# 5. 每天每组等权平均收益 -> (datetime x grp) 矩阵
daily = (
    df.groupby([df.index.get_level_values("datetime"), "grp"])["ret"]
    .mean()
    .unstack("grp")
)
daily = daily.sort_index()
daily.columns = [int(c) for c in daily.columns]
daily = daily[sorted(daily.columns)]

# 6. 复利累计
cum = (1 + daily).cumprod() - 1
cum.to_csv(OUT_CSV)
print(f"saved cumulative returns -> {OUT_CSV}")

# 7. 画图: 10 条累计收益曲线
fig, ax = plt.subplots(figsize=(13, 7))
colors = plt.cm.RdYlGn(np.linspace(0, 1, N_GROUPS))
for i, g in enumerate(sorted(cum.columns)):
    lbl = f"G{g}" + (
        " (lowest score)" if g == 1 else " (highest score)" if g == N_GROUPS else ""
    )
    ax.plot(cum.index, cum[g] * 100, color=colors[i], label=lbl, lw=1.6)
ax.axhline(0, color="gray", lw=0.8, ls="--")
ax.set_title(
    "Decile cumulative return by predicted score (t+2 label, equal-weight, no costs/limits)"
)
ax.set_xlabel("signal date")
ax.set_ylabel("cumulative return (%)")
ax.legend(loc="upper left", ncol=2, fontsize=9)
ax.grid(alpha=0.3)
fig.tight_layout()
fig.savefig(OUT_PNG, dpi=130)
print(f"saved figure -> {OUT_PNG}")

# 8. 单调性概览
final = cum.iloc[-1].sort_index()
print("\n=== final cumulative return by group ===")
for g in sorted(final.index):
    print(f"  G{g:<2d}: {final[g] * 100:+7.2f}%")
ls = daily[N_GROUPS] - daily[1]
ls_cum = (1 + ls).cumprod() - 1
print(f"\nlong-short (G{N_GROUPS}-G1) final cum: {ls_cum.iloc[-1] * 100:+.2f}%")
mono = final.is_monotonic_increasing
print(f"monotonic increasing G1->G{N_GROUPS}: {mono}")
