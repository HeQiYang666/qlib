"""用已有模型对新的时间范围做预测（复用 qlib DataHandlerLP 管道）。

用法:
  python scripts/predict_forward.py <result_dir> <start_date> <end_date> [--output-dir <dir>]

  python scripts/predict_forward.py backtest_results/.../round_6_xxx/ 2025-01-01 2025-06-30
  python scripts/predict_forward.py backtest_results/.../round_6_xxx/ 2025-01-01 2025-06-30 -o my_preds/

原理:
  不再手算特征，而是用 qlib 现有管道重建训练时的 DataHandlerLP，再调用 model.predict：
  1. 加载训练好的 LGBModel（wrapper，不是裸 booster）
  2. 重建 NestedDataLoader = Alpha158DL(ALPHA20 base 特征) + StaticDataLoader(combined_factors_df.parquet)
  3. 重建 infer_processors: RobustZScoreNorm(在训练窗口上拟合) → Fillna —— 与回测完全一致
  4. DatasetH 划出 predict 段 [start, end]，model.predict 走 DK_I（infer）路径
  5. 这条路径与回测 SignalRecord 生成 pred.pkl 用的是同一套调用，故预测值与回测逐行一致
  6. 输出目录: 每天一个 YYYY-MM-DD.csv（instrument, score），并保存 pred.pkl 和 factors.parquet

约束:
  自定义因子取自 combined_factors_df.parquet。预测区间需落在该 parquet 的日期覆盖范围内，
  否则超出部分缺自定义因子，结果不可靠（脚本会给出警告）。

依赖: 数据已通过 qlib.init(provider_uri=...) 可访问
"""
import sys
import warnings
from pathlib import Path
import pandas as pd
import qlib
from qlib.utils import init_instance_by_config
from qlib.data.dataset import DatasetH
from qlib.data.dataset.handler import DataHandlerLP

# StaticDataLoader 不识别 instruments="all"，会回退到读取全量 parquet（正是所需），
# 该 UserWarning 无害，精准静音这一条（不影响其它警告）。
warnings.filterwarnings("ignore", message=r"If the value of .* cannot be processed", category=UserWarning)

# 与 RD-Agent 保持一致的 base features（rdagent/utils/qlib.py:ALPHA20）。
# 顺序即 ALPHA20 dict 顺序，与训练时 feature_names/feature_expressions 完全对齐。
BASE_FEATURES = {
    "RESI5": "Resi($close, 5)/$close",
    "WVMA5": "Std(Abs($close/Ref($close, 1)-1)*$volume, 5)/(Mean(Abs($close/Ref($close, 1)-1)*$volume, 5)+1e-12)",
    "RSQR5": "Rsquare($close, 5)",
    "KLEN": "($high-$low)/$open",
    "RSQR10": "Rsquare($close, 10)",
    "CORR5": "Corr($close, Log($volume+1), 5)",
    "CORD5": "Corr($close/Ref($close,1), Log($volume/Ref($volume, 1)+1), 5)",
    "CORR10": "Corr($close, Log($volume+1), 10)",
    "ROC60": "Ref($close, 60)/$close",
    "RESI10": "Resi($close, 10)/$close",
    "VSTD5": "Std($volume, 5)/($volume+1e-12)",
    "RSQR60": "Rsquare($close, 60)",
    "CORR60": "Corr($close, Log($volume+1), 60)",
    "WVMA60": "Std(Abs($close/Ref($close, 1)-1)*$volume, 60)/(Mean(Abs($close/Ref($close, 1)-1)*$volume, 60)+1e-12)",
    "STD5": "Std($close, 5)/$close",
    "RSQR20": "Rsquare($close, 20)",
    "CORD60": "Corr($close/Ref($close,1), Log($volume/Ref($volume, 1)+1), 60)",
    "CORD10": "Corr($close/Ref($close,1), Log($volume/Ref($volume, 1)+1), 10)",
    "CORR20": "Corr($close, Log($volume+1), 20)",
    "KLOW": "(Less($open, $close)-$low)/$open",
}

# RD-Agent .env 默认路径（其中 QLIB_FACTOR_train_* 定义训练切分窗口）
DEFAULT_ENV_PATH = Path.home() / "RD-Agent" / ".env"


def load_train_window(env_path: Path) -> tuple[str, str]:
    """从 RD-Agent 的 .env 读取 RobustZScoreNorm 的拟合窗口（训练切分）。

    必须与回测一致，否则归一化统计量不同，预测值会偏离。
    """
    if not env_path.exists():
        raise FileNotFoundError(f"未找到 RD-Agent .env: {env_path}（可用 --env-file 指定）")
    cfg = {}
    for line in env_path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, val = line.split("=", 1)
        cfg[key.strip()] = val.strip().strip('"').strip("'")
    try:
        return cfg["QLIB_FACTOR_train_start"], cfg["QLIB_FACTOR_train_end"]
    except KeyError as e:
        raise KeyError(f"{env_path} 缺少字段 {e}")


def load_model(result_dir: Path):
    """加载训练好的 LGBModel（返回 qlib wrapper，使其支持 model.predict(dataset)）。"""
    model_paths = list(result_dir.rglob("params.pkl"))
    if not model_paths:
        raise FileNotFoundError(f"未找到模型文件 params.pkl: {result_dir}")
    model = pd.read_pickle(model_paths[0])
    print(f"模型已加载: {type(model).__name__}, 期望特征数: {model.model.num_feature()}")
    return model


def find_factors_parquet(result_dir: Path) -> Path:
    """定位 combined_factors_df.parquet（自定义因子来源）。"""
    p = result_dir / "combined_factors_df.parquet"
    if not p.exists():
        raise FileNotFoundError(f"未找到自定义因子文件: {p}")
    return p


def build_handler_config(parquet_path: Path, end_date: str, train_start: str, train_end: str) -> dict:
    """重建训练时的 DataHandlerLP 配置（NestedDataLoader + 同款 processors）。"""
    return {
        "class": "DataHandlerLP",
        "module_path": "qlib.data.dataset.handler",
        "kwargs": {
            "start_time": train_start,  # 需覆盖 RobustZScoreNorm 拟合窗口
            "end_time": end_date,
            "instruments": "all",
            "data_loader": {
                "class": "NestedDataLoader",
                "module_path": "qlib.data.dataset.loader",
                "kwargs": {
                    "dataloader_l": [
                        {
                            "class": "qlib.contrib.data.loader.Alpha158DL",
                            "kwargs": {
                                "config": {
                                    "label": [["Ref($close, -2)/Ref($close, -1) - 1"], ["LABEL0"]],
                                    "feature": [list(BASE_FEATURES.values()), list(BASE_FEATURES.keys())],
                                }
                            },
                        },
                        {
                            "class": "qlib.data.dataset.loader.StaticDataLoader",
                            "kwargs": {"config": str(parquet_path)},
                        },
                    ]
                },
            },
            "infer_processors": [
                {
                    "class": "RobustZScoreNorm",
                    "kwargs": {
                        "fields_group": "feature",
                        "clip_outlier": True,
                        "fit_start_time": train_start,
                        "fit_end_time": train_end,
                    },
                },
                {"class": "Fillna", "kwargs": {"fields_group": "feature"}},
            ],
            "learn_processors": [
                {"class": "DropnaLabel"},
                {"class": "CSZScoreNorm", "kwargs": {"fields_group": "label"}},
            ],
        },
    }


def convert_code(code: str) -> str:
    prefix = code[:2]
    suffix = {"sh": "SH", "sz": "SZ", "bj": "BJ"}.get(prefix, prefix.upper())
    return code[2:] + "." + suffix


def main():
    import argparse

    parser = argparse.ArgumentParser(description="用已有模型对新的时间范围做预测（复用 qlib 管道）")
    parser.add_argument("result_dir", help="RD-Agent 实验结果目录（含 params.pkl 与 combined_factors_df.parquet）")
    parser.add_argument("start_date", help="起始日期 YYYY-MM-DD")
    parser.add_argument("end_date", help="结束日期 YYYY-MM-DD")
    parser.add_argument("--output-dir", "-o", default=None,
                        help="输出目录，默认为 backtest_results/predictions/pred_<start>_<end>/")
    parser.add_argument("--env-file", default=None,
                        help=f"RD-Agent .env 路径（读取训练切分窗口），默认 {DEFAULT_ENV_PATH}")
    parsed = parser.parse_args()

    result_dir = Path(parsed.result_dir).resolve()
    if not result_dir.is_dir():
        print(f"错误: result_dir 不存在: {result_dir}")
        sys.exit(1)

    start_date, end_date = parsed.start_date, parsed.end_date
    safe_start, safe_end = start_date.replace("-", ""), end_date.replace("-", "")
    out_dir = (Path(parsed.output_dir) if parsed.output_dir
               else Path(f"backtest_results/predictions/pred_{safe_start}_{safe_end}"))
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"日期范围: {start_date} ~ {end_date}")
    print(f"结果目录: {result_dir}")
    print(f"输出目录: {out_dir}")

    env_path = Path(parsed.env_file) if parsed.env_file else DEFAULT_ENV_PATH
    train_start, train_end = load_train_window(env_path)
    print(f"训练切分窗口（RobustZScoreNorm 拟合）: {train_start} ~ {train_end}  (来自 {env_path})")

    qlib.init(provider_uri="~/.qlib/qlib_data/cn_data")

    booster_wrapper = load_model(result_dir)
    parquet_path = find_factors_parquet(result_dir)

    # 检查自定义因子的日期覆盖
    factor_dts = pd.read_parquet(parquet_path).index.get_level_values("datetime")
    f_min, f_max = factor_dts.min(), factor_dts.max()
    print(f"自定义因子覆盖: {f_min.date()} ~ {f_max.date()}")
    if pd.Timestamp(end_date) > f_max or pd.Timestamp(start_date) < f_min:
        print(f"  警告: 预测区间 [{start_date}, {end_date}] 超出因子覆盖范围，"
              f"超出部分缺自定义因子，结果不可靠。")

    # 重建管道并预测
    print("正在重建 DataHandlerLP 管道（Alpha158 ALPHA20 + 自定义因子 + RobustZScoreNorm/Fillna）...")
    handler = init_instance_by_config(build_handler_config(parquet_path, end_date, train_start, train_end))
    dataset = DatasetH(handler, segments={"predict": (start_date, end_date)})

    print("正在预测（DK_I 路径，与回测 SignalRecord 一致）...")
    scores = booster_wrapper.predict(dataset, segment="predict")
    scores = scores.dropna()
    if scores.empty:
        print("错误: 预测结果为空。检查日期范围内是否有数据 / 因子是否覆盖。")
        sys.exit(1)
    print(f"预测完成: {len(scores)} 条记录")

    pred = scores.to_frame(name="score")
    if pred.index.names != ["datetime", "instrument"]:
        pred.index = pred.index.reorder_levels(["datetime", "instrument"])
    pred = pred.sort_index()

    # 处理后的特征矩阵（模型实际看到的输入），用于核对
    factor_df = dataset.prepare("predict", col_set="feature", data_key=DataHandlerLP.DK_I)
    factor_df = factor_df.loc[pred.index]

    # 保存: 每天一个 CSV + pred.pkl + factors.parquet
    for date, group in pred.groupby(level="datetime"):
        daily = group.reset_index("datetime", drop=True)
        daily.index = daily.index.map(convert_code)
        daily.to_csv(out_dir / (date.strftime("%Y-%m-%d") + ".csv"))

    pred.to_pickle(out_dir / "pred.pkl")
    factor_df.to_parquet(out_dir / "factors.parquet")

    n_days = pred.index.get_level_values("datetime").nunique()
    print(f"\n预测结果已保存至 {out_dir}/")
    print(f"  每日 CSV: {n_days} 个文件")
    print(f"  pred.pkl, factors.parquet")

    first_date = pred.index.get_level_values("datetime")[0]
    print(f"\n{first_date.strftime('%Y-%m-%d')}.csv 样例:")
    sample = pred.xs(first_date, level="datetime").copy()
    sample.index = sample.index.map(convert_code)
    print(sample.head(10))

    print("\n统计:")
    print(f"  score 均值:    {pred['score'].mean():.6f}")
    print(f"  score 标准差:  {pred['score'].std():.6f}")
    print(f"  覆盖股票数:    {pred.index.get_level_values('instrument').nunique()}")
    print(f"  覆盖交易日数:  {n_days}")


if __name__ == "__main__":
    main()
