#!/usr/bin/env python3
"""
用 ALPHA20 + LGBM (C20 baseline pipeline) 对指定日期范围的每只股票生成每日预测值。

数据管道: Alpha158DL(ALPHA20) + RobustZScoreNorm + Fillna + LightGBM
对标: backtest_results/baseline_metrics.txt 的 Scenario C20。

使用方法:
    python scripts/predict_future.py --future-start 2026-01-01 --future-end 2026-06-15

输出:
    backtest_results/predictions/<start>_<end>/
        YYYY-MM-DD.csv    (instrument, score)
"""

import argparse
import logging
import os
from contextlib import redirect_stdout
from pathlib import Path

# 禁止可选模型依赖导入时的 print 提示
with redirect_stdout(open(os.devnull, "w")):
    import lightgbm as lgb
    import qlib
    from qlib.contrib.data.loader import Alpha158DL
    from qlib.contrib.data.handler import DataHandlerLP
    from qlib.contrib.model.gbdt import LGBModel
    from qlib.data.dataset import DatasetH
    from qlib.data.dataset.loader import NestedDataLoader
    from qlib.log import get_module_logger

logger = get_module_logger("predict_future")

# RD-Agent's ALPHA20 — 20 representative Alpha158 expressions
ALPHA20_EXPR = [
    "Corr($close/Ref($close,1), Log($volume/Ref($volume, 1)+1), 10)",
    "Corr($close/Ref($close,1), Log($volume/Ref($volume, 1)+1), 5)",
    "Corr($close/Ref($close,1), Log($volume/Ref($volume, 1)+1), 60)",
    "Corr($close, Log($volume+1), 10)",
    "Corr($close, Log($volume+1), 20)",
    "Corr($close, Log($volume+1), 5)",
    "Corr($close, Log($volume+1), 60)",
    "($high-$low)/$open",
    "(Less($open, $close)-$low)/$open",
    "Resi($close, 10)/$close",
    "Resi($close, 5)/$close",
    "Ref($close, 60)/$close",
    "Rsquare($close, 10)",
    "Rsquare($close, 20)",
    "Rsquare($close, 5)",
    "Rsquare($close, 60)",
    "Std($close, 5)/$close",
    "Std($volume, 5)/($volume+1e-12)",
    "Std(Abs($close/Ref($close, 1)-1)*$volume, 5)/(Mean(Abs($close/Ref($close, 1)-1)*$volume, 5)+1e-12)",
    "Std(Abs($close/Ref($close, 1)-1)*$volume, 60)/(Mean(Abs($close/Ref($close, 1)-1)*$volume, 60)+1e-12)",
]
ALPHA20_NAMES = [
    "CORD10", "CORD5", "CORD60", "CORR10", "CORR20", "CORR5", "CORR60",
    "KLEN", "KLOW", "RESI10", "RESI5", "ROC60", "RSQR10", "RSQR20",
    "RSQR5", "RSQR60", "STD5", "VSTD5", "WVMA5", "WVMA60",
]


def main():
    parser = argparse.ArgumentParser(description="ALPHA20 + LGBM (C20 baseline) 未来数据预测")
    parser.add_argument("--future-start", required=True, help="未来数据起始日期, e.g. 2026-01-01")
    parser.add_argument("--future-end", required=True, help="未来数据结束日期, e.g. 2026-06-15")
    parser.add_argument("--provider-uri", default="~/.qlib/qlib_data/cn_data", help="Qlib 数据路径")
    parser.add_argument("--market", default="all", help="股票池")
    parser.add_argument("--train-start", default="2019-01-01", help="训练起始日期")
    parser.add_argument("--train-end", default="2022-12-31", help="训练结束日期")
    parser.add_argument("--valid-start", default="2023-01-01", help="验证起始日期")
    parser.add_argument("--valid-end", default="2024-06-30", help="验证结束日期")
    parser.add_argument("--output-dir", default="backtest_results/predictions", help="输出目录")
    parser.add_argument("--model-path", default=None, help="已有模型路径，跳过训练直接加载")
    args = parser.parse_args()

    # 1. 初始化 Qlib
    provider_uri = Path(args.provider_uri).expanduser().as_posix()
    qlib.init(provider_uri=provider_uri, region="cn")
    logger.info(f"Qlib initialized, data_path={provider_uri}")

    # 2. 构造 DataHandler (C20 pipeline: Alpha158DL + RobustZScoreNorm + Fillna)
    handler_config = {
        "start_time": args.train_start,
        "end_time": args.future_end,
        "instruments": args.market,
        "data_loader": {
            "class": "NestedDataLoader",
            "kwargs": {
                "dataloader_l": [
                    {
                        "class": "qlib.contrib.data.loader.Alpha158DL",
                        "kwargs": {
                            "config": {
                                "label": [
                                    ["Ref($close, -2)/Ref($close, -1) - 1"],
                                    ["LABEL0"],
                                ],
                                "feature": [ALPHA20_EXPR, ALPHA20_NAMES],
                            }
                        },
                    }
                ]
            },
        },
        "infer_processors": [
            {
                "class": "RobustZScoreNorm",
                "kwargs": {
                    "fields_group": "feature",
                    "clip_outlier": True,
                    "fit_start_time": args.train_start,
                    "fit_end_time": args.train_end,
                },
            },
            {"class": "Fillna", "kwargs": {"fields_group": "feature"}},
        ],
        "learn_processors": [
            {"class": "DropnaLabel"},
            {"class": "CSZScoreNorm", "kwargs": {"fields_group": "label"}},
        ],
    }
    handler = DataHandlerLP(**handler_config)
    logger.info(f"ALPHA20 DataHandler configured with {len(ALPHA20_NAMES)} features")

    # 3. 构造 Dataset
    segments = {
        "train": (args.train_start, args.train_end),
        "valid": (args.valid_start, args.valid_end),
        "future": (args.future_start, args.future_end),
    }
    dataset = DatasetH(handler=handler, segments=segments)
    logger.info(f"Dataset segments: {segments}")

    # 4. 训练或加载模型
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.model_path and Path(args.model_path).exists():
        logger.info(f"Loading model from {args.model_path}")
        model = LGBModel(
            loss="mse",
            colsample_bytree=0.8879,
            learning_rate=0.2,
            subsample=0.8789,
            lambda_l1=205.6999,
            lambda_l2=580.9768,
            max_depth=8,
            num_leaves=210,
            num_threads=20,
            seed=42,
        )
        model.model = lgb.Booster(model_file=args.model_path)
    else:
        model = LGBModel(
            loss="mse",
            colsample_bytree=0.8879,
            learning_rate=0.2,
            subsample=0.8789,
            lambda_l1=205.6999,
            lambda_l2=580.9768,
            max_depth=8,
            num_leaves=210,
            num_threads=20,
            seed=42,
            early_stopping_rounds=50,
        )
        logger.info("Training model with ALPHA20 (C20 pipeline) ...")
        model.fit(dataset, reweighter=None)

        # 验证集评估
        valid_pred = model.predict(dataset, segment="valid")
        logger.info(f"Validation predictions shape: {valid_pred.shape}")

        # 保存模型 (LightGBM 原生格式，不含代码执行)
        model_file = output_dir / "lgb_model.txt"
        model.model.save_model(str(model_file))
        logger.info(f"Model saved to {model_file}")

    # 5. 对未来数据预测
    logger.info(f"Predicting on future data: {args.future_start} ~ {args.future_end}")
    pred = model.predict(dataset, segment="future")
    logger.info(f"Predictions: {len(pred)} rows, index sample: {pred.index[:3].tolist()}")

    # 6. 保存预测结果：每天一个 CSV
    safe_start = args.future_start.replace("-", "")
    safe_end = args.future_end.replace("-", "")
    out_dir = output_dir / f"pred_{safe_start}_{safe_end}_baseline"
    out_dir.mkdir(parents=True, exist_ok=True)

    pred_s = pred.rename("score")

    # 股票代码转换: sh600000 → 600000.SH
    def convert_code(code):
        prefix = code[:2]
        suffix = {"sh": "SH", "sz": "SZ", "bj": "BJ"}.get(prefix, prefix.upper())
        return code[2:] + "." + suffix

    for date, group in pred_s.groupby("datetime"):
        fname = date.strftime("%Y-%m-%d") + ".csv"
        daily = group.reset_index("datetime", drop=True)
        daily.index = daily.index.map(convert_code)
        daily.to_csv(out_dir / fname)
    logger.info(f"Predictions saved to {out_dir}/ ({pred_s.groupby('datetime').ngroups} daily files)")

    # 7. 预览
    first_date = pred_s.index.get_level_values("datetime")[0].strftime("%Y-%m-%d")
    print(f"\n=== 预测结果预览 ===")
    print(f"输出目录: {out_dir}/")
    print(f"日期数: {pred_s.groupby('datetime').ngroups} 天")
    print(f"日均股票数: {pred_s.groupby('datetime').size().mean():.0f}")
    print(f"\n{first_date}.csv 样例:")
    sample = pred_s.loc[pred_s.index.get_level_values("datetime") == first_date].reset_index("datetime", drop=True)
    sample.index = sample.index.map(convert_code)
    print(sample.head(10))


if __name__ == "__main__":
    main()
