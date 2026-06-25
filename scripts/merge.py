import os
import pandas as pd

# ===== 配置 =====
data_dir = "backtest_results/predictions/pred_20240701_20260101"   # 存放csv文件的目录
output_file = "merged.csv"
fill_value = -1e9     # 非常负的值（你可以改）

# ===== 读取所有文件 =====
all_dfs = []

for file in os.listdir(data_dir):
    if file.endswith(".csv"):
        date = file.replace(".csv", "")  # 从文件名提取日期
        
        file_path = os.path.join(data_dir, file)
        df = pd.read_csv(file_path)
        
        # 确保列名正确
        df.columns = ["instrument", "score"]
        
        # 增加一列日期
        df["date"] = date
        
        all_dfs.append(df)

# ===== 合并所有数据 =====
big_df = pd.concat(all_dfs, ignore_index=True)

# ===== pivot 成目标格式 =====
pivot_df = big_df.pivot(index="instrument", columns="date", values="score")

# ===== 填充缺失值 =====
pivot_df = pivot_df.fillna(fill_value)

# ===== 可选：按日期排序列 =====
pivot_df = pivot_df.reindex(sorted(pivot_df.columns), axis=1)

# ===== 保存 =====
pivot_df.to_csv(output_file)

print("完成！输出文件：", output_file)
