"""NHANES 论文1 复现 —— 全局配置。"""
import os

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)                 # data/nhanes
PROJECT = os.path.dirname(os.path.dirname(ROOT))  # 项目根

WIDE_PARQUET = os.path.join(ROOT, "nhanes_wide.parquet")
WIDE_CSV = os.path.join(ROOT, "nhanes_wide.csv")
REPORTS = os.path.join(ROOT, "reports")
ARTIFACTS = os.path.join(ROOT, "artifacts")
os.makedirs(REPORTS, exist_ok=True)
os.makedirs(ARTIFACTS, exist_ok=True)

RANDOM_STATE = 42
N_SPLITS = 5

# 队列划分（对应论文1的「发现队列 + 内部验证 + 外部验证」）
TRAIN_CYCLES = ["2013-2014", "2015-2016"]   # 发现队列
INTERNAL_CYCLES = ["2017-2018"]             # 内部验证（同期不同人群）
EXTERNAL_CYCLES = ["2021-2023"]             # 外部时间验证（跨年代漂移）

# 抽样权重：合并 n 个 2 年周期，权重需除以 n（NHANES 官方规则）
WEIGHT_VAR = "WTMEC2YR"
STRATA_VAR = "SDMVSTRA"
PSU_VAR = "SDMVPSU"
FASTING_WEIGHT_VAR = "WTSAF2YR"

# 缺失率阈值（论文1：<20% 插补，>20% 剔除）
MISSING_IMPUTE_THRESHOLD = 0.20
