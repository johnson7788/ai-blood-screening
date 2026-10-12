#!/usr/bin/env python3
"""把 NHANES 各模块按 SEQN 合并成建模宽表，并输出可用性报告。

输入：raw/<cycle>/*.xpt
输出：nhanes_wide.parquet  跨周期纵向合并宽表
      availability_report.txt  各变量可用性
"""
import glob
import os
import sys

import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
RAW = os.path.join(HERE, "raw")
OUT_PARQUET = os.path.join(HERE, "nhanes_wide.parquet")
OUT_REPORT = os.path.join(HERE, "availability_report.txt")

CYCLES = ["2013-2014", "2015-2016", "2017-2018", "2021-2023"]

# 关注的核心变量（论文1 的输入特征对应物 + 结局标签）
CORE = {
    # 人口学 / 权重
    "RIDAGEYR": "年龄", "RIAGENDR": "性别", "RIDRETH1": "种族",
    "INDFMPIR": "家庭收入贫困比", "BMXBMI": "BMI", "BMXWAIST": "腰围",
    "WTMEC2YR": "体检权重", "SDMVPSU": "PSU", "SDMVSTRA": "分层",
    # 血常规 CBC
    "LBXWBCSI": "白细胞", "LBXLYPCT": "淋巴细胞%", "LBXMOPCT": "单核细胞%",
    "LBXNEPCT": "中性粒细胞%", "LBXEOPCT": "嗜酸细胞%", "LBXBAPCT": "嗜碱细胞%",
    "LBXRBCSI": "红细胞", "LBXHGB": "血红蛋白", "LBXHCT": "红细胞压积",
    "LBXMCVSI": "MCV", "LBXMCHSI": "MCH", "LBXMC": "MCHC",
    "LBXPLTSI": "血小板", "LBXRDW": "RDW",
    # 生化 BIOPRO
    "LBXSCR": "肌酐", "LBXSUA": "尿酸", "LBXSBU": "尿素氮",
    "LBXSATSI": "ALT", "LBXSASSI": "AST", "LBXSTB": "总胆红素",
    "LBXSAL": "白蛋白", "LBXSCA": "钙", "LBXSNASI": "钠", "LBXSKSI": "钾",
    "LBXSGL": "血糖生化", "LBXSCH": "总胆固醇生化", "LBXSTR": "甘油三酯生化",
    # 血脂 / 糖化 / 胰岛素
    "LBXGH": "糖化血红蛋白", "LBDHDD": "HDL", "LBDLDL": "LDL",
    "LBDTRSI": "甘油三酯", "LBXTC": "总胆固醇",
    "LBXIN": "空腹胰岛素", "LBXGLU": "空腹血糖", "LBXHSCRP": "C反应蛋白",
    # 结局标签（问卷）
    "DIQ010": "糖尿病", "BPQ020": "高血压", "KIQ022": "肾病",
    "MCQ160B": "充血性心衰", "MCQ160C": "冠心病", "MCQ160D": "心绞痛",
    "MCQ160E": "心梗", "MCQ160F": "中风", "MCQ220": "癌症",
}


def load(cycle):
    """读入一个周期的全部模块并外连接。"""
    d = os.path.join(RAW, cycle)
    files = sorted(glob.glob(os.path.join(d, "*.xpt")))
    if not files:
        return None
    merged = None
    for f in files:
        try:
            df = pd.read_sas(f, format="xport")
        except Exception as e:
            print(f"  跳过 {os.path.basename(f)}: {e}", file=sys.stderr)
            continue
        df.columns = [c.decode() if isinstance(c, bytes) else c for c in df.columns]
        if "SEQN" not in df.columns:
            continue
        # 只保留核心变量，避免宽表爆炸
        keep = [c for c in df.columns if c in CORE or c == "SEQN"]
        if len(keep) <= 1:
            keep = ["SEQN"]
        df = df[keep]
        # 同名变量冲突时保留先到的（CBC 优先于 BIOPRO 的派生项）
        if merged is None:
            merged = df
        else:
            dup = [c for c in df.columns if c in merged.columns and c != "SEQN"]
            df = df.drop(columns=dup)
            if df.shape[1] > 1:
                merged = merged.merge(df, on="SEQN", how="outer")
    if merged is not None:
        merged.insert(0, "cycle", cycle)
    return merged


frames = []
for c in CYCLES:
    print(f"读取 {c} ...")
    r = load(c)
    if r is not None:
        print(f"  -> {r.shape[0]} 行 x {r.shape[1]} 列")
        frames.append(r)

if not frames:
    raise SystemExit("没有读到任何数据")

wide = pd.concat(frames, ignore_index=True, sort=False)

# 优先 parquet（体积小、保留类型），无 pyarrow 时回退 CSV
saved = None
try:
    wide.to_parquet(OUT_PARQUET, index=False)
    saved = OUT_PARQUET
except ImportError:
    saved = os.path.join(HERE, "nhanes_wide.csv")
    wide.to_csv(saved, index=False, encoding="utf-8-sig")
print(f"\n宽表已保存: {saved}")
print(f"总规模: {wide.shape[0]} 行 x {wide.shape[1]} 列")

lines = []
lines.append(f"NHANES 建模宽表可用性报告")
lines.append(f"总行数: {wide.shape[0]}   总列数: {wide.shape[1]}")
lines.append("=" * 72)
lines.append(f"{'周期':<12}{'行数':>8}")
lines.append("-" * 72)
for c, g in wide.groupby("cycle"):
    lines.append(f"{c:<12}{len(g):>8}")

lines.append("")
lines.append("=" * 72)
lines.append(f"{'变量':<12}{'含义':<20}{'非空数':>10}{'非空率':>10}")
lines.append("-" * 72)
n = len(wide)
for v, name in CORE.items():
    if v in wide.columns:
        nn = int(wide[v].notna().sum())
        lines.append(f"{v:<12}{name:<20}{nn:>10}{nn / n * 100:>9.1f}%")
    else:
        lines.append(f"{v:<12}{name:<20}{'-':>10}{'缺失':>10}")

report = "\n".join(lines)
open(OUT_REPORT, "w", encoding="utf-8").write(report)
print(report)
