"""数据加载、标签编码、队列构建与缺失处理。

关于缺失处理的重要说明
----------------------
论文1 的规则是「缺失 <20% 中位数插补，>20% 剔除」。但 NHANES 的缺失是
**结构性的**（子样本设计），不是条目级漏检：
  - CBC 只在体检子样本中测（约 81%/64%）
  - BIOPRO 只在更小的子样本中测（约 63%/53%）
  - 空腹项目（血糖/胰岛素/甘油三酯/LDL）只在 1/3 空腹子样本中测
若把这条规则机械套在**变量**上，会删掉几乎全部血液指标。

因此本复现把它改用在**样本**层面：先取 CBC+BIOPRO+人口学齐全的
完整案例队列（对应论文1 的「入组→纳入」过程），再在该队列内对残留的
条目级缺失套用 20% 阈值。这一改动在报告中明确记录。
"""
import numpy as np
import pandas as pd

from config import (EXTERNAL_CYCLES, INTERNAL_CYCLES, MISSING_IMPUTE_THRESHOLD,
                    TRAIN_CYCLES, WEIGHT_VAR, WIDE_PARQUET)
from variables import (DISEASES, DEMO_FEATURES, LEAKAGE_EXCLUSIONS,
                       feature_columns)

# 结构性缺失变量：不进主分析（空腹子样本、以及2013-2014未测的CRP）
STRUCTURAL_EXCLUDE = ["LBDLDL", "LBDTRSI", "LBXGLU", "LBXIN", "LBXHSCRP"]


def load_wide(path=WIDE_PARQUET):
    return pd.read_parquet(path)


def recode_outcome(s: pd.Series) -> pd.Series:
    """NHANES 问卷编码 -> 0/1；7(拒答) 9(不知道) 及缺失 -> NaN。"""
    out = s.where(s.isin([1, 2]))
    return out.map({1: 1.0, 2: 0.0})


def analysis_mask(df: pd.DataFrame) -> pd.Series:
    """完整案例掩码：CBC + BIOPRO + 人口学齐全。"""
    from variables import BIOPRO_FEATURES, CBC_FEATURES
    cols = CBC_FEATURES + BIOPRO_FEATURES + DEMO_FEATURES
    return df[cols].notna().all(axis=1)


def build_cohort(df, disease, feature_set="lab", cycles=None,
                 complete_case=True, drop_structural=True,
                 ignore_leakage=False):
    """构建建模队列。

    ignore_leakage=True 时**故意不剔除**该病种的同源诊断变量，用于量化
    标签泄漏带来的性能虚高（对照实验）。此时空腹血糖/胰岛素也不再按
    结构性缺失剔除，否则泄漏变量会被一并删掉、对照失效。

    返回 dict(X, y, w, cycle, var_names, dropped)
    """
    d = df.copy()
    if cycles:
        d = d[d["cycle"].isin(cycles)].copy()
    if complete_case:
        d = d[analysis_mask(d)].copy()

    # 标签
    lab_var = DISEASES[disease]["var"]
    d["_y"] = recode_outcome(d[lab_var])
    d = d[d["_y"].notna()].copy()

    # 特征
    if ignore_leakage:
        from variables import FEATURE_SETS as _FS
        cols = list(_FS[feature_set])
    else:
        cols = feature_columns(feature_set, disease=disease)
    if drop_structural:
        keep_for_leakage = {"LBXGLU", "LBXIN", "LBXGH", "LBXSGL",
                            "LBXSCR", "LBXSBU", "LBXSUA"} if ignore_leakage else set()
        cols = [c for c in cols
                if c not in STRUCTURAL_EXCLUDE or c in keep_for_leakage]
    cols = [c for c in cols if c in d.columns]

    # 队列内条目级缺失率，套用 20% 阈值
    miss = d[cols].isna().mean()
    dropped = miss[miss > MISSING_IMPUTE_THRESHOLD].index.tolist()
    cols = [c for c in cols if c not in dropped]

    X = d[cols].astype(float).copy()
    # 中位数插补
    med = X.median()
    X = X.fillna(med)

    # 抽样权重：合并 n 个 2 年周期需除以 n（NHANES 官方规则）
    n_cycles = d["cycle"].nunique()
    w = (d[WEIGHT_VAR] / n_cycles).astype(float)

    return {
        "X": X,
        "y": d["_y"].to_numpy(),
        "w": w.to_numpy(),
        "cycle": d["cycle"].to_numpy(),
        "var_names": cols,
        "dropped": dropped,
        "n": len(d),
        "prevalence": float(d["_y"].mean()),
        "weighted_prevalence": float(np.average(d["_y"], weights=w)),
    }


def describe_cohort(df, out_path=None):
    """加权 / 未加权描述统计——用于检验选择偏倚（坑①）。"""
    d = df[analysis_mask(df)].copy()
    rows = []
    for c in sorted(d["cycle"].unique()):
        g = d[d["cycle"] == c]
        n = len(g)
        w = g[WEIGHT_VAR] / 1
        rows.append({
            "周期": c,
            "n": n,
            "年龄_未加权": round(float(g["RIDAGEYR"].mean()), 2),
            "年龄_加权": round(float(np.average(g["RIDAGEYR"], weights=w)), 2),
            "女性比例_未加权": round(float((g["RIAGENDR"] == 2).mean()), 4),
            "女性比例_加权": round(float(np.average((g["RIAGENDR"] == 2).astype(float), weights=w)), 4),
            "BMI_未加权": round(float(g["BMXBMI"].mean()), 2),
            "BMI_加权": round(float(np.average(g["BMXBMI"], weights=w)), 2),
            "血红蛋白_未加权": round(float(g["LBXHGB"].mean()), 2),
            "血红蛋白_加权": round(float(np.average(g["LBXHGB"], weights=w)), 2),
        })
    out = pd.DataFrame(rows)
    if out_path:
        out.to_csv(out_path, index=False, encoding="utf-8-sig")
    return out


def leakage_table():
    """导出「标签—禁用变量」对照表。"""
    from variables import EXCLUSION_REASONS, VAR_LABELS
    rows = []
    for key, info in DISEASES.items():
        banned = LEAKAGE_EXCLUSIONS.get(key, [])
        if banned:
            for b in banned:
                rows.append({
                    "病种": info["name"], "标签变量": info["var"],
                    "禁用特征": b, "特征名": VAR_LABELS.get(b, b),
                    "剔除理由": EXCLUSION_REASONS.get(b, ""),
                })
        else:
            rows.append({
                "病种": info["name"], "标签变量": info["var"],
                "禁用特征": "（无）", "特征名": "—",
                "剔除理由": "诊断不依赖血液指标，无同源变量",
            })
    return pd.DataFrame(rows)
