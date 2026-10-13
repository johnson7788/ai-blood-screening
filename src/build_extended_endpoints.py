"""Phase 1（扩充版）：提取疾病终点，构建扩充版宽表。

背景
----
原宽表只用了 7 个疾病标签。NHANES 问卷中有大量「医生是否告知患有 X」的
标准问法，可提供远多于 7 个的疾病终点，用于方向二的「多方法分歧度」系统验证。

终点来源（三类）
----------------
1. **经典疾病模块** MCQ / DIQ / KIQ / BPQ —— 「医生告知患病」标准问法
2. **扩展模块** RHQ（生殖）/ SLQ（睡眠）/ HEQ（肝炎）/ OSQ（骨质疏松）
3. **死亡终点** 来自 NCHS 公开版链接死亡率文件（LMF）

筛选标准
--------
- 标签为真正的疾病诊断（排除治疗、用药、症状、风险因素、医生建议类问题）
- 二元编码 1=是 / 2=否，阳性数 ≥100
- **主终点**：覆盖 ≥3 个抽样周期（可做跨周期外部验证）
- **次级终点**：覆盖 2 个周期（仅用于敏感性分析）

产出
----
nhanes_endpoints.parquet       扩充终点表
reports/endpoints_inventory.csv 终点清单与统计
"""
import glob
import os
import re
import sys
import time

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
OUT_PARQUET = os.path.join(ROOT, "nhanes_endpoints.parquet")
REPORTS = os.path.join(ROOT, "reports")
LMF = os.path.join(ROOT, "mortality", "lmf_parsed.parquet")
os.makedirs(REPORTS, exist_ok=True)

CYCLES = ["2013-2014", "2015-2016", "2017-2018", "2021-2023"]
MODULES = ["MCQ", "DIQ", "KIQ", "BPQ", "RHQ", "SLQ", "HEQ", "OSQ"]

# ---- 主终点：真正的疾病诊断，命名依据 XPT 文件自带的官方变量标签 ----
ENDPOINTS = {
    # 心血管 / 代谢
    "BPQ020": "高血压",
    "BPQ080": "高胆固醇",
    "DIQ010": "糖尿病",
    "DIQ160": "糖尿病前期",
    "MCQ160B": "充血性心衰",
    "MCQ160C": "冠心病",
    "MCQ160D": "心绞痛",
    "MCQ160E": "心梗",
    "MCQ160F": "中风",
    # 呼吸
    "MCQ010": "哮喘",
    "MCQ160G": "肺气肿",
    "MCQ160K": "慢性支气管炎",
    "MCQ160O": "慢阻肺",
    # 其他系统
    "MCQ160A": "关节炎",
    "MCQ160L": "肝脏疾病",
    "MCQ160M": "甲状腺问题",
    "MCQ160N": "痛风",
    "MCQ220": "癌症",
    "MCQ203": "黄疸",
    "MCQ550": "胆结石",
    # 肾脏
    "KIQ022": "肾功能不全",
    "KIQ026": "肾结石",
    # 扩展模块
    "RHQ162": "妊娠糖尿病",
    "SLQ050": "睡眠障碍",
    "HEQ010": "乙型肝炎",
    "HEQ030": "丙型肝炎",
    # 次级（仅 2 个周期）
    "OSQ060": "骨质疏松",
    "OSQ080": "骨折",
}

# 覆盖周期数少于 3 的终点（仅用于敏感性分析）
SECONDARY = {"OSQ060", "OSQ080", "MCQ550"}


def log(m):
    print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)


def recode(s):
    """NHANES 编码 1=是 2=否 -> 1/0；7(拒答) 9(不知道) 及缺失 -> NaN。"""
    return s.where(s.isin([1, 2])).map({1: 1.0, 2: 0.0})


def load_questionnaire():
    """从各周期的问卷模块按 SEQN 外连接，返回长表。"""
    frames = []
    for cyc in CYCLES:
        parts = []
        for f in sorted(glob.glob(os.path.join(ROOT, "raw", cyc, "*.xpt"))):
            base = os.path.basename(f).split("_")[0]
            if base not in MODULES:
                continue
            try:
                d = pd.read_sas(f, format="xport")
            except Exception:
                continue
            d.columns = [c.decode() if isinstance(c, bytes) else c
                         for c in d.columns]
            keep = ["SEQN"] + [c for c in ENDPOINTS if c in d.columns]
            if "SEQN" in d.columns and len(keep) > 1:
                parts.append(d[keep])
        if not parts:
            continue
        m = parts[0]
        for p in parts[1:]:
            dup = [c for c in p.columns if c in m.columns and c != "SEQN"]
            p = p.drop(columns=dup)
            if p.shape[1] > 1:
                m = m.merge(p, on="SEQN", how="outer")
        m["cycle"] = cyc
        frames.append(m)
        log(f"  {cyc}: {len(m)} 行, {m.shape[1]-2} 个终点变量")
    return pd.concat(frames, ignore_index=True, sort=False)


def add_mortality(df):
    """合并 LMF 死亡终点（二分类：随访期内是否死亡）。"""
    if not os.path.exists(LMF):
        log("  ⚠️ 未找到 LMF，跳过死亡终点")
        return df, []
    lmf = pd.read_parquet(LMF)
    cols = ["SEQN", "cycle", "MORTSTAT", "UCOD_LEADING"]
    m = df.merge(lmf[cols], on=["SEQN", "cycle"], how="left")
    made = []
    elig = m["MORTSTAT"].notna()
    if elig.sum() < 100:
        return df, []
    m["DEATH_ALL"] = np.where(elig, (m["MORTSTAT"] == 1).astype(float), np.nan)
    made.append("DEATH_ALL")
    # 死因分类：001 心脏病 + 005 脑血管 -> 心血管死亡
    ucod = m["UCOD_LEADING"].fillna("")
    cvd = ucod.isin(["001", "005"])
    can = ucod.eq("002")
    oth = (~cvd) & (~can) & (m["MORTSTAT"] == 1)
    m["DEATH_CVD"] = np.where(elig, cvd.astype(float), np.nan)
    m["DEATH_CANCER"] = np.where(elig, can.astype(float), np.nan)
    m["DEATH_OTHER"] = np.where(elig, oth.astype(float), np.nan)
    made += ["DEATH_CVD", "DEATH_CANCER", "DEATH_OTHER"]
    for c in made:
        n = int((m[c] == 1).sum())
        log(f"  死亡终点 {c}: 阳性 {n}")
    return m.drop(columns=["MORTSTAT", "UCOD_LEADING"]), made


def main():
    log("=== 提取问卷终点 ===")
    df = load_questionnaire()
    log(f"合并后: {df.shape}")

    for v in ENDPOINTS:
        if v in df.columns:
            df[v] = recode(df[v])

    log("=== 合并死亡终点 ===")
    df, death_vars = add_mortality(df)

    names = dict(ENDPOINTS)
    names.update({"DEATH_ALL": "全因死亡", "DEATH_CVD": "心血管死亡",
                  "DEATH_CANCER": "癌症死亡", "DEATH_OTHER": "其他原因死亡"})

    rows = []
    for v, nm in names.items():
        if v not in df.columns:
            rows.append({"variable": v, "name": nm, "cycles": 0,
                         "n_labeled": 0, "positive": 0, "prevalence": np.nan,
                         "usable": False, "tier": "缺失", "reason": "变量不存在"})
            continue
        s = df[v]
        n_lab = int(s.notna().sum())
        pos = int((s == 1).sum())
        cov = int(df.groupby("cycle")[v].apply(lambda x: x.notna().any()).sum())
        prev = pos / n_lab if n_lab else np.nan
        if v in SECONDARY:
            usable, tier = (cov >= 2 and pos >= 100), "次级"
            reason = "" if usable else f"周期{cov}/阳性{pos}"
        else:
            usable, tier = (cov >= 3 and pos >= 100), "主"
            reason = "" if usable else f"周期{cov}/阳性{pos}"
        rows.append({"variable": v, "name": nm, "cycles": cov,
                     "n_labeled": n_lab, "positive": pos,
                     "prevalence": round(prev, 4) if n_lab else np.nan,
                     "usable": usable, "tier": tier, "reason": reason})

    inv = pd.DataFrame(rows).sort_values(["usable", "positive"],
                                         ascending=[False, False])
    inv.to_csv(os.path.join(REPORTS, "endpoints_inventory.csv"),
               index=False, encoding="utf-8-sig")

    ok = inv[inv.usable]
    log(f"可用终点: {len(ok)} / {len(names)}")
    for r in ok.itertuples():
        log(f"  [{r.tier}] {r.variable:<12} {r.name:<10} 周期{r.cycles} "
            f"阳性{r.positive:>5} 患病率{r.prevalence*100:>5.1f}%")
    bad = inv[~inv.usable]
    if len(bad):
        log("排除:")
        for r in bad.itertuples():
            log(f"  {r.variable:<12} {r.name:<10} {r.reason}")

    keep = ["SEQN", "cycle"] + ok["variable"].tolist()
    out = df[keep].copy()
    out.to_parquet(OUT_PARQUET, index=False)
    log(f"已保存 -> {OUT_PARQUET}  {out.shape}")
    log(f"清单 -> reports/endpoints_inventory.csv")
    log(f"主终点 {len(ok[ok.tier=='主'])} 个，次级 {len(ok[ok.tier=='次级'])} 个")


if __name__ == "__main__":
    main()
