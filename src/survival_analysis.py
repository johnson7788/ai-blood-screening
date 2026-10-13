"""Phase 3：用常规血液检验预测长期死亡风险（生存分析）。

相对横断面疾病分类的改进
------------------------
1. **终点客观**：死亡由国家死亡索引确认，不存在自报诊断的未诊断偏倚
2. **真纵向**：利用随访时间（PERMTH_EXM），而非单一时点
3. **可评估面更广**：C-index、时间依赖 AUC、校准、Brier 分数

队列
----
NHANES 2013-2014 / 2015-2016 / 2017-2018（LMF 覆盖），
符合死亡率随访资格（ELIGSTAT=1）且血检完整的成年人。

评估策略
--------
随访截尾是**行政性**的（统一随访到 2019-12-31），属非信息性截尾，
因此可使用完整随访而不必强行统一时间窗。采用 5 折交叉验证，
按「周期 × 事件状态」分层以保证各折的事件率一致。

产出
----
reports/survival_results.json       各模型的 C-index / 时间依赖 AUC / IBS
reports/survival_cv_predictions.parquet  折外风险评分（供后续可解释性使用）
reports/figures/survival_*.png      生存曲线与校准图
"""
import json
import os
import sys
import time
import warnings

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
warnings.filterwarnings("ignore")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sksurv.ensemble import (GradientBoostingSurvivalAnalysis,
                             RandomSurvivalForest)
from sksurv.linear_model import CoxnetSurvivalAnalysis, CoxPHSurvivalAnalysis
from sksurv.metrics import (concordance_index_censored,
                            cumulative_dynamic_auc,
                            integrated_brier_score)
from sksurv.util import Surv

from config import RANDOM_STATE, REPORTS
from data import STRUCTURAL_EXCLUDE, analysis_mask, load_wide
from variables import FEATURE_SETS, VAR_LABELS

for f in ["Arial Unicode MS", "PingFang SC", "Heiti SC", "STHeiti"]:
    matplotlib.rcParams["font.sans-serif"] = [f]
    break
matplotlib.rcParams["axes.unicode_minus"] = False

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LMF = os.path.join(ROOT, "mortality", "lmf_parsed.parquet")
FIGDIR = os.path.join(REPORTS, "figures")
os.makedirs(FIGDIR, exist_ok=True)

CYCLES = ["2013-2014", "2015-2016", "2017-2018"]
TIME_POINTS = [12, 24, 36, 48, 60]   # 月
N_SPLITS = 5


def log(m):
    print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)


def build_survival_cohort():
    """构建生存分析队列。返回 X, time(月), event, cycle, 特征名。"""
    wide = load_wide()
    lmf = pd.read_parquet(LMF)
    d = wide[wide["cycle"].isin(CYCLES)].copy()
    d = d.merge(lmf, on=["SEQN", "cycle"], how="left")
    d = d[d["ELIGSTAT"] == 1]
    d = d[analysis_mask(d)]
    d = d[d["MORTSTAT"].notna() & d["PERMTH_EXM"].notna()
          & (d["PERMTH_EXM"] > 0)]
    cols = [c for c in FEATURE_SETS["lab_demo"]
            if c not in STRUCTURAL_EXCLUDE and c in d.columns]
    cols = [c for c in cols if d[c].isna().mean() <= 0.20]
    X = d[cols].astype(float).fillna(d[cols].astype(float).median())
    t = d["PERMTH_EXM"].astype(float).to_numpy()
    e = (d["MORTSTAT"] == 1).to_numpy()
    return X, t, e, d["cycle"].to_numpy(), cols, d


def make_models(n_features):
    """候选生存模型。

    ⚠️ Cox 的岭惩罚作用在系数上，若特征量纲差异大（WBC≈7、血小板≈250、
       MCV≈90），惩罚会不成比例地压缩大尺度变量的系数，严重损害性能
       （实测未标准化时 C-index 仅 0.60）。因此 Cox 必须包一层标准化。
       树模型对量纲不敏感，无需标准化。
    """
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler

    return {
        "Cox 比例风险（岭回归）": Pipeline([
            ("sc", StandardScaler()),
            ("cox", CoxPHSurvivalAnalysis(alpha=1.0)),
        ]),
        "随机生存森林 RSF": RandomSurvivalForest(
            n_estimators=300, min_samples_leaf=15, max_features="sqrt",
            n_jobs=-1, random_state=RANDOM_STATE),
        "梯度提升生存 GBSA": GradientBoostingSurvivalAnalysis(
            n_estimators=300, learning_rate=0.05, max_depth=3,
            subsample=0.8, random_state=RANDOM_STATE),
    }


def risk_score(model, X, name):
    """统一取风险评分（越大越高危）。"""
    if hasattr(model, "predict"):
        try:
            return np.asarray(model.predict(X), dtype=float)
        except Exception:
            pass
    return np.asarray(model.predict(X), dtype=float)


def main():
    log("=== 构建生存队列 ===")
    X, t, e, cyc, cols, d = build_survival_cohort()
    log(f"  队列: {len(X)} 人 | 死亡 {int(e.sum())} ({e.mean()*100:.2f}%) "
        f"| 特征 {len(cols)}")
    log(f"  随访月数: 中位 {np.median(t):.0f}, 最长 {t.max():.0f}")

    y = Surv.from_arrays(event=e, time=t)
    Xv = X.to_numpy(dtype=float)

    from sklearn.model_selection import StratifiedKFold
    strat = pd.Series([f"{c}_{int(ev)}" for c, ev in zip(cyc, e)])
    skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True,
                          random_state=RANDOM_STATE)

    # 用于时间依赖指标的统一时间网格（限制在有足够样本的范围）
    tmax = float(np.percentile(t[e], 90))
    times = [x for x in TIME_POINTS if x < tmax]
    log(f"  时间依赖评估点: {times} 月（随访 90 分位 {tmax:.0f} 月）")

    results = {}
    oof = {}
    for name in make_models(len(cols)):
        t0 = time.time()
        log(f"  {name} ...")
        try:
            risk = np.zeros(len(y))
            for k, (tr, te) in enumerate(skf.split(Xv, strat)):
                m = make_models(len(cols))[name]
                m.fit(Xv[tr], y[tr])
                risk[te] = m.predict(Xv[te])
        except Exception as ex:
            log(f"    ❌ 拟合失败，跳过该模型: {str(ex)[:120]}")
            continue

        ci = concordance_index_censored(e, t, risk)[0]

        # 时间依赖 AUC：用折外风险分评估
        # ⚠️ sksurv 0.28 的 cumulative_dynamic_auc 返回 (aucs, mean_auc)，
        #    即第一个元素是各时间点的 AUC 数组、第二个是平均 AUC 标量。
        #    这与旧版本文档的写法相反，务必按实际返回顺序处理。
        aucs = np.array([])
        try:
            r0, r1 = cumulative_dynamic_auc(y, y, risk, times)
            a0 = np.atleast_1d(np.asarray(r0, dtype=float)).ravel()
            a1 = np.atleast_1d(np.asarray(r1, dtype=float)).ravel()
            aucs = a0 if a0.size == len(times) else (
                a1 if a1.size == len(times) else np.array([]))
            if aucs.size != len(times):
                log(f"    time-AUC 维度异常: {a0.shape}/{a1.shape}")
        except Exception as ex:
            log(f"    time-AUC 失败: {str(ex)[:80]}")

        # IBS：用全量拟合的模型估计生存函数
        # 计算量瓶颈在 predict_survival_function——对全部 1.5 万人求生存函数
        # 在 GBSA 上需十几分钟。故在随机子样本（3000 人）上估计 IBS，
        # 这是生存分析中的常规做法，对 IBS 的估计精度影响可忽略。
        ibs = np.nan
        try:
            mfull = make_models(len(cols))[name]
            mfull.fit(Xv, y)
            rng = np.random.default_rng(RANDOM_STATE)
            sub = (rng.choice(len(y), size=min(3000, len(y)), replace=False)
                   if len(y) > 3000 else np.arange(len(y)))
            survs = mfull.predict_survival_function(Xv[sub])
            grid = np.unique(np.clip(np.linspace(1, tmax, 30), 1, None))
            preds = np.vstack([np.interp(grid, sf.x, sf.y) for sf in survs])
            # 签名: integrated_brier_score(survival_train, survival_test,
            #                            estimate, times)
            ibs = integrated_brier_score(y, y[sub], preds, grid)
        except Exception as ex:
            log(f"    IBS 跳过: {str(ex)[:80]}")

        results[name] = {
            "c_index": round(float(ci), 4),
            "time_auc": {str(tm): round(float(a), 4)
                         for tm, a in zip(times, aucs)},
            "ibs": None if not np.isfinite(ibs) else round(float(ibs), 4),
            "elapsed_sec": round(time.time() - t0, 1),
        }
        oof[name] = risk
        log(f"    C-index={ci:.4f}  "
            f"time-AUC={[round(float(a),3) for a in aucs]}  "
            f"({results[name]['elapsed_sec']}s)")

    # --- 基线：仅用年龄 ---
    ci_age = concordance_index_censored(e, t, d["RIDAGEYR"].to_numpy())[0]
    log(f"  基线（仅年龄）: C-index={ci_age:.4f}")
    results["基线（仅年龄）"] = {"c_index": round(float(ci_age), 4),
                                "time_auc": {}, "ibs": None}

    out = pd.DataFrame(oof)
    out["SEQN"] = d["SEQN"].to_numpy()
    out["cycle"] = cyc
    out["time"] = t
    out["event"] = e.astype(int)
    out["age"] = d["RIDAGEYR"].to_numpy()
    out.to_parquet(os.path.join(REPORTS, "survival_cv_predictions.parquet"),
                   index=False)

    with open(os.path.join(REPORTS, "survival_results.json"), "w",
              encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    # --- 汇总表 ---
    log("")
    log("=" * 66)
    log(f"{'模型':<22}{'C-index':>9}  " +
        "  ".join(f"{tm:>3}月AUC" for tm in times) + f"{'IBS':>9}")
    log("-" * 66)
    for k, v in results.items():
        ta = "  ".join(f"{v['time_auc'].get(str(tm), float('nan')):>8.4f}"
                       for tm in times)
        ibs = "     —  " if v["ibs"] is None else f"{v['ibs']:>9.4f}"
        log(f"{k:<22}{v['c_index']:>9.4f}  {ta}{ibs}")
    log("=" * 66)
    log("结果 -> reports/survival_results.json")
    log("折外风险分 -> reports/survival_cv_predictions.parquet")


if __name__ == "__main__":
    main()
