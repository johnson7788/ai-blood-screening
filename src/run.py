"""论文1 复现主流程（修正版）。

流程（每个病种独立执行）：
  1. 构建训练队列（2013-2016 完整案例）
  2. 算一次 8 基学习器的 OOF 预测矩阵
  3. 穷举 255 种基学习器子集，按 CV AUC 选最优
  4. 内部验证（2017-2018）与外部时间验证（2021-2023）
  5. 递归特征消除到 12 特征，**用同样的无泄漏 CV 评估**
  6. 标签泄漏对照实验（糖尿病/肾病）

评估纪律
--------
所有「训练集」指标一律用**折外预测**得到，绝不用模型在自身训练数据上的
预测。否则会出现「RFE 后 AUC 从 0.83 涨到 0.92」这类假象。
"""
import json
import os
import sys
import time
import warnings

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_val_predict

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
warnings.filterwarnings("ignore")

from config import (EXTERNAL_CYCLES, INTERNAL_CYCLES, RANDOM_STATE, REPORTS,
                    TRAIN_CYCLES)
from data import build_cohort, describe_cohort, leakage_table, load_wide
from metrics import binary_metrics
from models import (StackingEnsemble, cv_meta_auc, make_base_learners,
                    oof_matrix, oof_meta_probability, rfe_by_importance,
                    search_subsets)
from variables import DISEASES


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def run_disease(df, key, feature_set="lab_demo", n_rfe=12, do_rfe=True,
                ignore_leakage=False):
    info = DISEASES[key]
    out = {"disease": key, "name": info["name"], "label_var": info["var"],
           "feature_set": feature_set, "ignore_leakage": ignore_leakage}
    since = time.time()

    def cohort(cycles):
        return build_cohort(df, key, feature_set=feature_set, cycles=cycles,
                            ignore_leakage=ignore_leakage)

    tr = cohort(TRAIN_CYCLES)
    X = tr["X"].to_numpy(dtype=float)
    y = tr["y"]
    w = tr["w"]
    names = tr["var_names"]
    out.update(n_train=tr["n"], n_features=len(names),
               prevalence=round(tr["prevalence"], 4),
               weighted_prevalence=round(tr["weighted_prevalence"], 4),
               features=names)
    log(f"  {info['name']}: 训练 n={tr['n']} 特征={len(names)} "
        f"患病率={tr['prevalence']:.3f} 加权={tr['weighted_prevalence']:.3f}")

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    learners = make_base_learners(RANDOM_STATE)

    # --- 1) 单模型基线（折外） ---
    base_scores = {}
    for nm, est in learners.items():
        p = cross_val_predict(est, X, y, cv=cv, method="predict_proba",
                              n_jobs=1)[:, 1]
        base_scores[nm] = float(roc_auc_score(y, p))
    out["base_learner_cv_auc"] = {k: round(v, 4) for k, v in base_scores.items()}
    best_single = max(base_scores, key=base_scores.get)
    out["best_single_learner"] = best_single
    out["best_single_auc"] = round(base_scores[best_single], 4)
    log(f"    最佳单模型 {best_single} AUC={base_scores[best_single]:.4f}")

    # --- 2) OOF 矩阵 + 255 子集穷举 ---
    O, lnames = oof_matrix(X, y, cv, learners, RANDOM_STATE)
    log("    穷举 255 种子集 ...")
    subs = search_subsets(O, y, lnames, cv, RANDOM_STATE)
    out["subset_top10"] = [
        {"subset": s["subset"], "n": s["n_learners"],
         "cv_auc": round(s["auc_mean"], 4), "cv_std": round(s["auc_std"], 4)}
        for s in subs[:10]
    ]
    best_subset = subs[0]["subset"]
    out["best_subset"] = best_subset
    out["stacking_cv_auc"] = round(subs[0]["auc_mean"], 4)
    out["stacking_cv_std"] = round(subs[0]["auc_std"], 4)
    log(f"    最优子集 {'+'.join(best_subset)} CV AUC={subs[0]['auc_mean']:.4f} "
        f"(单模型 {base_scores[best_single]:.4f})")

    # --- 3) 各队列评估 ---
    out["validation"] = {}

    # 训练集：元学习器折外概率
    p_tr = oof_meta_probability(O, y, best_subset, lnames, cv, RANDOM_STATE)
    out["validation"]["train_cv"] = _round_metrics(binary_metrics(y, p_tr, w))

    # 最终模型（供外部队列使用）
    model = StackingEnsemble(subset=best_subset, random_state=RANDOM_STATE)
    model.fit(X, y, cv=cv)

    for tag, cycles in [("internal_2017_2018", INTERNAL_CYCLES),
                        ("external_2021_2023", EXTERNAL_CYCLES)]:
        co = cohort(cycles)
        shared = [c for c in names if c in co["var_names"]]
        if len(shared) < len(names):
            idx = [names.index(c) for c in shared]
            m2 = StackingEnsemble(subset=best_subset, random_state=RANDOM_STATE)
            m2.fit(X[:, idx], y, cv=cv)
            p = m2.predict_proba(co["X"][shared].to_numpy(dtype=float))[:, 1]
        else:
            p = model.predict_proba(co["X"][names].to_numpy(dtype=float))[:, 1]
        out[f"n_{tag}"] = co["n"]
        out["validation"][tag] = _round_metrics(
            binary_metrics(co["y"], p, co["w"]))

    tr_auc = out["validation"]["train_cv"]["auc"]
    for tag in ["internal_2017_2018", "external_2021_2023"]:
        out["validation"][tag]["auc_drop_vs_train"] = round(
            out["validation"][tag]["auc"] - tr_auc, 4)
    log(f"    训练CV={tr_auc:.4f} | 内部="
        f"{out['validation']['internal_2017_2018']['auc']:.4f} | 外部="
        f"{out['validation']['external_2021_2023']['auc']:.4f} "
        f"(掉幅 {out['validation']['external_2021_2023']['auc_drop_vs_train']:+.4f})")

    # --- 4) 递归特征消除（用无泄漏 CV 评估） ---
    if do_rfe:
        log(f"    递归特征消除 -> {n_rfe} 特征 ...")
        kept, hist = rfe_by_importance(X, y, names, cv, n_features=n_rfe,
                                       random_state=RANDOM_STATE)
        out["rfe_kept"] = kept
        out["rfe_history"] = hist
        idx = [names.index(k) for k in kept]
        Xr = X[:, idx]
        Or, lnr = oof_matrix(Xr, y, cv, learners, RANDOM_STATE)
        subr = search_subsets(Or, y, lnr, cv, RANDOM_STATE)
        out["rfe_best_subset"] = subr[0]["subset"]
        out["rfe_cv_auc"] = round(subr[0]["auc_mean"], 4)
        out["rfe_cv_std"] = round(subr[0]["auc_std"], 4)
        out["rfe_auc_delta"] = round(subr[0]["auc_mean"] - subs[0]["auc_mean"], 4)

        # RFE 模型的外部队列评估
        mr = StackingEnsemble(subset=subr[0]["subset"], random_state=RANDOM_STATE)
        mr.fit(Xr, y, cv=cv)
        out["rfe_validation"] = {}
        for tag, cycles in [("internal_2017_2018", INTERNAL_CYCLES),
                            ("external_2021_2023", EXTERNAL_CYCLES)]:
            co = cohort(cycles)
            shared = [c for c in kept if c in co["var_names"]]
            if len(shared) == len(kept):
                p = mr.predict_proba(co["X"][kept].to_numpy(dtype=float))[:, 1]
                out["rfe_validation"][tag] = round(
                    binary_metrics(co["y"], p, co["w"])["auc"], 4)
        log(f"    RFE 后 CV AUC={subr[0]['auc_mean']:.4f} "
            f"(全特征 {subs[0]['auc_mean']:.4f}, Δ={out['rfe_auc_delta']:+.4f})")

    out["elapsed_sec"] = round(time.time() - since, 1)
    return out


def _round_metrics(m):
    return {k: (round(v, 4) if isinstance(v, float) else v) for k, v in m.items()}


def main():
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    which = args[0].split(",") if args else list(DISEASES.keys())
    do_rfe = "--no-rfe" not in sys.argv
    df = load_wide()
    log(f"宽表载入: {df.shape}")

    leakage_table().to_csv(os.path.join(REPORTS, "leakage_table.csv"),
                           index=False, encoding="utf-8-sig")
    log("已导出「标签—禁用变量」对照表 -> reports/leakage_table.csv")

    ds = describe_cohort(df, os.path.join(REPORTS, "descriptive_stats.csv"))
    log("已导出描述统计 -> reports/descriptive_stats.csv")
    print(ds.to_string(index=False))

    results = []
    for key in which:
        log(f"=== {DISEASES[key]['name']} ({key}) ===")
        r = run_disease(df, key, do_rfe=do_rfe)
        if key in ("diabetes", "ckd"):
            log("    [对照] 不剔除泄漏变量重跑 ...")
            r2 = run_disease(df, key, do_rfe=False, ignore_leakage=True)
            r["leakage_control"] = {
                "n_features": r2["n_features"],
                "cv_auc_with_leakage": r2["stacking_cv_auc"],
                "cv_auc_without_leakage": r["stacking_cv_auc"],
                "inflation": round(r2["stacking_cv_auc"] - r["stacking_cv_auc"], 4),
                "external_with_leakage":
                    r2["validation"]["external_2021_2023"]["auc"],
                "external_without_leakage":
                    r["validation"]["external_2021_2023"]["auc"],
                "leaked_features": [f for f in r2["features"]
                                    if f not in r["features"]],
            }
            log(f"    [对照] 含泄漏 CV AUC={r2['stacking_cv_auc']:.4f} "
                f"vs 剔除后 {r['stacking_cv_auc']:.4f} "
                f"(虚高 {r['leakage_control']['inflation']:+.4f})")
        results.append(r)
        # 增量保存，防止中断丢结果
        with open(os.path.join(REPORTS, "results.json"), "w",
                  encoding="utf-8") as f:
            json.dump(results, f, ensure_ascii=False, indent=2)

    rows = []
    for r in results:
        v, e = r["validation"], r["validation"]["external_2021_2023"]
        rows.append({
            "病种": r["name"], "训练n": r["n_train"],
            "患病率": r["prevalence"], "特征数": r["n_features"],
            "最佳单模型": r["best_single_learner"], "单模型AUC": r["best_single_auc"],
            "堆叠CV": r["stacking_cv_auc"],
            "内部": v["internal_2017_2018"]["auc"],
            "外部": e["auc"], "外部掉幅": e["auc_drop_vs_train"],
            "外部加权": e["auc_weighted"],
            "RFE_CV": r.get("rfe_cv_auc"), "RFE_Δ": r.get("rfe_auc_delta"),
            "最优子集": "+".join(r["best_subset"]),
        })
    summary = pd.DataFrame(rows)
    summary.to_csv(os.path.join(REPORTS, "summary.csv"),
                   index=False, encoding="utf-8-sig")
    print()
    print(summary.to_string(index=False))
    log("汇总已保存 -> reports/summary.csv")


if __name__ == "__main__":
    main()
