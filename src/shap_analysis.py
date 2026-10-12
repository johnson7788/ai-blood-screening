"""SHAP 可解释性分析。

对每个病种：
  1. 用 XGBoost 基学习器做 TreeSHAP，得到每个特征的全局重要性（平均 |SHAP|）
  2. 导出堆叠集成元学习器（LR）的系数，看哪些基学习器真正在起作用
  3. 输出各特征对阳性的正负贡献方向
"""
import json
import os
import sys
import warnings

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
warnings.filterwarnings("ignore")

import shap
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold

from config import RANDOM_STATE, REPORTS, TRAIN_CYCLES
from data import build_cohort, load_wide
from models import make_base_learners, oof_matrix
from variables import DISEASES, VAR_LABELS


def analyse(df, key, top_n=15):
    tr = build_cohort(df, key, feature_set="lab_demo", cycles=TRAIN_CYCLES)
    X = tr["X"].to_numpy(dtype=float)
    y = tr["y"]
    names = tr["var_names"]

    learners = make_base_learners(RANDOM_STATE)
    xgb = learners["XGBoost"]
    xgb.fit(X, y)
    expl = shap.TreeExplainer(xgb)
    sv = expl.shap_values(X)
    sv = np.asarray(sv)
    if sv.ndim == 3:
        sv = sv[:, :, 1]

    mean_abs = np.abs(sv).mean(axis=0)
    # 阳性样本上的平均贡献方向
    direction = sv[y == 1].mean(axis=0) if (y == 1).any() else np.zeros_like(mean_abs)

    imp = pd.DataFrame({
        "variable": names,
        "feature": [VAR_LABELS.get(v, v) for v in names],
        "mean_abs_shap": mean_abs,
        "direction_in_cases": direction,
    }).sort_values("mean_abs_shap", ascending=False).reset_index(drop=True)
    imp["rank"] = np.arange(1, len(imp) + 1)
    imp["mean_abs_shap"] = imp["mean_abs_shap"].round(5)
    imp["direction_in_cases"] = imp["direction_in_cases"].round(5)

    # 元学习器系数
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    O, lnames = oof_matrix(X, y, cv, learners, RANDOM_STATE)
    meta = LogisticRegression(max_iter=2000, random_state=RANDOM_STATE)
    meta.fit(O, y)
    meta_df = pd.DataFrame({
        "base_learner": lnames,
        "meta_coef": meta.coef_[0].round(4),
    }).sort_values("meta_coef", key=abs, ascending=False).reset_index(drop=True)

    return imp.head(top_n), meta_df, imp


def main():
    df = load_wide()
    keys = sys.argv[1].split(",") if len(sys.argv) > 1 else list(DISEASES.keys())
    all_imp = []
    summary = {}
    for key in keys:
        name = DISEASES[key]["name"]
        print(f"[SHAP] {name} ...", flush=True)
        top, meta_df, full = analyse(df, key)
        full.to_csv(os.path.join(REPORTS, f"shap_{key}.csv"),
                    index=False, encoding="utf-8-sig")
        for r in top.itertuples():
            all_imp.append({"disease": name, "rank": r.rank,
                            "feature": r.feature, "variable": r.variable,
                            "mean_abs_shap": r.mean_abs_shap,
                            "direction": r.direction_in_cases})
        summary[key] = {
            "name": name,
            "top5": [{"feature": r.feature, "shap": r.mean_abs_shap}
                     for r in top.head(5).itertuples()],
            "meta_coef": meta_df.to_dict("records"),
        }
        print(f"  Top5: " + ", ".join(
            f"{r.feature}({r.mean_abs_shap:.4f})" for r in top.head(5).itertuples()))

    pd.DataFrame(all_imp).to_csv(os.path.join(REPORTS, "shap_top_features.csv"),
                                 index=False, encoding="utf-8-sig")
    with open(os.path.join(REPORTS, "shap_summary.json"), "w",
              encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print("已保存 -> reports/shap_summary.json, shap_top_features.csv")


if __name__ == "__main__":
    main()
