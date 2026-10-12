"""训练并持久化最终模型，供开源发布与推理复用。

每个病种保存两个模型：
  1. 全特征堆叠集成（33 项特征，性能上限）
  2. RFE 精简模型（12 项特征，推荐上线版本）
以及一份 metadata.json（特征列表、阈值、指标），保证可脱离原始代码复现推理。
"""
import json
import os
import sys
import time
import warnings

import joblib
import numpy as np
from sklearn.model_selection import StratifiedKFold

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
warnings.filterwarnings("ignore")

from config import ARTIFACTS, RANDOM_STATE, REPORTS, TRAIN_CYCLES
from data import build_cohort, load_wide
from metrics import binary_metrics, best_youden_threshold
from models import StackingEnsemble
from variables import DISEASES, VAR_LABELS

MODELDIR = os.path.join(ARTIFACTS, "models")
os.makedirs(MODELDIR, exist_ok=True)


def log(m):
    print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)


def main():
    with open(os.path.join(REPORTS, "results.json"), encoding="utf-8") as f:
        results = json.load(f)
    df = load_wide()
    manifest = []

    for res in results:
        key = res["disease"]
        name = res["name"]
        log(f"=== {name} ===")
        tr = build_cohort(df, key, feature_set="lab_demo", cycles=TRAIN_CYCLES)
        X = tr["X"].to_numpy(dtype=float)
        y = tr["y"]
        names = tr["var_names"]
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)

        entry = {"disease": key, "name": name, "models": {}}

        # 1) 全特征模型
        m_full = StackingEnsemble(subset=res["best_subset"],
                                  random_state=RANDOM_STATE)
        m_full.fit(X, y, cv=cv)
        p = m_full.predict_proba(X)[:, 1]
        thr = best_youden_threshold(y, p)
        f_full = os.path.join(MODELDIR, f"{key}_full.joblib")
        joblib.dump(m_full, f_full, compress=3)
        entry["models"]["full"] = {
            "file": os.path.basename(f_full),
            "features": names,
            "n_features": len(names),
            "subset": res["best_subset"],
            "threshold": round(float(thr), 6),
            "cv_auc": res["stacking_cv_auc"],
            "external_auc": res["validation"]["external_2021_2023"]["auc"],
            "size_mb": round(os.path.getsize(f_full) / 1e6, 2),
        }
        log(f"  全特征模型 {os.path.getsize(f_full) / 1e6:.1f} MB "
            f"({len(names)} 特征, {len(res['best_subset'])} 基学习器)")

        # 2) RFE 精简模型
        if res.get("rfe_kept"):
            kept = res["rfe_kept"]
            idx = [names.index(k) for k in kept]
            m_rfe = StackingEnsemble(subset=res.get("rfe_best_subset",
                                                    res["best_subset"]),
                                     random_state=RANDOM_STATE)
            m_rfe.fit(X[:, idx], y, cv=cv)
            pr = m_rfe.predict_proba(X[:, idx])[:, 1]
            f_rfe = os.path.join(MODELDIR, f"{key}_rfe12.joblib")
            joblib.dump(m_rfe, f_rfe, compress=3)
            entry["models"]["rfe12"] = {
                "file": os.path.basename(f_rfe),
                "features": kept,
                "n_features": len(kept),
                "subset": res.get("rfe_best_subset", res["best_subset"]),
                "threshold": round(float(best_youden_threshold(y, pr)), 6),
                "cv_auc": res.get("rfe_cv_auc"),
                "external_auc": res.get("rfe_validation", {}).get(
                    "external_2021_2023"),
                "size_mb": round(os.path.getsize(f_rfe) / 1e6, 2),
            }
            log(f"  RFE 模型   {os.path.getsize(f_rfe) / 1e6:.1f} MB "
                f"({len(kept)} 特征)")

        entry["label_var"] = res["label_var"]
        entry["n_train"] = res["n_train"]
        entry["prevalence"] = res["prevalence"]
        entry["feature_labels"] = {c: VAR_LABELS.get(c, c) for c in names}
        manifest.append(entry)

    out = {
        "description": "NHANES 常规血检多病种筛查 — 堆叠集成模型",
        "random_state": RANDOM_STATE,
        "train_cycles": TRAIN_CYCLES,
        "internal_cycles": ["2017-2018"],
        "external_cycles": ["2021-2023"],
        "usage": ("import joblib, pandas as pd; "
                  "m = joblib.load('<file>.joblib'); "
                  "m.predict_proba(X[features])[:, 1] >= threshold"),
        "models": manifest,
    }
    with open(os.path.join(MODELDIR, "metadata.json"), "w",
              encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)

    total = sum(f["size_mb"] for e in manifest for f in e["models"].values())
    log(f"模型总计约 {total:.1f} MB，清单 -> {MODELDIR}/metadata.json")

    # 自检：重新加载并验证推理一致
    log("自检：重新加载模型验证推理 ...")
    e0 = manifest[0]
    m = joblib.load(os.path.join(MODELDIR, e0["models"]["full"]["file"]))
    tr = build_cohort(df, e0["disease"], feature_set="lab_demo",
                      cycles=TRAIN_CYCLES)
    p1 = m.predict_proba(tr["X"].to_numpy(float))[:, 1]
    log(f"  {e0['name']}: 推理正常，输出范围 "
        f"[{p1.min():.4f}, {p1.max():.4f}]")


if __name__ == "__main__":
    main()
