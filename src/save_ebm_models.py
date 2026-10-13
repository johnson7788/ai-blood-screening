"""训练并持久化 EBM 可解释模型。

EBM 的价值在于完全可加性：预测 = 截距 + Σ 各特征形状函数 + Σ 成对交互项，
因此每个指标的贡献可直接画成曲线，且与模型真实决策路径严格一致。

产出 artifacts/models/ebm_<disease>.joblib 与更新后的 metadata.json。
"""
import json
import os
import sys
import time
import warnings

import joblib
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
warnings.filterwarnings("ignore")

from config import ARTIFACTS, RANDOM_STATE, REPORTS, TRAIN_CYCLES
from data import build_cohort, load_wide
from metrics import binary_metrics, best_youden_threshold
from variables import DISEASES, VAR_LABELS

MODELDIR = os.path.join(ARTIFACTS, "models")
os.makedirs(MODELDIR, exist_ok=True)


def log(m):
    print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)


def main():
    from interpret.glassbox import ExplainableBoostingClassifier

    df = load_wide()
    # 读取已完成的对比结果，取外部 AUC 用于元数据
    ext = {}
    for p in (os.path.join(REPORTS, "compare_models.json"),
              os.path.join(REPORTS, "compare_models_sub5000.json")):
        if os.path.exists(p):
            for r in json.load(open(p, encoding="utf-8")):
                if r["model"] == "ebm" and not p.endswith("sub5000.json"):
                    ext[r["disease"]] = r["validation"]["external_2021_2023"]["auc"]

    entries = []
    for key, info in DISEASES.items():
        tr = build_cohort(df, key, feature_set="lab_demo", cycles=TRAIN_CYCLES)
        X = tr["X"]
        y = tr["y"]
        names = list(X.columns)
        m = ExplainableBoostingClassifier(
            random_state=RANDOM_STATE, max_bins=256, interactions=10,
            outer_bags=8, learning_rate=0.03, n_jobs=-1)
        m.fit(X, y)

        p = m.predict_proba(X)[:, 1]
        # 阈值必须用**折外预测**标定：样本内预测会因过拟合而偏低，
        # 导致上线后阳性率虚高（实测糖尿病会从 11.5% 虚高到 22%）。
        from sklearn.model_selection import (StratifiedKFold,
                                             cross_val_predict)
        cv = StratifiedKFold(n_splits=5, shuffle=True,
                             random_state=RANDOM_STATE)
        p_oof = cross_val_predict(m.__class__(**m.get_params()), X, y, cv=cv,
                                  method="predict_proba", n_jobs=1)[:, 1]
        thr = best_youden_threshold(y, p_oof)
        metric_oof = binary_metrics(y, p_oof, tr["w"])
        # 存储样本内拟合的模型（用于上线推理），但指标与阈值取自折外
        f = os.path.join(MODELDIR, f"ebm_{key}.joblib")
        joblib.dump(m, f, compress=3)

        # 特征重要性（含交互项）——EBM 的原生可解释性输出
        imp = np.asarray(m.term_importances())
        terms = list(m.term_names_)
        top = sorted(zip(terms, imp), key=lambda t: -t[1])[:10]

        entries.append({
            "disease": key, "name": info["name"],
            "variant": "ebm",
            "file": os.path.basename(f),
            "n_features": len(names),
            "features": names,
            "feature_labels": {c: VAR_LABELS.get(c, c) for c in names},
            "threshold": round(float(thr), 6),
            "train_cv_auc": round(float(metric_oof["auc"]), 4),
            "external_auc": ext.get(key),
            "size_mb": round(os.path.getsize(f) / 1e6, 2),
            "interpretability": "完全可加性（形状函数可导出）",
            "top_terms": [{"term": t, "feature": VAR_LABELS.get(t, t),
                           "importance": round(float(v), 4)} for t, v in top],
        })
        log(f"  {info['name']}: {os.path.getsize(f)/1e6:.1f} MB | "
            f"训练CV AUC={metric_oof['auc']:.4f} | 外部AUC={ext.get(key)} | "
            f"阈值={thr:.4f}")

    mp = os.path.join(MODELDIR, "metadata.json")
    meta = json.load(open(mp, encoding="utf-8")) if os.path.exists(mp) else {
        "description": "NHANES 常规血检多病种筛查模型",
        "random_state": RANDOM_STATE,
        "models": [],
    }
    meta["models"] = [e for e in meta.get("models", [])
                      if not e.get("variant") == "ebm"] 
    # 把 EBM 作为每个病种的 ebm 变体挂到对应条目上
    by_dis = {e["disease"]: e for e in meta["models"]}
    for e in entries:
        d = by_dis.get(e["disease"])
        if d is None:
            meta["models"].append({"disease": e["disease"], "name": e["name"],
                                   "models": {}})
            d = meta["models"][-1]
        d.setdefault("models", {})["ebm"] = {
            "file": e["file"], "features": e["features"],
            "n_features": e["n_features"], "threshold": e["threshold"],
            "cv_auc": e["train_cv_auc"], "external_auc": e["external_auc"],
            "size_mb": e["size_mb"],
            "interpretability": e["interpretability"],
            "top_terms": e["top_terms"],
        }
    meta["model_families"] = meta.get("model_families") or {}
    meta["model_families"]["ebm"] = (
        "Explainable Boosting Machine —— 完全可加性玻璃盒模型，"
        "每个特征的形状函数可导出，适合需要可解释性的临床场景。"
        "加载需要 `interpret` 包。")
    json.dump(meta, open(mp, "w", encoding="utf-8"),
              ensure_ascii=False, indent=2)
    tot = sum(e["size_mb"] for e in entries)
    log(f"EBM 模型合计 {tot:.1f} MB；metadata.json 已更新")


if __name__ == "__main__":
    main()
