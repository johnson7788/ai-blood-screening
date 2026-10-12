"""多分类任务：复现论文1 的双任务设定，并记录在 NHANES 上的可行性边界。

重要方法学发现
--------------
论文1 采用「单标签互斥」多分类，前提是每位患者只有一种眼内疾病（论文明确
说明多重眼病诊断者被排除）。眼病之间天然互斥，该设定成立。

但**系统性疾病共病是常态**，NHANES 训练队列（9,925 人标签齐全）中：
    0 种病 54.8% | 1 种病 26.5% | ≥2 种病 18.7%
各病种总阳性中属于「单一诊断」的比例：
    高血压 53% | 癌症 32% | 糖尿病 23% | 慢性肾病 17%
    中风 13% | 心梗 8% | 冠心病 6%
即冠心病患者中仅 6% 是纯冠心病。若强行套用单标签设定，冠心病只剩 22 例、
心梗 31 例，8 分类不可行。

因此本复现：
  1. 多分类限定为样本量≥250 的类别（健康/高血压/糖尿病/癌症）
  2. 外部队列**固定使用训练集的类别集合**，保证宏平均 AUC 可比
  3. 明确指出：系统性疾病筛查应采用多标签（one-vs-rest）而非多分类，
     本项目的 7 个二分类模型即为多标签方案
"""
import itertools
import json
import os
import sys
import time
import warnings

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_val_predict

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
warnings.filterwarnings("ignore")

from config import (EXTERNAL_CYCLES, INTERNAL_CYCLES, RANDOM_STATE, REPORTS,
                    TRAIN_CYCLES, WEIGHT_VAR)
from data import STRUCTURAL_EXCLUDE, analysis_mask, load_wide, recode_outcome
from models import make_base_learners
from variables import DISEASES, FEATURE_SETS

MIN_CLASS = 250


def log(m):
    print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)


def build_multiclass(df, cycles, keep_classes=None, min_class=MIN_CLASS):
    """返回 X, labels(str 数组), cols, w, 类别计数。

    keep_classes 为 None 时按 min_class 自动确定类别（训练集用）；
    否则严格使用给定类别集合（外部队列用），保证与训练集可比。
    """
    d = df[df["cycle"].isin(cycles)].copy()
    d = d[analysis_mask(d)].copy()

    Y = pd.DataFrame({k: recode_outcome(d[i["var"]]) for k, i in DISEASES.items()},
                     index=d.index).dropna()
    d = d.loc[Y.index]
    Y = Y.loc[d.index]
    n_pos = Y.sum(axis=1)

    # 论文1 单标签设定：多重诊断者排除；0 种病 -> healthy
    labels = pd.Series(np.where(n_pos == 1, Y.idxmax(axis=1), "healthy"),
                       index=d.index)
    counts = labels.value_counts()

    if keep_classes is None:
        keep_classes = [c for c in counts.index if counts[c] >= min_class]
    keep = labels.isin(keep_classes)
    labels = labels[keep]
    d = d.loc[labels.index]

    cols = [c for c in FEATURE_SETS["lab_demo"]
            if c not in STRUCTURAL_EXCLUDE and c in d.columns]
    cols = [c for c in cols if d[c].isna().mean() <= 0.20]
    X = d[cols].astype(float).fillna(d[cols].astype(float).median())
    w = (d[WEIGHT_VAR] / d["cycle"].nunique()).to_numpy()
    return X, labels.to_numpy(), cols, w, counts


def oof_tensor(X, y, cv, learners, n_classes):
    names = list(learners.keys())
    O = np.zeros((len(y), len(names), n_classes))
    for j, nm in enumerate(names):
        p = cross_val_predict(learners[nm], X, y, cv=cv,
                              method="predict_proba", n_jobs=1)
        O[:, j, :p.shape[1]] = p
    return O, names


def macro_auc(y, P, class_names):
    """返回 (宏平均AUC, {类名: AUC})。按类名返回，避免跳过类别时错位。"""
    per = {}
    for ci, c in enumerate(class_names):
        yb = (np.asarray(y) == ci).astype(int)
        if 0 < yb.sum() < len(yb):
            try:
                per[c] = roc_auc_score(yb, P[:, ci])
            except ValueError:
                pass
    return (float(np.mean(list(per.values()))) if per else float("nan")), per


def main():
    df = load_wide()
    X, labels, cols, w, counts = build_multiclass(df, TRAIN_CYCLES)
    classes = sorted(set(labels))
    cls_index = {c: i for i, c in enumerate(classes)}
    y = np.array([cls_index[c] for c in labels])
    n_cls = len(classes)

    log(f"多分类训练队列: n={len(X)} 特征={len(cols)} 类别={classes}")
    for c in classes:
        log(f"    {c:<14}{int((y == cls_index[c]).sum()):>6}")

    Xv = X.to_numpy(float)
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    learners = make_base_learners(RANDOM_STATE)
    log("计算多分类 OOF 张量 ...")
    O, names = oof_tensor(Xv, y, cv, learners, n_cls)

    def feat(Ot, idx):
        return np.concatenate([Ot[:, j, :] for j in idx], axis=1)

    log(f"穷举 {2 ** len(names) - 1} 种子集（宏平均 AUC）...")
    best = None
    for r in range(1, len(names) + 1):
        for combo in itertools.combinations(range(len(names)), r):
            F = feat(O, list(combo))
            aucs = []
            for tr, te in cv.split(F, y):
                lr = LogisticRegression(max_iter=1000, random_state=RANDOM_STATE)
                lr.fit(F[tr], y[tr])
                m, _ = macro_auc(y[te], lr.predict_proba(F[te]), classes)
                aucs.append(m)
            m = float(np.mean(aucs))
            if best is None or m > best["auc"]:
                best = {"auc": m, "subset": [names[i] for i in combo]}
    log(f"最优子集 {'+'.join(best['subset'])}  宏平均 AUC={best['auc']:.4f}")

    idx = [names.index(n) for n in best["subset"]]
    F = feat(O, idx)
    P_oof = np.zeros((len(y), n_cls))
    for tr, te in cv.split(F, y):
        m = LogisticRegression(max_iter=1000, random_state=RANDOM_STATE)
        m.fit(F[tr], y[tr])
        P_oof[te] = m.predict_proba(F[te])
    macro_tr, per_tr = macro_auc(y, P_oof, classes)
    acc_tr = float((P_oof.argmax(1) == y).mean())
    log(f"训练折外 宏平均AUC={macro_tr:.4f} 准确率={acc_tr:.4f}")

    final_lr = LogisticRegression(max_iter=1000, random_state=RANDOM_STATE)
    final_lr.fit(F, y)

    out = {
        "classes": classes, "n_train": int(len(X)), "n_features": len(cols),
        "best_subset": best["subset"], "macro_auc_train": round(macro_tr, 4),
        "accuracy_train": round(acc_tr, 4),
        "per_class_auc_train": {k: round(v, 4) for k, v in per_tr.items()},
        "class_counts": {c: int((y == cls_index[c]).sum()) for c in classes},
        "feasibility_note": (
            "论文1 的单标签互斥设定在系统性疾病上不适用：NHANES 训练队列中仅 "
            "26.5% 的参与者恰好患 1 种病，冠心病患者中只有 6% 是单一诊断。"
            "故多分类限定为样本量≥250 的类别；完整 7 病种筛查应使用多标签"
            "（one-vs-rest），即本项目 7 个二分类模型。"),
        "cohort_class_distribution": {},
        "validation": {},
    }

    # 外部队列：固定使用训练集类别集合，保证宏平均可比
    for tag, cycles in [("internal_2017_2018", INTERNAL_CYCLES),
                        ("external_2021_2023", EXTERNAL_CYCLES)]:
        Xe, lab_e, cols_e, we, cnt_e = build_multiclass(
            df, cycles, keep_classes=classes)
        out["cohort_class_distribution"][tag] = {
            c: int((lab_e == c).sum()) for c in classes}
        ye = np.array([cls_index[c] for c in lab_e])

        shared = [c for c in cols if c in cols_e]
        i_tr = [cols.index(c) for c in shared]
        PE = np.zeros((len(Xe), len(names), n_cls))
        for j, nm in enumerate(names):
            est = make_base_learners(RANDOM_STATE)[nm]
            est.fit(Xv[:, i_tr], y)
            p = est.predict_proba(Xe[shared].to_numpy(float))
            mc = list(est.classes_)
            for ci in range(n_cls):
                if ci in mc:
                    PE[:, j, ci] = p[:, mc.index(ci)]
        Pe = final_lr.predict_proba(feat(PE, idx))
        macro_e, per_e = macro_auc(ye, Pe, classes)
        acc_e = float((Pe.argmax(1) == ye).mean())
        out["validation"][tag] = {
            "n": int(len(Xe)), "macro_auc": round(macro_e, 4),
            "accuracy": round(acc_e, 4),
            "per_class_auc": {k: round(v, 4) for k, v in per_e.items()},
        }
        log(f"  {tag}: n={len(Xe)} 各类分布="
            f"{out['cohort_class_distribution'][tag]} "
            f"宏平均AUC={macro_e:.4f} 准确率={acc_e:.4f}")

    with open(os.path.join(REPORTS, "multiclass.json"), "w",
              encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    log("已保存 -> reports/multiclass.json")


if __name__ == "__main__":
    main()
