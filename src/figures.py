"""结果可视化：ROC 曲线、模型对比、标签泄漏对照。"""
import json
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import roc_curve
from sklearn.model_selection import StratifiedKFold

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import EXTERNAL_CYCLES, INTERNAL_CYCLES, RANDOM_STATE, REPORTS, TRAIN_CYCLES
from data import build_cohort, load_wide
from models import (StackingEnsemble, make_base_learners, oof_matrix,
                    oof_meta_probability)
from variables import DISEASES

# 中文字体
for f in ["Arial Unicode MS", "PingFang SC", "Heiti SC", "STHeiti"]:
    matplotlib.rcParams["font.sans-serif"] = [f]
    break
matplotlib.rcParams["axes.unicode_minus"] = False

FIGDIR = os.path.join(REPORTS, "figures")
os.makedirs(FIGDIR, exist_ok=True)

RESULTS = os.path.join(REPORTS, "results.json")


def roc_figure(df, res):
    """每个病种一张 ROC 图：训练折外 / 内部 / 外部。"""
    key = res["disease"]
    name = res["name"]
    tr = build_cohort(df, key, feature_set="lab_demo", cycles=TRAIN_CYCLES)
    X = tr["X"].to_numpy(dtype=float)
    y = tr["y"]
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    learners = make_base_learners(RANDOM_STATE)
    O, lnames = oof_matrix(X, y, cv, learners, RANDOM_STATE)
    p_tr = oof_meta_probability(O, y, res["best_subset"], lnames, cv, RANDOM_STATE)

    model = StackingEnsemble(subset=res["best_subset"], random_state=RANDOM_STATE)
    model.fit(X, y, cv=cv)
    names = tr["var_names"]

    curves = [("训练折外 (2013-2016)", y, p_tr, "#1f77b4")]
    for tag, cyc, color in [("内部验证 (2017-2018)", INTERNAL_CYCLES, "#2ca02c"),
                            ("外部验证 (2021-2023)", EXTERNAL_CYCLES, "#d62728")]:
        co = build_cohort(df, key, feature_set="lab_demo", cycles=cyc)
        shared = [c for c in names if c in co["var_names"]]
        if len(shared) < len(names):
            idx = [names.index(c) for c in shared]
            m2 = StackingEnsemble(subset=res["best_subset"], random_state=RANDOM_STATE)
            m2.fit(X[:, idx], y, cv=cv)
            p = m2.predict_proba(co["X"][shared].to_numpy(dtype=float))[:, 1]
        else:
            p = model.predict_proba(co["X"][names].to_numpy(dtype=float))[:, 1]
        curves.append((tag, co["y"], p, color))

    fig, ax = plt.subplots(figsize=(6, 5.5))
    for tag, yy, pp, c in curves:
        fpr, tpr, _ = roc_curve(yy, pp)
        auc = np.trapezoid(tpr, fpr)
        ax.plot(fpr, tpr, color=c, lw=2, label=f"{tag}  AUC={auc:.3f}")
    ax.plot([0, 1], [0, 1], "k--", lw=1, alpha=0.5)
    ax.set_xlabel("假阳性率 (1-特异度)")
    ax.set_ylabel("真阳性率 (敏感度)")
    ax.set_title(f"{name} — 堆叠集成 ROC")
    ax.legend(loc="lower right", fontsize=9)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    path = os.path.join(FIGDIR, f"roc_{key}.png")
    fig.savefig(path, dpi=140)
    plt.close(fig)
    return path


def summary_figure(results):
    """模型对比条形图：最佳单模型 / 堆叠 / RFE12特征，及外部掉幅。"""
    names = [r["name"] for r in results]
    single = [r["best_single_auc"] for r in results]
    stack = [r["stacking_cv_auc"] for r in results]
    ext = [r["validation"]["external_2021_2023"]["auc"] for r in results]
    rfe = [r.get("rfe_cv_auc") for r in results]

    x = np.arange(len(names))
    w = 0.2
    fig, axes = plt.subplots(1, 2, figsize=(15, 5.5))

    ax = axes[0]
    ax.bar(x - 1.5 * w, single, w, label="最佳单模型 (CV)", color="#7f7f7f")
    ax.bar(x - 0.5 * w, stack, w, label="堆叠集成 (CV)", color="#1f77b4")
    ax.bar(x + 0.5 * w, rfe, w, label="RFE 12特征 (CV)", color="#ff7f0e")
    ax.bar(x + 1.5 * w, ext, w, label="外部验证 2021-2023", color="#d62728")
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=20, ha="right")
    ax.set_ylabel("AUC")
    ax.set_ylim(0.5, 1.0)
    ax.axhline(0.8, ls="--", lw=1, color="green", alpha=0.6)
    ax.text(len(names) - 0.5, 0.805, "验收线 AUC 0.80", color="green", fontsize=8)
    ax.set_title("各病种模型性能对比")
    ax.legend(fontsize=8)
    ax.grid(axis="y", alpha=0.3)

    ax = axes[1]
    drops = [r["validation"]["external_2021_2023"]["auc_drop_vs_train"]
             for r in results]
    colors = ["#2ca02c" if d > -0.05 else "#d62728" for d in drops]
    ax.barh(names, drops, color=colors)
    ax.axvline(-0.05, ls="--", lw=1.5, color="red", alpha=0.7)
    ax.text(-0.05, len(names) - 0.5, " 验收线 -0.05", color="red", fontsize=8)
    ax.set_xlabel("外部验证 AUC 相对训练 CV 的掉幅")
    ax.set_title("分布漂移鲁棒性（越接近 0 越好）")
    ax.grid(axis="x", alpha=0.3)

    fig.tight_layout()
    path = os.path.join(FIGDIR, "summary.png")
    fig.savefig(path, dpi=140)
    plt.close(fig)
    return path


def leakage_figure(results):
    """标签泄漏对照图。"""
    lk = [r for r in results if r.get("leakage_control")]
    if not lk:
        return None
    names, with_lk, without_lk = [], [], []
    for r in lk:
        c = r["leakage_control"]
        names.append(r["name"])
        with_lk.append(c["cv_auc_with_leakage"])
        without_lk.append(c["cv_auc_without_leakage"])

    x = np.arange(len(names))
    w = 0.35
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.bar(x - w / 2, with_lk, w, label="使用同源诊断变量（含泄漏）", color="#d62728")
    ax.bar(x + w / 2, without_lk, w, label="剔除同源变量（本复现采用）", color="#1f77b4")
    for i, (a, b) in enumerate(zip(with_lk, without_lk)):
        ax.annotate(f"虚高 {a - b:+.3f}", (i, max(a, b) + 0.008),
                    ha="center", fontsize=9, color="#d62728")
    ax.set_xticks(x)
    ax.set_xticklabels(names)
    ax.set_ylabel("训练集折外 CV AUC")
    ax.set_ylim(0.7, 1.0)
    ax.set_title("标签泄漏的影响：同源诊断变量会显著虚高性能")
    ax.legend(fontsize=9)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    path = os.path.join(FIGDIR, "leakage_control.png")
    fig.savefig(path, dpi=140)
    plt.close(fig)
    return path


def main():
    with open(RESULTS, encoding="utf-8") as f:
        results = json.load(f)
    df = load_wide()
    made = []
    for r in results:
        p = roc_figure(df, r)
        made.append(p)
        print("已生成", os.path.basename(p), flush=True)
    made.append(summary_figure(results))
    print("已生成 summary.png")
    p = leakage_figure(results)
    if p:
        made.append(p)
        print("已生成 leakage_control.png")
    print(f"\n共 {len(made)} 张图 -> {FIGDIR}")


if __name__ == "__main__":
    main()
