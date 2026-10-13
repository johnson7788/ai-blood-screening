"""导出 EBM 的可解释性产物：特征重要性、形状函数与局部解释。

EBM（Explainable Boosting Machine）是完全可加性模型：
    预测 = 截距 + Σ 每个特征的形状函数值 + Σ 成对交互项
因此每个特征对风险的贡献可以直接画出来，且与真实决策过程严格一致
（不同于 SHAP 对黑盒的近似，也不同于 LLM 的事后叙事）。

产出：
    reports/ebm_importance_<disease>.csv   各特征重要性与增减方向
    reports/figures/ebm_shapes_<disease>.png  顶部特征的形状函数曲线
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

from config import RANDOM_STATE, REPORTS, TRAIN_CYCLES
from data import build_cohort, load_wide
from variables import DISEASES, VAR_LABELS

for f in ["Arial Unicode MS", "PingFang SC", "Heiti SC", "STHeiti"]:
    matplotlib.rcParams["font.sans-serif"] = [f]
    break
matplotlib.rcParams["axes.unicode_minus"] = False

FIGDIR = os.path.join(REPORTS, "figures")
os.makedirs(FIGDIR, exist_ok=True)


def log(m):
    print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)


def analyse(df, key, top_n=8):
    from interpret.glassbox import ExplainableBoostingClassifier

    tr = build_cohort(df, key, feature_set="lab_demo", cycles=TRAIN_CYCLES)
    X = tr["X"].to_numpy(dtype=float)
    y = tr["y"]
    names = tr["var_names"]
    Xd = pd.DataFrame(X, columns=names)

    m = ExplainableBoostingClassifier(
        random_state=RANDOM_STATE, max_bins=256, interactions=10,
        outer_bags=8, learning_rate=0.03, n_jobs=-1)
    m.fit(Xd, y)

    # --- 特征重要性 ---
    imp = np.asarray(m.term_importances())
    terms = list(m.term_names_)
    exp = m.explain_global()

    rows = []
    for i, t in enumerate(terms):
        center = 0.0
        direction = ""
        try:
            xs, ys = exp.data(i)
            xs = np.asarray(xs, dtype=float)
            ys = np.asarray(ys, dtype=float)
            ok = np.isfinite(xs) & np.isfinite(ys)
            if ok.sum() >= 2:
                # 形状函数在高值端相对低值端的净变化 -> 方向
                lo = ys[ok][: max(1, ok.sum() // 4)].mean()
                hi = ys[ok][-max(1, ok.sum() // 4):].mean()
                direction = "升高↑" if hi > lo else "降低↓"
                center = float(np.abs(ys[ok]).mean())
        except Exception:
            pass
        label = VAR_LABELS.get(t, t)
        rows.append({
            "term": t, "feature": label,
            "importance": float(imp[i]),
            "mean_abs_contribution": round(center, 5),
            "方向": direction,
            "type": "交互项" if ("&" in t or " x " in t) else "单特征",
        })
    imp_df = pd.DataFrame(rows).sort_values("importance", ascending=False)
    imp_df.to_csv(os.path.join(REPORTS, f"ebm_importance_{key}.csv"),
                  index=False, encoding="utf-8-sig")

    # --- 形状函数图 ---
    single = imp_df[imp_df["type"] == "单特征"].head(top_n)
    ncol = 2
    nrow = int(np.ceil(len(single) / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(11, 2.4 * nrow))
    axes = np.atleast_1d(axes).ravel()
    for ax, (_, r) in zip(axes, single.iterrows()):
        i = terms.index(r["term"])
        try:
            xs, ys = exp.data(i)
            xs = np.asarray(xs, dtype=float)
            ys = np.asarray(ys, dtype=float)
            ok = np.isfinite(xs) & np.isfinite(ys)
            ax.plot(xs[ok], ys[ok], lw=1.8, color="#1f77b4")
            ax.axhline(0, color="grey", lw=0.8, ls="--", alpha=0.6)
            ax.set_title(f"{r['feature']}  (重要性 {r['importance']:.3f})",
                         fontsize=10)
            ax.set_xlabel("指标取值", fontsize=8)
            ax.set_ylabel("对 log-odds 的贡献", fontsize=8)
            ax.tick_params(labelsize=7)
            ax.grid(alpha=0.25)
        except Exception:
            ax.axis("off")
    for ax in axes[len(single):]:
        ax.axis("off")
    fig.suptitle(f"{DISEASES[key]['name']} — EBM 特征形状函数（可加性贡献）",
                 fontsize=12, y=1.0)
    fig.tight_layout()
    path = os.path.join(FIGDIR, f"ebm_shapes_{key}.png")
    fig.savefig(path, dpi=135, bbox_inches="tight")
    plt.close(fig)

    return imp_df, path


def main():
    df = load_wide()
    keys = sys.argv[1].split(",") if len(sys.argv) > 1 else list(DISEASES.keys())
    summary = {}
    for key in keys:
        log(f"{DISEASES[key]['name']} ...")
        imp_df, path = analyse(df, key)
        top = imp_df.head(6)
        summary[key] = {
            "name": DISEASES[key]["name"],
            "top": [{"feature": r["feature"],
                     "importance": round(float(r["importance"]), 4),
                     "direction": r["方向"], "type": r["type"]}
                    for r in top.to_dict("records")],
            "figure": os.path.basename(path),
        }
        log("  Top6: " + ", ".join(
            f"{r['feature']}({r['importance']:.3f}{r['方向']})"
            for _, r in top.iterrows()))

    with open(os.path.join(REPORTS, "ebm_summary.json"), "w",
              encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    log(f"已保存 -> reports/ebm_summary.json 与 ebm_shapes_*.png")


if __name__ == "__main__":
    main()
