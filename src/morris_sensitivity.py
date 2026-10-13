"""Morris 全局敏感性分析（Method of Morris / Elementary Effects）。

原理
----
Morris 方法通过在输入空间生成多条「一次一个变量」（OAT）轨迹来估计每个变量的
**基本效应（Elementary Effect, EE）**：

    EE_i = [ f(x + Δ·e_i) - f(x) ] / Δ

对每条轨迹、每个变量各算一次 EE，再跨轨迹汇总得到：

    μ       —— EE 的均值（正负抵消，可判断作用方向）
    μ*      —— |EE| 的均值（**重要性**，不受方向抵消影响）
    σ       —— EE 的标准差（**非线性与交互作用强度**：σ 大说明该变量的效应
                依赖其他变量的取值，即存在交互或强非线性）

因此 μ*-σ 图是经典解读工具：
    - μ* 大、σ 小  → 重要且效应近似线性、可加
    - μ* 大、σ 大  → 重要但高度依赖其他变量（交互）
    - μ* 小         → 不重要

重要说明
--------
Morris 度量的是**模型对输入的敏感性**，不是现实世界的因果效应。
本项目的特征在真实数据中彼此相关，而 Morris 会独立地扰动各变量，
因此会产生一些在真实人群中罕见的特征组合。解读时应视为
「模型如何响应」，而非「干预该指标会如何改变风险」。
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

N_TRAJECTORIES = 20      # 轨迹数 r
NUM_LEVELS = 4           # 每个变量的网格层数
LOWER_Q, UPPER_Q = 1, 99  # 取值边界分位数（避免外推到极端区域）


def log(m):
    print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)


def build_model(key, df):
    """训练 EBM 作为被分析模型（与 ebm_explain.py 保持一致）。"""
    from interpret.glassbox import ExplainableBoostingClassifier

    tr = build_cohort(df, key, feature_set="lab_demo", cycles=TRAIN_CYCLES)
    X = tr["X"].to_numpy(dtype=float)
    y = tr["y"]
    names = tr["var_names"]
    m = ExplainableBoostingClassifier(
        random_state=RANDOM_STATE, max_bins=256, interactions=10,
        outer_bags=8, learning_rate=0.03, n_jobs=-1)
    m.fit(pd.DataFrame(X, columns=names), y)
    return m, X, y, names


def morris_analyse(key, df, verbose=False):
    from SALib.sample.morris import sample as morris_sample
    from SALib.analyze.morris import analyze as morris_analyze

    m, X, y, names = build_model(key, df)
    lo = np.percentile(X, LOWER_Q, axis=0)
    hi = np.percentile(X, UPPER_Q, axis=0)
    # 防止常量列导致零区间
    span = hi - lo
    span[span <= 0] = 1e-6
    hi = lo + span

    problem = {
        "num_vars": len(names),
        "names": names,
        "bounds": [[float(a), float(b)] for a, b in zip(lo, hi)],
    }
    Xs = morris_sample(problem, N=N_TRAJECTORIES, num_levels=NUM_LEVELS,
                       seed=RANDOM_STATE)
    # 模型输入需要是 DataFrame 且列名一致（EBM 用 DataFrame 训练）
    ys = m.predict_proba(pd.DataFrame(Xs, columns=names))[:, 1]
    Si = morris_analyze(problem, Xs, ys, num_resamples=200, conf_level=0.95,
                        print_to_console=False, seed=RANDOM_STATE)

    df_out = pd.DataFrame({
        "variable": names,
        "feature": [VAR_LABELS.get(v, v) for v in names],
        "mu": np.asarray(Si["mu"], dtype=float),
        "mu_star": np.asarray(Si["mu_star"], dtype=float),
        "sigma": np.asarray(Si["sigma"], dtype=float),
        "mu_star_conf": np.asarray(Si["mu_star_conf"], dtype=float),
    })
    df_out["rank"] = df_out["mu_star"].rank(ascending=False).astype(int)
    df_out = df_out.sort_values("mu_star", ascending=False).reset_index(drop=True)
    df_out["方向"] = np.where(df_out["mu"] > 0, "升高↑", "降低↓")
    df_out.to_csv(os.path.join(REPORTS, f"morris_{key}.csv"),
                  index=False, encoding="utf-8-sig")

    if verbose:
        top = df_out.head(8)
        log("  Top8(μ*): " + ", ".join(
            f"{r.feature}({r.mu_star:.4f})" for r in top.itertuples()))

    # μ*-σ 图
    fig, ax = plt.subplots(figsize=(7.5, 6))
    ax.scatter(df_out["mu_star"], df_out["sigma"], s=28, color="#1f77b4",
               alpha=0.8, edgecolors="white", linewidths=0.5)
    for r in df_out.head(10).itertuples():
        ax.annotate(r.feature, (r.mu_star, r.sigma), fontsize=8,
                    xytext=(4, 3), textcoords="offset points")
    ax.set_xlabel("μ*  —— 重要性（平均绝对基本效应）")
    ax.set_ylabel("σ  —— 非线性与交互作用强度")
    ax.set_title(f"{DISEASES[key]['name']} — Morris 敏感性分析\n"
                 f"（{N_TRAJECTORIES} 条轨迹，{NUM_LEVELS} 层网格）")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    path = os.path.join(FIGDIR, f"morris_{key}.png")
    fig.savefig(path, dpi=140)
    plt.close(fig)

    return df_out, path, len(Xs)


def main():
    df = load_wide()
    keys = sys.argv[1].split(",") if len(sys.argv) > 1 else list(DISEASES.keys())
    summary = {}
    for key in keys:
        log(f"{DISEASES[key]['name']} ...")
        out, path, n_samples = morris_analyse(key, df, verbose=True)
        summary[key] = {
            "name": DISEASES[key]["name"],
            "n_trajectories": N_TRAJECTORIES,
            "n_model_evals": int(n_samples),
            "top": [{"feature": r.feature, "mu_star": round(r.mu_star, 5),
                     "sigma": round(r.sigma, 5), "direction": r.方向}
                    for r in out.head(8).itertuples()],
            "figure": os.path.basename(path),
        }
    with open(os.path.join(REPORTS, "morris_summary.json"), "w",
              encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    log("已保存 -> reports/morris_summary.json 与 morris_*.csv / morris_*.png")


if __name__ == "__main__":
    main()
