"""候选方案公平对比框架。

在完全相同的队列划分与验证口径下，比较不同模型：
    - 训练集：2013-2016 完整案例（5 折交叉验证给出无泄漏的训练集估计）
    - 内部验证：2017-2018
    - 外部验证：2021-2023（跨年代分布漂移）

与 run.py 的堆叠集成基线使用同一套 data.py / metrics.py，保证可比。

设计约定：每个候选提供一个 `fit(X, y) -> model`，其中 model 具备
`predict_proba_pos(X) -> 一维阳性概率`。交叉验证用 `_CVWrapper` 适配。
这样可避免「predict_proba 返回训练集内预测」这类隐蔽的评估泄漏。

用法：
    python src/compare_models.py ebm
    python src/compare_models.py tabpfn
    python src/compare_models.py autogluon
    python src/compare_models.py ebm,tabpfn,autogluon
"""
import json
import os
import sys
import time
import warnings

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.model_selection import StratifiedKFold, cross_val_predict

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
warnings.filterwarnings("ignore")

from config import (EXTERNAL_CYCLES, INTERNAL_CYCLES, RANDOM_STATE, REPORTS,
                    TRAIN_CYCLES)
from data import build_cohort, load_wide
from metrics import binary_metrics
from variables import DISEASES

OUT = os.path.join(REPORTS, "compare_models.json")
SUBSAMPLE = int(os.environ.get("CMP_SUBSAMPLE", "0")) or None


def log(m):
    print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)


class _CVWrapper(BaseEstimator, ClassifierMixin):
    """把 fit -> model 的候选适配成 sklearn 分类器，供 cross_val_predict 使用。"""

    def __init__(self, fit_fn):
        self.fit_fn = fit_fn

    def get_params(self, deep=True):
        return {"fit_fn": self.fit_fn}

    def set_params(self, **kw):
        return self

    def fit(self, X, y):
        self.model_ = self.fit_fn(np.asarray(X, dtype=float), np.asarray(y))
        self.classes_ = np.array([0, 1])
        return self

    def predict_proba(self, X):
        p = self.model_.predict_proba_pos(np.asarray(X, dtype=float))
        return np.column_stack([1.0 - p, p])

    def predict(self, X):
        return (self.predict_proba(X)[:, 1] >= 0.5).astype(int)


# --------------------------------------------------------------------------
# 候选方案工厂：返回 fit(X, y) -> model
# --------------------------------------------------------------------------

def make_ebm(seed=RANDOM_STATE):
    from interpret.glassbox import ExplainableBoostingClassifier

    class _M:
        def __init__(self, m):
            self.m = m

        def predict_proba_pos(self, X):
            return self.m.predict_proba(X)[:, 1]

    def fit(X, y):
        m = ExplainableBoostingClassifier(
            random_state=seed,
            max_bins=256,
            interactions=10,
            outer_bags=8,
            learning_rate=0.03,
            n_jobs=-1,
        )
        m.fit(X, y)
        return _M(m)
    return fit


def make_tabpfn(seed=RANDOM_STATE):
    from tabpfn import TabPFNClassifier
    import torch

    class _M:
        def __init__(self, m):
            self.m = m

        def predict_proba_pos(self, X):
            return self.m.predict_proba(X)[:, 1]

    def fit(X, y):
        # 本机实测：MPS 在 11.4k 样本上显存溢出（8.29 GiB 上限）；
        # CPU 默认拒绝 >5000 样本，且全量单次 fit 超 10 分钟。
        # 因此 TabPFN 一律在 CPU 上、于其原生适用区间（<=5000 样本）内评测。
        m = TabPFNClassifier(device="cpu", random_state=seed,
                             ignore_pretraining_limits=True)
        m.fit(X, y)
        return _M(m)
    return fit


def make_autogluon(seed=RANDOM_STATE, time_limit=45):
    """AutoGluon 自动集成。

    环境限制说明（macOS + Homebrew libomp）：
      - GBM(LightGBM) 与 XGB 在本环境触发 SIGSEGV（OpenMP 运行时冲突），
        实测 KMP_DUPLICATE_LIB_OK / OMP_NUM_THREADS 均无法规避，故排除；
      - CAT / FASTAI 未安装；
      - TABPFN-* / TABICL 受许可门槛限制，无法下载权重。
    因此使用可稳定运行的安全子集 RF/XT/KNN/LR/NN_TORCH/REALMLP/TABM。
    这使 AutoGluon 无法动用其最强的 GBDT 成员，结果须据此解读。
    """
    from autogluon.tabular import TabularPredictor

    SAFE_PARAMS = {"RF": {}, "XT": {}, "KNN": {}, "LR": {},
                   "NN_TORCH": {}, "REALMLP": {}, "TABM": {}}

    class _M:
        def __init__(self, p, cols):
            self.p = p
            self.cols = cols

        def predict_proba_pos(self, X):
            Xd = pd.DataFrame(np.asarray(X, dtype=float), columns=self.cols)
            return self.p.predict_proba(Xd).iloc[:, 1].to_numpy()

    def fit(X, y):
        X = np.asarray(X, dtype=float)
        cols = [f"f{i}" for i in range(X.shape[1])]
        tr = pd.DataFrame(X, columns=cols)
        tr["__y__"] = np.asarray(y)
        p = TabularPredictor(label="__y__", eval_metric="roc_auc",
                             verbosity=0).fit(
            tr, presets="medium_quality", time_limit=time_limit,
            num_bag_folds=0, num_stack_levels=0,
            hyperparameters=SAFE_PARAMS)
        return _M(p, cols)
    return fit


CANDIDATES = {
    "ebm": ("EBM（可解释加性模型）", make_ebm),
    "tabpfn": ("TabPFN（表格基础模型）", make_tabpfn),
    "autogluon": ("AutoGluon（自动集成）", make_autogluon),
}


def run(name, df, diseases=None, subsample=None):
    label, factory = CANDIDATES[name]
    fit = factory()
    diseases = diseases or list(DISEASES.keys())
    results = []

    for key in diseases:
        info = DISEASES[key]
        t0 = time.time()
        tr = build_cohort(df, key, feature_set="lab_demo", cycles=TRAIN_CYCLES)
        X = tr["X"].to_numpy(dtype=float)
        y = tr["y"]
        w = tr["w"]
        names = tr["var_names"]
        n_full = len(y)
        if subsample and subsample < n_full:
            rng = np.random.default_rng(RANDOM_STATE)
            # 分层抽样，保持患病率
            idx0 = np.where(y == 0)[0]; idx1 = np.where(y == 1)[0]
            k0 = int(round(subsample * len(idx0) / n_full))
            k1 = subsample - k0
            sel = np.concatenate([
                rng.choice(idx0, size=min(k0, len(idx0)), replace=False),
                rng.choice(idx1, size=min(k1, len(idx1)), replace=False)])
            sel.sort()
            X, y, w = X[sel], y[sel], w[sel]
            log(f"  {info['name']}: 训练 n={len(y)}/{n_full}（子采样）"
                f" 特征={len(names)} 患病率={y.mean():.3f}")
        else:
            log(f"  {info['name']}: 训练 n={tr['n']} 特征={len(names)}")

        # 训练集：折外交叉验证（无泄漏）
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
        p_cv = cross_val_predict(_CVWrapper(fit), X, y, cv=cv,
                                 method="predict_proba", n_jobs=1)[:, 1]

        # 全量训练后预测各验证队列
        model = fit(X, y)
        cohorts = {}
        for tag, cycles in [("internal_2017_2018", INTERNAL_CYCLES),
                            ("external_2021_2023", EXTERNAL_CYCLES)]:
            co = build_cohort(df, key, feature_set="lab_demo", cycles=cycles)
            p = model.predict_proba_pos(co["X"][names].to_numpy(dtype=float))
            cohorts[tag] = (co["y"], p, co["w"], co["n"])

        m_cv = binary_metrics(y, p_cv, w)
        entry = {
            "model": name, "model_label": label,
            "disease": key, "name": info["name"],
            "n_train": int(len(y)), "n_full": int(n_full),
            "n_features": len(names),
            "train_cv_auc": round(m_cv["auc"], 4),
            "train_cv_auc_weighted": round(m_cv["auc_weighted"], 4),
            "validation": {},
        }
        for tag, (yy, pp, ww, nn) in cohorts.items():
            m = binary_metrics(yy, pp, ww)
            entry["validation"][tag] = {
                "n": int(nn), "auc": round(m["auc"], 4),
                "auc_weighted": round(m["auc_weighted"], 4),
                "auprc": round(m["auprc"], 4),
                "sensitivity": round(m["sensitivity"], 4),
                "specificity": round(m["specificity"], 4),
                "ppv": round(m["ppv"], 4), "npv": round(m["npv"], 4),
                "brier": round(m["brier"], 4), "ece": round(m["ece"], 4),
                "threshold": round(m["threshold"], 6),
            }
        ext = entry["validation"]["external_2021_2023"]["auc"]
        entry["auc_drop_vs_train"] = round(ext - entry["train_cv_auc"], 4)
        entry["elapsed_sec"] = round(time.time() - t0, 1)

        log(f"    训练CV={entry['train_cv_auc']:.4f} "
            f"内部={entry['validation']['internal_2017_2018']['auc']:.4f} "
            f"外部={ext:.4f} ({entry['elapsed_sec']}s)")
        results.append(entry)
        # 增量保存，防中断丢结果
        _save(results, name)
    return results


def _out_path():
    if SUBSAMPLE:
        return os.path.join(REPORTS, f"compare_models_sub{SUBSAMPLE}.json")
    return OUT


def _load():
    p = _out_path()
    if os.path.exists(p):
        try:
            return json.load(open(p, encoding="utf-8"))
        except Exception:
            return []
    return []


def _save(new, name):
    old = [r for r in _load() if r["model"] != name]
    with open(_out_path(), "w", encoding="utf-8") as f:
        json.dump(old + new, f, ensure_ascii=False, indent=2)


def summarise():
    allr = _load()
    if not allr:
        return
    rows = []
    for r in allr:
        rows.append({
            "模型": r["model"], "病种": r["name"],
            "训练CV": r["train_cv_auc"],
            "内部": r["validation"]["internal_2017_2018"]["auc"],
            "外部": r["validation"]["external_2021_2023"]["auc"],
            "外部掉幅": r["auc_drop_vs_train"],
        })
    s = pd.DataFrame(rows)
    print()
    print("=== 按病种 × 模型 ===")
    print(s.sort_values(["病种", "模型"]).to_string(index=False))
    print()
    print("=== 按模型平均（外部 AUC）===")
    print(s.groupby("模型")[["训练CV", "内部", "外部"]].mean()
          .round(4).sort_values("外部", ascending=False).to_string())


def main():
    if len(sys.argv) < 2:
        print("用法: python src/compare_models.py <ebm|tabpfn|autogluon>[,...]")
        sys.exit(1)
    names = sys.argv[1].split(",")
    diseases = sys.argv[2].split(",") if len(sys.argv) > 2 else None
    df = load_wide()
    for nm in names:
        if nm not in CANDIDATES:
            log(f"跳过未知模型 {nm}（可选：{list(CANDIDATES)}）")
            continue
        log(f"===== 候选方案: {nm} — {CANDIDATES[nm][0]} =====")
        try:
            run(nm, df, diseases, subsample=SUBSAMPLE)
        except Exception as e:
            import traceback
            log(f"  ❌ {nm} 失败: {type(e).__name__}: {str(e)[:300]}")
            traceback.print_exc(limit=2)
    summarise()


if __name__ == "__main__":
    main()
