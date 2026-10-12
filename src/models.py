"""堆叠集成、子集穷举与递归特征消除（复现论文1 的核心方法）。

关键工程优化
------------
论文1 穷举 255 种基学习器子集。若对每个子集都重训全部基学习器，代价是
255 × 5 折 × 8 模型，完全不可行。但基学习器之间相互独立，其 out-of-fold
预测**不随子集变化**，因此：
  1. 先算一次 8 个基学习器的 OOF 预测矩阵（n × 8）
  2. 子集搜索时只需在 OOF 矩阵上重训 LR 元学习器（255 次 LR 拟合，秒级）
结果与逐子集重训完全等价，但速度提高两个数量级。
"""
import itertools
import warnings

import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin, clone
from sklearn.ensemble import AdaBoostClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

warnings.filterwarnings("ignore")

try:
    from xgboost import XGBClassifier
    HAS_XGB = True
except Exception:  # pragma: no cover
    HAS_XGB = False


def _scaled(est):
    """需要标准化的模型包一层 Pipeline。"""
    return Pipeline([("imp", SimpleImputer(strategy="median")),
                     ("sc", StandardScaler()), ("m", est)])


def make_base_learners(random_state=42):
    """论文1 的 8 个基学习器。"""
    from sklearn.tree import DecisionTreeClassifier
    learners = {
        "AdaBoost": AdaBoostClassifier(
            estimator=DecisionTreeClassifier(max_depth=1, random_state=random_state),
            n_estimators=200, random_state=random_state),
        "KNN": _scaled(KNeighborsClassifier(n_neighbors=25, n_jobs=-1)),
        "LR": _scaled(LogisticRegression(max_iter=2000, C=1.0,
                                         random_state=random_state)),
        "MLP": _scaled(MLPClassifier(hidden_layer_sizes=(64, 32), max_iter=300,
                                     early_stopping=True, random_state=random_state)),
        "NB": _scaled(GaussianNB()),
        "RF": RandomForestClassifier(n_estimators=300, min_samples_leaf=5,
                                     n_jobs=-1, random_state=random_state),
        "SVM": _scaled(SVC(probability=True, C=1.0, gamma="scale",
                           random_state=random_state)),
        "XGBoost": None,
    }
    if HAS_XGB:
        learners["XGBoost"] = XGBClassifier(
            n_estimators=300, max_depth=5, learning_rate=0.08,
            subsample=0.8, colsample_bytree=0.8, eval_metric="logloss",
            n_jobs=-1, random_state=random_state, verbosity=0)
    else:
        learners.pop("XGBoost")
    return learners


def oof_matrix(X, y, cv, learners=None, random_state=42):
    """一次性算出所有基学习器的 out-of-fold 预测概率矩阵 (n_samples, n_learners)。"""
    learners = learners or make_base_learners(random_state)
    names = list(learners.keys())
    O = np.zeros((len(y), len(names)))
    for j, name in enumerate(names):
        est = learners[name]
        p = cross_val_predict(est, X, y, cv=cv, method="predict_proba", n_jobs=1)[:, 1]
        O[:, j] = p
    return O, names


class StackingEnsemble(BaseEstimator, ClassifierMixin):
    """两层堆叠：8 基学习器 + LR 元学习器，支持指定基学习器子集。"""

    def __init__(self, subset=None, random_state=42, meta_C=1.0):
        self.subset = subset
        self.random_state = random_state
        self.meta_C = meta_C

    def fit(self, X, y, cv=None):
        X = np.asarray(X, dtype=float)
        y = np.asarray(y)
        cv = cv or StratifiedKFold(n_splits=5, shuffle=True,
                                  random_state=self.random_state)
        all_learners = make_base_learners(self.random_state)
        self.all_names_ = list(all_learners.keys())
        self.names_ = self.subset or self.all_names_
        self.learners_ = {n: all_learners[n] for n in self.names_}

        # 元学习器的训练特征 = 基学习器的 OOF 预测
        O, names = oof_matrix(X, y, cv, {n: self.learners_[n] for n in self.names_},
                              self.random_state)
        self.meta_ = LogisticRegression(max_iter=2000, C=self.meta_C,
                                        random_state=self.random_state)
        self.meta_.fit(O, y)
        # 基学习器在全量训练集上重训，供预测使用
        for n in self.names_:
            self.learners_[n].fit(X, y)
        self.classes_ = np.unique(y)
        return self

    def predict_proba(self, X):
        X = np.asarray(X, dtype=float)
        M = np.column_stack([self.learners_[n].predict_proba(X)[:, 1]
                             for n in self.names_])
        return self.meta_.predict_proba(M)

    def predict(self, X):
        return (self.predict_proba(X)[:, 1] >= 0.5).astype(int)


def search_subsets(O, y, names, cv, random_state=42, meta_C=1.0):
    """穷举 2^8-1 = 255 种非空基学习器子集，按 CV AUC 排序。"""
    results = []
    for r in range(1, len(names) + 1):
        for combo in itertools.combinations(range(len(names)), r):
            sub = O[:, list(combo)]
            scores = []
            for tr, te in cv.split(sub, y):
                lr = LogisticRegression(max_iter=2000, C=meta_C,
                                        random_state=random_state)
                lr.fit(sub[tr], y[tr])
                scores.append(roc_auc_score(y[te], lr.predict_proba(sub[te])[:, 1]))
            results.append({
                "subset": [names[i] for i in combo],
                "n_learners": r,
                "auc_mean": float(np.mean(scores)),
                "auc_std": float(np.std(scores)),
            })
    results.sort(key=lambda d: -d["auc_mean"])
    return results


def cv_meta_auc(O, y, subset, names, cv, random_state=42, meta_C=1.0):
    """在 OOF 矩阵上对**元学习器**再做一次 CV，得到无泄漏的 CV AUC。

    注意：绝不能拿元学习器在 O 上拟合、又在同一个 O 上预测——那是在训练集上
    评估，AUC 会被系统性高估。这里对元学习器本身做折外预测。
    """
    idx = [names.index(n) for n in subset]
    scores = []
    for tr, te in cv.split(O, y):
        lr = LogisticRegression(max_iter=2000, C=meta_C,
                                random_state=random_state)
        lr.fit(O[tr][:, idx], y[tr])
        scores.append(roc_auc_score(y[te], lr.predict_proba(O[te][:, idx])[:, 1]))
    return float(np.mean(scores)), float(np.std(scores))


def oof_meta_probability(O, y, subset, names, cv, random_state=42, meta_C=1.0):
    """元学习器的折外预测概率（用于画 ROC / 算阈值，无泄漏）。"""
    idx = [names.index(n) for n in subset]
    p = np.zeros(len(y))
    for tr, te in cv.split(O, y):
        lr = LogisticRegression(max_iter=2000, C=meta_C,
                                random_state=random_state)
        lr.fit(O[tr][:, idx], y[tr])
        p[te] = lr.predict_proba(O[te][:, idx])[:, 1]
    return p


def rfe_by_importance(X, y, var_names, cv, n_features=12, random_state=42,
                      verbose=False):
    """基于排列重要性的递归特征消除。

    论文1 用置换重要性谱 + 迭代五折 CV 做递归消除。此处用 XGBoost 的
    排列重要性作为消除依据（每次去掉重要性最低的特征），再用五折 CV 确认，
    最后把保留的特征集交回完整堆叠管线评估。
    """
    from sklearn.inspection import permutation_importance

    X = np.asarray(X, dtype=float)
    y = np.asarray(y)
    keep = list(range(X.shape[1]))
    history = []
    while len(keep) > n_features:
        model = (XGBClassifier(n_estimators=200, max_depth=5, learning_rate=0.08,
                               subsample=0.8, colsample_bytree=0.8,
                               eval_metric="logloss", n_jobs=-1,
                               random_state=random_state, verbosity=0)
                 if HAS_XGB else RandomForestClassifier(
                     n_estimators=200, n_jobs=-1, random_state=random_state))
        # 用折内 CV AUC 作为当前特征集的性能
        p = cross_val_predict(model, X[:, keep], y, cv=cv,
                              method="predict_proba", n_jobs=1)[:, 1]
        auc = roc_auc_score(y, p)

        model.fit(X[:, keep], y)
        pi = permutation_importance(model, X[:, keep], y, n_repeats=5,
                                    random_state=random_state,
                                    scoring="roc_auc", n_jobs=-1)
        worst = int(np.argmin(pi.importances_mean))
        removed = var_names[keep[worst]]
        history.append({"n_features": len(keep), "cv_auc": round(float(auc), 4),
                        "removed": removed})
        if verbose:
            print(f"    {len(keep):>3} 特征  CV AUC={auc:.4f}  移除 {removed}")
        keep.pop(worst)
    return [var_names[i] for i in keep], history
