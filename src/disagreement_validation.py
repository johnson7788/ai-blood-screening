"""Phase 2：多方法重要性「分歧度」指标的系统验证。

研究假设
--------
H1：三种重要性归因方法（EBM 原生 / SHAP / Morris）的排序分歧度，
    与该疾病终点的**外部验证性能**负相关——分歧越大，性能越差。
若成立，则「分歧度」可作为**无需外部验证数据**的信号质量指标。

方法学要点（相对此前工作的改进）
--------------------------------
之前的分析中，EBM 原生重要性来自 EBM 模型，而 SHAP 来自 XGBoost——
**模型不同会与"归因方法分歧"混淆**。本脚本让三种方法统一作用于
**同一个 EBM 模型**，从而干净地度量归因方法之间的分歧：
    1. EBM 原生项重要性（term_importances）
    2. SHAP（对同一 EBM 的预测函数）
    3. Morris μ*（对同一 EBM 做全局扰动）

分歧度指标
----------
    agreement_rho  : 三对 Spearman 秩相关的均值（越高越一致）
    disagreement   : 1 − agreement_rho
    jaccard_top5   : 三者 Top-5 特征集合的两两 Jaccard 均值
    jaccard_top10  : 同上，K=10
    rank_var       : 归一化排名方差（越小越一致）

产出
----
reports/disagreement_validation.csv   各终点的分歧度与性能
reports/disagreement_report.md        验证报告
reports/figures/disagreement_vs_auc.png
"""
import json
import os
import sys
import time
import warnings

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.model_selection import StratifiedKFold

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
warnings.filterwarnings("ignore")

from config import RANDOM_STATE, REPORTS, TRAIN_CYCLES
from data import STRUCTURAL_EXCLUDE, analysis_mask, load_wide
from metrics import binary_metrics
from variables import FEATURE_SETS, VAR_LABELS

EXT_PARQUET = os.path.join(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))), "nhanes_endpoints.parquet")

# 队列划分：训练 2013-2016；外部验证用 2017-2018（20 个终点全覆盖）
# 2021-2023 作为次级验证（仅 15 个终点有数据）
TRAIN = ["2013-2014", "2015-2016"]
EXTERNAL = ["2017-2018"]
EXTERNAL2 = ["2021-2023"]

N_MORRIS_TRAJ = 15
SHAP_SAMPLE = 1500

# 各终点的「同源诊断变量」——必须剔除，否则 AUC 被泄漏抬高，
# 而泄漏与分歧度之间可能产生伪相关，污染本研究要度量的关系。
_DIAB = ["LBXSGL", "LBXGLU", "LBXGH", "LBXIN"]      # 血糖相关
_RENAL = ["LBXSCR", "LBXSBU", "LBXSUA"]              # 肾功能相关
_LIVER = ["LBXSATSI", "LBXSASSI", "LBXSTB"]          # 肝功能相关
LEAKAGE = {
    "DIQ010": _DIAB,          # 糖尿病
    "DIQ160": _DIAB,          # 糖尿病前期
    "RHQ162": _DIAB,          # 妊娠糖尿病（由糖耐量试验诊断）
    "KIQ022": _RENAL,         # 肾功能不全
    "KIQ026": _RENAL,         # 肾结石
    "MCQ160L": _LIVER,        # 肝脏疾病（可由肝酶异常诊断）
    "MCQ203": _LIVER,         # 黄疸（胆红素相关）
    "HEQ010": _LIVER,         # 乙型肝炎（肝酶为随访指标）
    "HEQ030": _LIVER,         # 丙型肝炎
    # DEATH_* / MCQ160* 等不由血液指标定义，无同源变量
}


def log(m):
    print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)


def build(df, label, cycles):
    """构建队列：完整案例 + 指定标签 + **剔除该终点的同源诊断变量**。"""
    d = df[df["cycle"].isin(cycles)].copy()
    d = d[analysis_mask(d)].copy()
    y = d[label]
    d = d[y.notna()].copy()
    y = d[label].astype(int)
    banned = set(LEAKAGE.get(label, []))
    cols = [c for c in FEATURE_SETS["lab_demo"]
            if c not in STRUCTURAL_EXCLUDE and c not in banned
            and c in d.columns]
    cols = [c for c in cols if d[c].isna().mean() <= 0.20]
    X = d[cols].astype(float).fillna(d[cols].astype(float).median())
    w = (d["WTMEC2YR"] / d["cycle"].nunique()).to_numpy()
    return X, y.to_numpy(), w, cols


def fit_ebm(X, y):
    from interpret.glassbox import ExplainableBoostingClassifier
    m = ExplainableBoostingClassifier(
        random_state=RANDOM_STATE, max_bins=256, interactions=10,
        outer_bags=8, learning_rate=0.03, n_jobs=-1)
    m.fit(X, y)
    return m


def imp_ebm(m, cols):
    return pd.Series(np.asarray(m.term_importances()),
                     index=list(m.term_names_)).reindex(cols).fillna(0)


def imp_shap(m, X, cols, seed=RANDOM_STATE):
    """对同一 EBM 的预测函数计算 SHAP 值。

    注意：必须显式设置 max_samples，否则 shap 会把背景集截断到 100，
    由此引入的**估计噪声会污染"分歧度"这一被测量**（把随机噪声误判为
    方法分歧）。此处使用全部背景样本。
    """
    import shap
    rng = np.random.default_rng(seed)
    n = len(X)
    n_bg = min(200, n)
    idx_bg = rng.choice(n, size=n_bg, replace=False)
    idx_ex = rng.choice(n, size=min(SHAP_SAMPLE, n), replace=False)
    f = lambda a: m.predict_proba(pd.DataFrame(a, columns=cols))[:, 1]  # noqa
    masker = shap.maskers.Independent(X.iloc[idx_bg].to_numpy(),
                                      max_samples=n_bg)
    expl = shap.Explainer(f, masker, seed=seed)
    sv = expl(X.iloc[idx_ex].to_numpy(), silent=True)
    vals = np.asarray(sv.values)
    vals = np.abs(vals).mean(axis=0) if vals.ndim == 2 \
        else np.abs(vals[:, :, 1]).mean(axis=0)
    return pd.Series(vals, index=cols)


def imp_morris(m, X, cols, seed=RANDOM_STATE):
    from SALib.sample.morris import sample as msample
    from SALib.analyze.morris import analyze as manalyze
    lo = np.percentile(X.to_numpy(), 1, axis=0)
    hi = np.percentile(X.to_numpy(), 99, axis=0)
    span = hi - lo
    span[span <= 0] = 1e-6
    hi = lo + span
    problem = {"num_vars": len(cols), "names": list(cols),
               "bounds": [[float(a), float(b)] for a, b in zip(lo, hi)]}
    Xs = msample(problem, N=N_MORRIS_TRAJ, num_levels=4, seed=seed)
    ys = m.predict_proba(pd.DataFrame(Xs, columns=cols))[:, 1]
    Si = manalyze(problem, Xs, ys, num_resamples=100, conf_level=0.95,
                  print_to_console=False, seed=seed)
    return pd.Series(np.asarray(Si["mu_star"], dtype=float), index=cols)


def disagreement(ranks: dict, topk=(5, 10)):
    """输入 {方法名: 重要性Series}，输出一致性/分歧度指标。"""
    names = list(ranks.keys())
    # 统一为「排名」：重要性越高排名越小(1)
    R = pd.DataFrame({k: v.rank(ascending=False) for k, v in ranks.items()})
    rhos = []
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            a, b = R[names[i]], R[names[j]]
            r = stats.spearmanr(a, b).correlation
            if np.isfinite(r):
                rhos.append(r)
    agree = float(np.mean(rhos)) if rhos else np.nan
    out = {"agreement_rho": agree, "disagreement": 1 - agree}
    for k in topk:
        sets = [set(s.nlargest(k).index) for s in ranks.values()]
        js = []
        for i in range(len(sets)):
            for j in range(i + 1, len(sets)):
                u = len(sets[i] | sets[j])
                js.append(len(sets[i] & sets[j]) / u if u else np.nan)
        out[f"jaccard_top{k}"] = float(np.nanmean(js)) if js else np.nan
    # 归一化排名方差
    Rz = R.div(len(R), axis=0)
    out["rank_var"] = float(Rz.var(axis=1).mean())
    return out


def analyse_endpoint(wide, ext, label, name):
    tr = build(wide, label, TRAIN)
    Xtr, ytr, wtr, cols = tr
    n_pos = int(ytr.sum())
    if n_pos < 100 or len(ytr) - n_pos < 100:
        return None

    rec = {"variable": label, "name": name, "n_train": len(ytr),
           "n_pos_train": n_pos, "prevalence": round(float(ytr.mean()), 4),
           "n_features": len(cols)}

    # 用扩展终点表取标签（宽表里没有这些列）
    # 这里 wide 已含终点列，直接使用
    m = fit_ebm(Xtr, ytr)

    # --- 三种重要性 ---
    try:
        e_ebm = imp_ebm(m, cols)
    except Exception as ex:
        log(f"    EBM 重要性失败: {ex}")
        return None
    try:
        e_shap = imp_shap(m, Xtr, cols)
    except Exception as ex:
        log(f"    SHAP 失败: {ex}")
        e_shap = None
    try:
        e_mor = imp_morris(m, Xtr, cols)
    except Exception as ex:
        log(f"    Morris 失败: {ex}")
        e_mor = None

    ranks = {"EBM": e_ebm}
    if e_shap is not None:
        ranks["SHAP"] = e_shap
    if e_mor is not None:
        ranks["Morris"] = e_mor
    rec.update(disagreement(ranks))

    # 保存三方法排名，便于后续细查
    rd = pd.DataFrame({k: v.rank(ascending=False) for k, v in ranks.items()})
    rd.index.name = "variable"
    rd.to_csv(os.path.join(REPORTS, f"ranks_{label}.csv"),
              encoding="utf-8-sig")

    # --- 外部性能 ---
    for tag, cycles in [("external_2017_2018", EXTERNAL),
                        ("external_2021_2023", EXTERNAL2)]:
        if not ext[ext["cycle"].isin(cycles)][label].notna().any():
            rec[tag + "_auc"] = np.nan
            continue
        try:
            co = build(ext, label, cycles)
            Xe, ye, we, _ = co
            # 固定使用训练集的特征列（EBM 要求列名与顺序一致）；
            # 外部队列若缺某列则补中位数，避免因缺失率差异导致列数不匹配。
            Xe = Xe.reindex(columns=cols)
            Xe = Xe.fillna(Xe.median())
            p = m.predict_proba(Xe)[:, 1]
            mm = binary_metrics(ye, p, we)
            rec[tag + "_auc"] = round(float(mm["auc"]), 4)
            rec[tag + "_n"] = int(len(ye))
        except Exception as ex:
            log(f"    {tag} 失败: {str(ex)[:100]}")
            rec[tag + "_auc"] = np.nan
    return rec


def main():
    wide = load_wide()
    ext = pd.read_parquet(EXT_PARQUET)

    inv = pd.read_csv(os.path.join(REPORTS, "endpoints_inventory.csv"))
    usable = inv[inv.usable][["variable", "name"]].values.tolist()
    log(f"待分析终点: {len(usable)}")

    # 把终点列并入宽表（按 SEQN+cycle）
    # 注意：原宽表里已含 7 个疾病变量（且是未重编码的 1/2 原始编码），
    # 直接 merge 会产生 _x/_y 后缀冲突。因此先丢弃宽表中的同名列，
    # 统一使用扩展表里已重编码为 1/0 的版本。
    overlap = [c for c in ext.columns if c in wide.columns
               and c not in ("SEQN", "cycle")]
    if overlap:
        log(f"  丢弃宽表中的同名列 {len(overlap)} 个: {overlap}")
        wide = wide.drop(columns=overlap)
    merged = wide.merge(ext, on=["SEQN", "cycle"], how="left")
    log(f"合并终点后: {merged.shape}")

    results = []
    for label, name in usable:
        t0 = time.time()
        log(f"  {name} ({label}) ...")
        r = analyse_endpoint(merged, merged, label, name)
        if r:
            r["elapsed_sec"] = round(time.time() - t0, 1)
            results.append(r)
            log(f"    分歧度={r['disagreement']:.3f}  "
                f"agreement_rho={r['agreement_rho']:.3f}  "
                f"外部AUC={r.get('external_2017_2018_auc')}  "
                f"({r['elapsed_sec']}s)")
        # 增量保存
        pd.DataFrame(results).to_csv(
            os.path.join(REPORTS, "disagreement_validation.csv"),
            index=False, encoding="utf-8-sig")

    df = pd.DataFrame(results)
    log(f"完成 {len(df)} 个终点")

    # --- 统计检验 ---
    log("")
    log("=" * 62)
    log("假设检验：分歧度 vs 外部性能")
    log("=" * 62)
    for perf in ["external_2017_2018_auc", "external_2021_2023_auc"]:
        if perf not in df.columns:
            continue
        for dis in ["disagreement", "jaccard_top5", "rank_var"]:
            d = df[[dis, perf]].dropna()
            if len(d) < 5:
                continue
            rho, p = stats.spearmanr(d[dis], d[perf])
            log(f"  {perf} ~ {dis:<14} n={len(d):>3} "
                f"Spearman ρ={rho:+.3f} p={p:.4f} "
                f"{'✅ 显著' if p < 0.05 else ''}")
    log("")
    # 控制患病率的偏相关
    log("控制患病率后的偏相关（外部 2017-2018）:")
    d = df[["disagreement", "external_2017_2018_auc", "prevalence"]].dropna()
    if len(d) >= 8:
        import numpy.linalg as la

        def resid(a, b):
            A = np.column_stack([np.ones(len(b)), b])
            coef, *_ = la.lstsq(A, a, rcond=None)
            return a - A @ coef
        ra = resid(d["disagreement"].to_numpy(), d["prevalence"].to_numpy())
        rb = resid(d["external_2017_2018_auc"].to_numpy(),
                   d["prevalence"].to_numpy())
        rho, p = stats.spearmanr(ra, rb)
        log(f"  ρ_partial={rho:+.3f} p={p:.4f} (n={len(d)})")

    df.to_csv(os.path.join(REPORTS, "disagreement_validation.csv"),
              index=False, encoding="utf-8-sig")
    log(f"结果 -> reports/disagreement_validation.csv")


if __name__ == "__main__":
    main()
