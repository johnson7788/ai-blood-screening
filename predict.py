"""推理入口 —— 加载已训练模型并对新数据打分。

支持两类模型：
  - **堆叠集成**（variant = `rfe12` / `full`）：8 基学习器 + LR 元学习器。
    以自定义类 `StackingEnsemble`（定义于 `src/models.py`）序列化，
    反序列化时需要该模块可被导入 —— 本模块统一处理路径问题。
  - **EBM**（variant = `ebm`）：完全可加性玻璃盒模型，需要 `interpret` 包。

用法（命令行）：
    python predict.py --list --disease diabetes --input data.csv
    python predict.py --disease diabetes --variant ebm --input new.csv --output preds.csv

用法（Python）：
    from predict import load_model, predict_frame
    model, spec = load_model("diabetes", "ebm")
    prob = predict_frame(model, spec, df)
"""
import argparse
import json
import os
import sys

import joblib
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.join(HERE, "src")
if SRC not in sys.path:
    sys.path.insert(0, SRC)          # 使 pickle 能解析 models.StackingEnsemble

MODELDIR = os.path.join(HERE, "artifacts", "models")
METADATA = os.path.join(MODELDIR, "metadata.json")


class _EBMWrapper:
    """EBM 需要 DataFrame 输入（训练时用了列名），此处做适配。"""

    def __init__(self, model, features):
        self.model = model
        self.features = list(features)

    def predict_proba_pos(self, X):
        Xd = pd.DataFrame(np.asarray(X, dtype=float), columns=self.features)
        return self.model.predict_proba(Xd)[:, 1]


def load_metadata():
    with open(METADATA, encoding="utf-8") as f:
        return json.load(f)


def available_variants(disease=None):
    meta = load_metadata()
    out = {}
    for e in meta["models"]:
        if disease and e["disease"] != disease:
            continue
        out[e["disease"]] = list(e["models"].keys())
    return out


def load_model(disease, variant="rfe12"):
    """加载指定病种与版本的模型。

    disease: diabetes / hypertension / cancer / chd / mi / stroke / ckd
    variant: 'rfe12'（12 特征，推荐）| 'full'（全特征）| 'ebm'（可解释模型）
    返回 (model, spec)；spec 含特征列表、决策阈值与指标。
    """
    meta = load_metadata()
    entry = next((e for e in meta["models"] if e["disease"] == disease), None)
    if entry is None:
        avail = [e["disease"] for e in meta["models"]]
        raise KeyError(f"未知病种 '{disease}'，可选：{avail}")
    if variant not in entry["models"]:
        raise KeyError(f"病种 '{disease}' 无 '{variant}' 版本，"
                       f"可选：{list(entry['models'])}")
    spec = entry["models"][variant]
    model = joblib.load(os.path.join(MODELDIR, spec["file"]))
    if variant == "ebm":
        model = _EBMWrapper(model, spec["features"])
    return model, spec


def _positive_proba(model, X):
    """统一取阳性类概率：兼容 predict_proba_pos 与 sklearn 的 predict_proba。"""
    if hasattr(model, "predict_proba_pos"):
        return np.asarray(model.predict_proba_pos(X), dtype=float)
    return np.asarray(model.predict_proba(X), dtype=float)[:, 1]


def predict_frame(model, spec, df):
    """对 DataFrame 打分，返回阳性概率数组。

    行内缺失值以该列中位数填补（与训练时的插补策略一致）。
    """
    missing = [c for c in spec["features"] if c not in df.columns]
    if missing:
        raise KeyError(f"输入缺少 {len(missing)} 个必需特征：{missing}")
    X = df[spec["features"]].astype(float)
    X = X.fillna(X.median())
    return _positive_proba(model, X.to_numpy(dtype=float))


def main():
    ap = argparse.ArgumentParser(description="NHANES 血检筛查模型推理")
    ap.add_argument("--disease")
    ap.add_argument("--variant", default="rfe12")
    ap.add_argument("--input")
    ap.add_argument("--output")
    ap.add_argument("--list", action="store_true", help="列出所有模型")
    args = ap.parse_args()

    if args.list or not args.disease:
        meta = load_metadata()
        fam = meta.get("model_families", {})
        print(f"{'病种':<14}{'版本':<8}{'特征数':>7}{'外部AUC':>10}{'阈值':>10}  文件")
        for e in meta["models"]:
            for v, s in e["models"].items():
                print(f"{e['name']:<14}{v:<8}{s['n_features']:>7}"
                      f"{str(s.get('external_auc')):>10}"
                      f"{s['threshold']:>10}  {s['file']}")
        if fam:
            print("\n模型族说明：")
            for k, v in fam.items():
                print(f"  {k}: {v}")
        return

    if not args.input:
        ap.error("--input 为必填（或使用 --list）")

    model, spec = load_model(args.disease, args.variant)
    df = (pd.read_parquet(args.input) if args.input.endswith(".parquet")
          else pd.read_csv(args.input))
    prob = predict_frame(model, spec, df)
    pred = (prob >= spec["threshold"]).astype(int)

    print(f"病种      : {args.disease} ({args.variant})")
    print(f"特征数    : {spec['n_features']}")
    print(f"决策阈值  : {spec['threshold']}")
    print(f"样本数    : {len(prob)}")
    print(f"预测阳性  : {int(pred.sum())} ({pred.mean() * 100:.1f}%)")
    if spec.get("interpretability"):
        print(f"可解释性  : {spec['interpretability']}")

    if args.output:
        out = df.copy()
        out["probability"] = prob
        out["prediction"] = pred
        out.to_csv(args.output, index=False, encoding="utf-8-sig")
        print(f"已写出    : {args.output}")


if __name__ == "__main__":
    main()
