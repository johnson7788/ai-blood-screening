"""推理入口 —— 加载已训练模型并对新数据打分。

模型是以自定义类 `StackingEnsemble`（定义于 `src/models.py`）序列化的，
反序列化时需要该模块可被导入。本模块统一处理路径问题，使用者无需关心。

用法（命令行）：
    python predict.py --disease diabetes --variant rfe12 --input new.csv --output preds.csv

用法（Python）：
    from predict import load_model, predict_frame
    model, spec = load_model("diabetes", "rfe12")
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


def load_metadata():
    with open(METADATA, encoding="utf-8") as f:
        return json.load(f)


def load_model(disease, variant="rfe12"):
    """加载指定病种与版本的模型。

    disease: diabetes / hypertension / cancer / chd / mi / stroke / ckd
    variant: 'rfe12'（12 特征，推荐）或 'full'（全特征）
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
    return model, spec


def predict_frame(model, spec, df):
    """对 DataFrame 打分，返回阳性概率数组。

    缺失的特征列会以训练时的中位数不可得，故直接报错提示；
    行内缺失值以该列中位数填补（与训练时的插补策略一致）。
    """
    missing = [c for c in spec["features"] if c not in df.columns]
    if missing:
        raise KeyError(f"输入缺少 {len(missing)} 个必需特征：{missing}")
    X = df[spec["features"]].astype(float)
    X = X.fillna(X.median())
    return model.predict_proba(X)[:, 1]


def main():
    ap = argparse.ArgumentParser(description="NHANES 血检筛查模型推理")
    ap.add_argument("--disease", required=True,
                    help="diabetes/hypertension/cancer/chd/mi/stroke/ckd")
    ap.add_argument("--variant", default="rfe12", choices=["rfe12", "full"])
    ap.add_argument("--input", required=True, help="输入 CSV/Parquet 路径")
    ap.add_argument("--output", help="输出 CSV 路径")
    ap.add_argument("--list", action="store_true", help="仅列出所有模型")
    args = ap.parse_args()

    if args.list:
        meta = load_metadata()
        print(f"{'病种':<14}{'版本':<8}{'特征数':>7}{'外部AUC':>10}{'阈值':>9}  文件")
        for e in meta["models"]:
            for v, s in e["models"].items():
                print(f"{e['name']:<14}{v:<8}{s['n_features']:>7}"
                      f"{str(s.get('external_auc')):>10}{s['threshold']:>9}  {s['file']}")
        return

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

    if args.output:
        out = df.copy()
        out["probability"] = prob
        out["prediction"] = pred
        out.to_csv(args.output, index=False, encoding="utf-8-sig")
        print(f"已写出    : {args.output}")


if __name__ == "__main__":
    main()
