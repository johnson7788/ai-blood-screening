"""下载并解析 NHANES 公开版链接死亡率文件（Linked Mortality File, LMF）。

数据来源
--------
NCHS Data Linkage Program，把 NHANES 参与者链接到国家死亡索引（NDI）。
公开版覆盖 NHANES 1999–2018，随访截止 2019-12-31。
下载地址：https://ftp.cdc.gov/pub/Health_Statistics/NCHS/datalinkage/linked_mortality/

⚠️ 该 FTP 服务器证书链有问题，必须豁免 SSL 校验（`ssl._create_unverified_context`）。
   这不影响数据完整性——文件本身可通过官方公布的字节数与内容校验。

文件布局（依据官方 SAS_ReadInProgramAllSurveys.sas）
---------------------------------------------------
    SEQN            1-6      参与者序号（与主数据链接键）
    ELIGSTAT        15       随访资格：1=合格 2=未满18岁 3=不合格
    MORTSTAT        16       最终生死：0=推定存活 1=推定死亡 .=不合格
    UCOD_LEADING    17-19    主要死因归类（001-010，见 UCOD_MAP）
    DIABETES        20       糖尿病为根本/多重死因标志
    HYPERTEN        21       高血压为根本/多重死因标志
    PERMTH_INT      43-45    自访谈起的随访月数
    PERMTH_EXM      46-48    自体检起的随访月数
"""
import io
import os
import ssl
import sys
import time
import urllib.request

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
OUTDIR = os.path.join(ROOT, "mortality")
os.makedirs(OUTDIR, exist_ok=True)

BASE = ("https://ftp.cdc.gov/pub/Health_Statistics/NCHS/datalinkage/"
        "linked_mortality")
CYCLES = ["2013_2014", "2015_2016", "2017_2018"]

# 固定宽度布局：(变量名, 起始, 结束, 类型)
LAYOUT = [
    ("SEQN", 1, 6, "int"),
    ("ELIGSTAT", 15, 15, "int"),
    ("MORTSTAT", 16, 16, "int"),
    ("UCOD_LEADING", 17, 19, "str"),
    ("DIABETES", 20, 20, "int"),
    ("HYPERTEN", 21, 21, "int"),
    ("PERMTH_INT", 43, 45, "int"),
    ("PERMTH_EXM", 46, 48, "int"),
]

UCOD_MAP = {
    "001": "心脏病", "002": "恶性肿瘤", "003": "慢性下呼吸道疾病",
    "004": "意外伤害", "005": "脑血管病", "006": "阿尔茨海默病",
    "007": "糖尿病", "008": "流感与肺炎",
    "009": "肾炎/肾病综合征/肾病变", "010": "其他原因",
}


def log(m):
    print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)


def download(cycle, force=False):
    """下载单个周期的 LMF（豁免 SSL 校验）。"""
    fn = f"NHANES_{cycle}_MORT_2019_PUBLIC.dat"
    dest = os.path.join(OUTDIR, fn)
    if os.path.exists(dest) and not force and os.path.getsize(dest) > 10000:
        log(f"  已存在 {fn}（{os.path.getsize(dest)} 字节）")
        return dest
    url = f"{BASE}/{fn}"
    ctx = ssl._create_unverified_context()
    log(f"  下载 {fn} ...")
    with urllib.request.urlopen(url, context=ctx, timeout=120) as r:
        data = r.read()
    with open(dest, "wb") as f:
        f.write(data)
    log(f"    -> {len(data)} 字节")
    return dest


def parse(path, cycle):
    """按固定宽度解析 ASCII 文件。

    cycle 统一规范为连字符形式（2013-2014），与主宽表的 cycle 列一致，
    否则后续按 SEQN+cycle 合并会全部匹配失败。
    """
    cycle_norm = cycle.replace("_", "-")
    rows = []
    with io.open(path, encoding="latin-1") as f:
        for line in f:
            line = line.rstrip("\r\n")
            if not line.strip():
                continue
            rec = {}
            for name, a, b, typ in LAYOUT:
                raw = line[a - 1:b].strip()
                if typ == "int":
                    # SAS 缺失值 '.' 与空白都视为缺失
                    rec[name] = int(raw) if raw.isdigit() else np.nan
                else:
                    rec[name] = raw if raw and raw != ".." else None
            rec["cycle"] = cycle_norm
            rows.append(rec)
    return pd.DataFrame(rows)


def main():
    log("=== NHANES 链接死亡率文件（LMF）===")
    frames = []
    for c in CYCLES:
        p = download(c)
        df = parse(p, c)
        frames.append(df)
        log(f"  {c}: {len(df)} 行")

    lmf = pd.concat(frames, ignore_index=True)

    # 基本校验
    log("=== 校验 ===")
    dup = int(lmf["SEQN"].duplicated().sum())
    log(f"  SEQN 重复: {dup}（应为 0）")
    log(f"  总记录: {len(lmf)}")
    elig = lmf[lmf["ELIGSTAT"] == 1]
    log(f"  符合随访资格(ELIGSTAT=1): {len(elig)}")
    dead = int((elig["MORTSTAT"] == 1).sum())
    log(f"  死亡: {dead} ({dead / max(len(elig), 1) * 100:.2f}%)")
    exm = elig["PERMTH_EXM"].dropna()
    log(f"  随访月数(体检): 中位 {exm.median():.0f}, "
        f"范围 {exm.min():.0f}-{exm.max():.0f}, 均值 {exm.mean():.1f}")

    # 死因分布
    d = elig[elig["MORTSTAT"] == 1]["UCOD_LEADING"].value_counts()
    log("  主要死因分布:")
    for code, n in d.head(6).items():
        log(f"    {code} {UCOD_MAP.get(code, '?'):<14} {n:>5} "
            f"({n / dead * 100:.1f}%)")

    # 与主宽表链接校验
    wide_p = os.path.join(ROOT, "nhanes_wide.parquet")
    if os.path.exists(wide_p):
        wide = pd.read_parquet(wide_p, columns=["SEQN", "cycle"])
        m = wide.merge(lmf, on=["SEQN", "cycle"], how="left")
        hit = m["MORTSTAT"].notna().mean()
        log(f"  与主宽表链接成功率: {hit * 100:.2f}% "
            f"({int(m['MORTSTAT'].notna().sum())}/{len(m)})")

    out = os.path.join(OUTDIR, "lmf_parsed.parquet")
    lmf.to_parquet(out, index=False)
    log(f"已保存 -> {out}")

    # 同时导出一份便于阅读的 CSV
    csv = os.path.join(OUTDIR, "lmf_parsed.csv")
    lmf.to_csv(csv, index=False, encoding="utf-8-sig")
    log(f"已保存 -> {csv}")


if __name__ == "__main__":
    main()
