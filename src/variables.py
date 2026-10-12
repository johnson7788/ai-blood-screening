"""变量词典、疾病标签定义，以及「标签—禁用变量」对照表。

本模块的核心目的是**防止标签泄漏**：
论文1 的眼病诊断来自眼科专项检查，与血检相互独立；
而 NHANES 的部分疾病诊断本身就基于血液检验（如糖尿病依据血糖/HbA1c、
慢性肾病依据肌酐），若把同源指标当特征，AUC 会被系统性抬高。
"""

# ---------------------------------------------------------------------------
# 候选疾病标签（NHANES 问卷自报确诊，编码 1=是 2=否 7=拒答 9=不知道）
# ---------------------------------------------------------------------------
DISEASES = {
    "hypertension": {"var": "BPQ020", "name": "高血压"},
    "diabetes": {"var": "DIQ010", "name": "糖尿病"},
    "cancer": {"var": "MCQ220", "name": "癌症"},
    "chd": {"var": "MCQ160C", "name": "冠心病"},
    "mi": {"var": "MCQ160E", "name": "心梗"},
    "stroke": {"var": "MCQ160F", "name": "中风"},
    "ckd": {"var": "KIQ022", "name": "慢性肾病"},
}
# 备选第 8 病种（用于扩充多分类）
DISEASES_EXTRA = {"chf": {"var": "MCQ160B", "name": "充血性心衰"}}

# ---------------------------------------------------------------------------
# 标签—禁用变量对照表（防标签泄漏）
#
# 原则：凡进入该疾病**临床诊断标准**的血液指标，一律从特征中剔除。
#   1. 糖尿病      : 空腹血糖、生化血糖、糖化血红蛋白、胰岛素 —— 直接诊断依据
#   2. 慢性肾病    : 肌酐、尿素氮（eGFR 由肌酐推算）、尿酸 —— 直接诊断依据
#   3. 高血压      : 诊断依据为血压测量，非血液指标 —— 无剔除
#   4. 冠心病/心梗 : 诊断依据为症状+心电图+影像，血脂仅为危险因素 —— 无剔除
#   5. 中风        : 诊断依据为神经影像 —— 无剔除
#   6. 癌症        : 主要依据病理 —— 无剔除
#
# 注意：「危险因素」不等于「诊断依据」。血脂之于冠心病是前者，故保留；
# 血糖之于糖尿病是后者，故剔除。这个区分是本复现可信度的关键。
# ---------------------------------------------------------------------------
LEAKAGE_EXCLUSIONS = {
    "diabetes": ["LBXSGL", "LBXGLU", "LBXGH", "LBXIN"],
    "ckd": ["LBXSCR", "LBXSBU", "LBXSUA"],
    "hypertension": [],
    "chd": [],
    "mi": [],
    "stroke": [],
    "cancer": [],
    "chf": [],
}

# 各剔除变量的中文名（用于报告）
EXCLUSION_REASONS = {
    "LBXSGL": "生化血糖，糖尿病诊断依据",
    "LBXGLU": "空腹血糖，糖尿病诊断依据",
    "LBXGH": "糖化血红蛋白，糖尿病诊断依据",
    "LBXIN": "空腹胰岛素，糖尿病诊断依据",
    "LBXSCR": "肌酐，eGFR/肾病诊断依据",
    "LBXSBU": "尿素氮，肾功能评估依据",
    "LBXSUA": "尿酸，肾功能相关",
}

# ---------------------------------------------------------------------------
# 特征池
# ---------------------------------------------------------------------------
# 血常规 CBC（论文1 最核心的发现来源）
CBC_FEATURES = [
    "LBXWBCSI", "LBXLYPCT", "LBXMOPCT", "LBXNEPCT", "LBXEOPCT", "LBXBAPCT",
    "LBXRBCSI", "LBXHGB", "LBXHCT", "LBXMCVSI", "LBXMCHSI", "LBXMC",
    "LBXPLTSI", "LBXRDW",
]
# 生化 BIOPRO
BIOPRO_FEATURES = [
    "LBXSCR", "LBXSUA", "LBXSBU", "LBXSATSI", "LBXSASSI", "LBXSTB",
    "LBXSAL", "LBXSCA", "LBXSNASI", "LBXSKSI",
    "LBXSGL", "LBXSCH", "LBXSTR",
]
# 血脂 / 血糖 / 炎症
LIPID_FEATURES = [
    "LBXGH", "LBDHDD", "LBDLDL", "LBDTRSI", "LBXTC", "LBXHSCRP",
    "LBXIN", "LBXGLU",
]
# 人口学与体格
DEMO_FEATURES = ["RIDAGEYR", "RIAGENDR", "BMXBMI"]
EXTRA_DEMO = ["BMXWAIST", "INDFMPIR"]

# 三类特征集，用于消融
FEATURE_SETS = {
    "lab": CBC_FEATURES + BIOPRO_FEATURES + LIPID_FEATURES,
    "lab_demo": CBC_FEATURES + BIOPRO_FEATURES + LIPID_FEATURES + DEMO_FEATURES,
    "cbc_only": CBC_FEATURES,
}


def feature_columns(feature_set="lab", disease=None):
    """返回指定特征集、并已剔除该病种泄漏变量后的特征列表。"""
    cols = list(FEATURE_SETS[feature_set])
    if disease:
        banned = set(LEAKAGE_EXCLUSIONS.get(disease, []))
        cols = [c for c in cols if c not in banned]
    return cols


def var_label(var):
    """变量中文名，用于报告。"""
    return VAR_LABELS.get(var, var)


VAR_LABELS = {
    "LBXWBCSI": "白细胞", "LBXLYPCT": "淋巴细胞%", "LBXMOPCT": "单核细胞%",
    "LBXNEPCT": "中性粒细胞%", "LBXEOPCT": "嗜酸细胞%", "LBXBAPCT": "嗜碱细胞%",
    "LBXRBCSI": "红细胞", "LBXHGB": "血红蛋白", "LBXHCT": "红细胞压积",
    "LBXMCVSI": "MCV", "LBXMCHSI": "MCH", "LBXMC": "MCHC",
    "LBXPLTSI": "血小板", "LBXRDW": "RDW",
    "LBXSCR": "肌酐", "LBXSUA": "尿酸", "LBXSBU": "尿素氮",
    "LBXSATSI": "ALT", "LBXSASSI": "AST", "LBXSTB": "总胆红素",
    "LBXSAL": "白蛋白", "LBXSCA": "钙", "LBXSNASI": "钠", "LBXSKSI": "钾",
    "LBXSGL": "生化血糖", "LBXSCH": "生化总胆固醇", "LBXSTR": "生化甘油三酯",
    "LBXGH": "糖化血红蛋白", "LBDHDD": "HDL", "LBDLDL": "LDL",
    "LBDTRSI": "甘油三酯", "LBXTC": "总胆固醇", "LBXHSCRP": "C反应蛋白",
    "LBXIN": "空腹胰岛素", "LBXGLU": "空腹血糖",
    "RIDAGEYR": "年龄", "RIAGENDR": "性别", "BMXBMI": "BMI",
    "BMXWAIST": "腰围", "INDFMPIR": "收入贫困比",
    "BPQ020": "高血压", "DIQ010": "糖尿病", "MCQ220": "癌症",
    "MCQ160C": "冠心病", "MCQ160E": "心梗", "MCQ160F": "中风",
    "KIQ022": "慢性肾病", "MCQ160B": "充血性心衰",
}
