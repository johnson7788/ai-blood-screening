# 数据许可与来源声明

## 一、原始数据

本项目的原始数据来自 **美国疾病控制与预防中心（CDC）/ 国家健康与营养调查（NHANES）**。

- 来源：<https://wwwn.cdc.gov/nchs/nhanes/>
- 法律状态：NHANES 数据由美国联邦政府机构制作，依 **17 U.S.C. §105** 属于**公共领域（public domain）**，不受版权保护。
- 分发政策：NCHS 的政策即通过公开数据文件向科学界提供数据，**无需注册、无需伦理审批、无需数据使用协议（DUA）**。
- 下载脚本：见 `download_nhanes.sh`，可从 CDC 官方地址直接获取全部原始文件。

**使用 NHANES 数据时请按 CDC 规范引用**：
<https://wwwn.cdc.gov/nchs/nhanes/NhanesCitation.aspx>

建议引用格式：

> National Center for Health Statistics. National Health and Nutrition Examination Survey.
> Hyattsville, MD: U.S. Department of Health and Human Services, Centers for Disease Control
> and Prevention. Available from: https://wwwn.cdc.gov/nchs/nhanes/

## 二、本项目产生的衍生数据

`nhanes_wide.parquet`、`availability_report.txt`、`manifest.tsv` 以及 `reports/` 下的
全部结果文件（CSV / JSON / 图表）由本项目基于 NHANES 公共领域数据生成，
采用 **Creative Commons Attribution 4.0 International (CC BY 4.0)** 许可发布。

- 许可全文：<https://creativecommons.org/licenses/by/4.0/legalcode>

您可以自由共享与改编，但须注明出处。

## 三、代码

`src/`、`download_nhanes.sh`、`build_dataset.py` 采用 **MIT License** 发布，
详见 `LICENSE`。

## 四、模型权重

`artifacts/models/` 下的模型由本项目训练产生，随代码一并采用 **MIT License** 发布。

## 五、未使用的受限数据（重要声明）

本项目的复现对象涉及若干篇已发表论文，这些论文的部分数据集需要申请或受数据使用协议
（DUA）约束。**本项目未使用其中任何受限数据**，具体包括：

| 数据集 | 归属 | 本项目是否使用 |
|---|---|---|
| HARMONY Alliance 白血病数据 | 欧盟 HARMONY | ❌ 未使用 |
| COMPASS 队列 | 温州医科大学附属第一医院等 | ❌ 未使用 |
| WEAR-ME 队列 | Google Research | ❌ 未使用 |
| UK Biobank | UK Biobank Ltd | ❌ 未使用 |
| ADNI | USC Laboratory of Neuro Imaging | ❌ 未使用 |
| AURORA 多组学队列 | 北京大学 | ❌ 未使用 |
| 任何医院电子病历数据 | — | ❌ 未使用 |

本项目**也未通过任何非官方镜像**获取受 DUA 约束的数据。全部建模数据仅来自
上述 NHANES 公开数据文件。

## 六、第三方依赖许可

| 依赖 | 许可 |
|---|---|
| scikit-learn | BSD-3-Clause |
| XGBoost | Apache-2.0 |
| SHAP | MIT |
| pandas | BSD-3-Clause |
| NumPy | BSD-3-Clause |
| Matplotlib | PSF-based |
| joblib | BSD-3-Clause |
| pyarrow | Apache-2.0 |

## 七、免责声明

本项目为**方法学复现研究**，模型输出**不构成任何医疗建议**，不得用于临床诊断或
治疗决策。研究报告中的性能指标基于美国人群的公开调查数据，不应外推为其他人群或
其他数据源上的预期表现。
