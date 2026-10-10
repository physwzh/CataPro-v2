# CataPro-v2 — 推理

[English](README.md) | **中文**

CataPro-v2 是 [CataPro](https://github.com/zchwang/CataPro) 的**微调**版本。它保留 CataPro 冻结的绝对
预测通路，并新增一条**可训练的突变效应残差通路**，使得对于同一酶/底物对的一组突变体，模型能够按测得的
动力学效应给它们排序——同时 kcat、Km 与 kcat/Km 仍共用**同一套架构**。

预测酶动力学参数——转化数（kcat）、米氏常数（Km）与催化效率（kcat/Km）——是酶发现与工程的核心。
CataPro-v2 建立在与 CataPro 相同的、无偏的 0.4 序列相似度 10 折交叉验证数据集之上（采集自 BRENDA 与
SABIO-RK），但与"每折一个预测头"不同，它是在已发布的分折基座模型之上微调一个*以靶点为中心*的残差。

对于共享同一个（酶，底物）组的一批行，

```
out = out_abs(x)  +  scale · delta(x − group_mean(x))
```

其中 `out_abs` 是冻结的 CataPro 预测（初始化时逐位一致），`delta` 是**有界**的突变效应项
（`tanh`，故 ∈ (−1, 1)），`scale = delta_max · sigmoid(gate)`（同样有界）。仅含单行的组
`delta = 0`，此时预测回退到已发布的 CataPro 基座模型。

<img src="models/fig1_architecture.png">

> **本仓库只包含 CataPro-v2 的推理代码与规范微调模型权重。** 微调流程（训练代码、超参数、交叉验证
> 以及一轴消融变体）不在本次发布之列；随附的只有 CataPro-v2 手稿中写明的模型族——
> `dual_residual_stabilized__composite_hinge`，且运行推理所需的全部权重都是自包含的（冻结的绝对通路
> 已存于各 `model.pth` 内部）。

---

## 1. 创建 CataPro-v2 环境

运行 CataPro-v2 需要创建一个包含以下包的 conda 环境：

```
pytorch >= 1.13.0
transformers
numpy
pandas
RDKit
```

此外，CataPro-v2 依赖与 CataPro 相同的两个预训练编码器，分别用于从酶和底物中提取特征。它们
**需要你自行下载**并放到 `models/` 下：

| 模型 | 用途 | 放置位置 |
|---|---|---|
| [prot_t5_xl_uniref50](https://huggingface.co/Rostlab/prot_t5_xl_uniref50) | 酶序列编码器（1024 维） | `models/prot_t5_xl_uniref50/` |
| [molt5-base-smiles2caption](https://huggingface.co/laituan245/molt5-base-smiles2caption) | 底物 SMILES 编码器（768 维） | `models/molt5-base-smiles2caption/` |

> `models/prot_t5_xl_uniref50` 与 `models/molt5-base-smiles2caption` 是**唯一**两个未随附的条目——
> 请从上面的链接下载权重并放到对应位置。仓库中其余文件都是真实文件，可直接运行。

---

## 2. 仓库结构

```
CataPro-v2/
  README.md                     # 英文说明
  README_zh.md                  # 本文件（中文说明）
  LICENSE
  models/
    fig1_architecture.png
    prot_t5_xl_uniref50/        # <- 需下载（链接见上）
    molt5-base-smiles2caption/  # <- 需下载（链接见上）
  samples/sample_inp.csv        # 推理输入示例
  inference/
    predict_v2.py               # CataPro-v2 推理（微调后的 delta 模型）
    utils.py                    # 特征提取（ProtT5 / MolT5 / MACCS）
  finetune/train/
    common/
      data.py                   # N_FEAT、组均值 delta 上下文、predict_array
      model.py                  # DualStreamRegressor / ActivityModel 定义
      eval_model.py             # load_model_for_run（读取 run_config.json + model.pth）
    kcat/dual_residual_stabilized__composite_hinge/          # 规范 kcat 权重
    Km/dual_residual_stabilized__composite_hinge/            # 规范 Km 权重
    kcat_over_Km/dual_residual_stabilized__composite_hinge/  # 规范 kcat/Km 权重
```

每个规范的 `.../dual_residual_stabilized__composite_hinge/` 目录只含推理所需的两个文件：
`model.pth`（微调权重，已含冻结的绝对通路）与 `run_config.json`（重建架构所用的开关）。

---

## 3. 使用

### 3.1 准备输入文件

酶与底物信息应组织成一个 pandas DataFrame（CSV）。每行必须包含 `Enzyme_id`（UniProt 登录号）、
`type`（野生型写 `wild`，突变体写突变本身，如 `V181I`）、酶的 `sequence` 以及底物的 `smiles`：

|   Enzyme_id   |   type    |   sequence    |   smiles  |
|---------------|-----------|---------------|-----------|
|   P10632  |   wild    |   MEPFVVLVLCLSFMLLFSLWRQSCRR...    |   CCCCCC=CCC=CCC=CCC=CCCCC(=O)O  |
|   P10632  |   V181I   |   MEPFVVLVLCLSFMLLFSLWRQSCRR...    |   CCCCCC=CCC=CCC=CCC=CCCCC(=O)O  |
|   P10632  |   I331T   |   MEPFVVLVLCLSFMLLFSLWRQSCRR...    |   CCCCCC=CCC=CCC=CCC=CCCCC(=O)O  |

`samples/sample_inp.csv` 完整给出该组——细胞色素 P450 2C8（`P10632`）在花生四烯酸上的四个单点突变
（`V181I`、`I331T`、`I244V`、`L361F`）。同一个酶、多个突变、共享同一个（酶，底物）组，故以靶点为中心的
delta 通路被激活；它取自外部突变测试集（*KcatTIE*）中排序效果较好的一组，CataPro-v2 在该组五个不同
变体上的按靶点 PCC 由 0.61 提升到 0.95（SCC 1.00）。

### 3.2 运行 CataPro-v2 推理

`inference/predict_v2.py` 从 `finetune/train/` 载入规范微调模型
（`dual_residual_stabilized__composite_hinge`），并施加以靶点为中心的 delta 通路：

```
python inference/predict_v2.py \
    -inp_fpath samples/sample_inp.csv \
    -out_fpath catapro_v2_predict_score.csv \
    -model_dpath models \
    -device cuda:0
```

输入列与上面相同；输出与 CataPro 预测文件同构：

```
fasta_id, smiles, pred_log10[kcat(s^-1)], pred_log10[Km(M)], pred_log10[kcat/Km(s^-1 M^-1)]
```

**关于分组的说明。** delta 项以由 `Enzyme_id` 与 `smiles` 构成的（酶，底物）组为中心（因推理输入中
没有底物名称，故以底物的结构即 SMILES 作为其标识）。只有当一组包含**至少两行**时它才有意义；单行的组
`delta = 0`，即冻结的 CataPro 基座模型输出。这与模型的训练和评估方式一致。

---

## 联系

wangzh336699@gmail.com
