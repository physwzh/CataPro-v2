# CataPro-v2

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

方法与各微调版本的运行方式见 **[`finetune/README_zh.md`](finetune/README_zh.md)**。

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

> 本次发布中，`models/prot_t5_xl_uniref50` 与 `models/molt5-base-smiles2caption` 是**唯一**两个
> 未作为项目文件随附的条目——请从上面的链接下载权重并放到对应位置。仓库中其余文件都是真实文件，
> 可直接运行。

---

## 2. 仓库结构

```
CataPro-v2/
  README.md                     # 英文说明
  README_zh.md                  # 本文件（中文说明）
  LICENSE
  datasets/                     # 0.4simi-10fold 原始数据集（kcat / Km / kcat-over-Km）
  datasets_finetune/            # 用于微调的、按靶点限制的子集（见 finetune/）
  models/
    kcat_models/ Km_models/     # 已发布的分折 CataPro 基座模型（冻结的绝对通路）
    act_models/                 # 已发布的分折 CataPro kcat/Km 融合模型
    fig1_architecture.png
    prot_t5_xl_uniref50/        # <- 需下载（链接见上）
    molt5-base-smiles2caption/  # <- 需下载（链接见上）
  samples/sample_inp.csv        # 推理输入示例
  inference/
    predict.py                  # 原始 CataPro 推理（10 折平均的基座模型）
    predict_v2.py               # CataPro-v2 推理（微调后的 delta 模型）
    run_catapro.sh              # predict.py 的封装脚本
    model.py  act_model.py  utils.py  generate_features.py
  finetune/
    README.md                   # 英文说明：方法 + 各微调版本的运行方式
    README_zh.md                # 中文说明
    train/                      # 微调代码包（data / model / losses / select / cv）
```

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

### 3.2 运行基座 CataPro 推理（10 折平均）

```
cd inference
python predict.py \
    -inp_fpath ../samples/sample_inp.csv \
    -model_dpath ../models \
    -batch_size 64 \
    -device cuda:0 \
    -out_fpath catapro_prediction.csv
```

或者简单地

```
cd inference && bash run_catapro.sh      # 写出 catapro_test-pred.csv
```

结果写入 `-out_fpath` 指定的 CSV，列名为
`fasta_id, smiles, pred_log10[kcat(s^-1)], pred_log10[Km(mM)], pred_log10[kcat/Km(s^-1mM^-1)]`。

### 3.3 运行 CataPro-v2 微调推理

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
`delta = 0`，即冻结的 CataPro 基座模型输出。这与模型的训练和评估方式一致（训练按 `UniProtID` 与
`Substrate` 分组）。

---

## 4. 微调

微调流程见 **[`finetune/README_zh.md`](finetune/README_zh.md)**
（English: [`finetune/README.md`](finetune/README.md)）。

---

## 联系

wangzh336699@gmail.com
