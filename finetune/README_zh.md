# CataPro-v2 微调

[English](README.md) | **中文**

微调 CataPro-v2 模型（kcat、Km、kcat/Km —— 共用一套架构）。所有命令均在 `finetune/train/` 下运行，
无需调度器 —— `run.py` 选取 `configs/variants.py` 中定义的 run，对每个 run 直接调用训练入口
`python -m common.train`。

**前置条件**：`../../models/` 下的两个编码器与各折基座模型已就位。

## 0. 一次性重建特征缓存

```bash
python -m common.gen_feats
```

之后的每次运行都复用 `cache/*.pkl`。若缓存缺失，首次训练运行也会自动重建。

## 1. family × loss 扫描

```bash
python run.py explore
```

对 kcat + Km 以 seed 42 训练全部 6 个 family × loss 配置（12 个 run）。按 fold-0 按靶点复合分挑选
method/loss；全局 PCC 仅作软下限。

## 2. Km seed 扫描

```bash
python run.py km-seed --family dual_residual_stabilized --loss composite_hinge --seeds 1,2,3,7
```

再加 4 个 Km run（seed 42 已由步骤 1 覆盖）。

## 3. kcat/Km 复用新的 kcat / Km 子模型

```bash
python run.py kok --family dual_residual_stabilized --loss composite_hinge \
    --kcat_ckpt kcat/dual_residual_stabilized__composite_hinge/model.pth \
    --Km_ckpt   Km/dual_residual_stabilized__composite_hinge/model.pth
```

## 4. 评估

```bash
python run.py eval
```

把每个已完成的 run 重新评估到 `<task>/<method>/eval/`，并刷新 `runs/summary.csv`。
`python run.py summary` 只打印汇总表。

## 5. 10 折 CV

```bash
python -m common.cv_prepare            # 把 <task>/<method>/ 复制到 cv/<task>/fold0/<method>/
python run.py cv                       # folds 1..9 × {kcat, Km}
python run.py eval
python run.py cv-kok                   # folds 1..9，kcat/Km（需先执行上面的 `cv`）
python run.py eval
python -m common.cv_summary            # 折均值 ± 标准差 -> cv/CV_SUMMARY.md + cv/cv_summary.csv
```

## 6. 单轴消融（10 折）

```bash
python run.py cv-ablate                # kcat + Km，全部变体
python run.py eval
python run.py cv-ablate-kok            # kcat/Km（需先执行上面的 cv-ablate）
python run.py eval
```

用 `--variants dual_residual__composite_hinge` 可只跑某个子集。

## run.py 选项

`--jobs N` 同时在跑的 run 数（默认 1）· `--device cuda:0` · `--dry-run` 只打印命令不执行。
每个 run 的日志：`runs/<task>__<method>__seed<seed>[__fold<k>].log`。

完整的轴矩阵、超参数与规范配方见 `configs/variants.py`。
