# CataPro-v2 fine-tuning

**English** | [中文](README_zh.md)

Fine-tune the CataPro-v2 models (kcat, Km, kcat/Km — one shared architecture). Every command
runs from `finetune/train/`; no scheduler is needed — `run.py` selects the runs defined in
`configs/variants.py` and calls the training entry point `python -m common.train` for each of
them directly.

**Prerequisites**: the two encoders and the per-fold base models present under `../../models/`.

## 0. Build the feature caches (once)

```bash
python -m common.gen_feats
```

Every later run reuses `cache/*.pkl`. The first training run also builds it automatically if
it is missing.

## 1. Family × loss sweep

```bash
python run.py explore
```

Trains seed 42 for kcat + Km over the 6 family × loss configs (12 runs). Pick the method/loss
with the best fold-0 per-target composite; global PCC is only a soft floor.

## 2. Km seed sweep

```bash
python run.py km-seed --family dual_residual_stabilized --loss composite_hinge --seeds 1,2,3,7
```

4 more Km runs (seed 42 already comes from step 1).

## 3. kcat/Km from the new kcat / Km sub-models

```bash
python run.py kok --family dual_residual_stabilized --loss composite_hinge \
    --kcat_ckpt kcat/dual_residual_stabilized__composite_hinge/model.pth \
    --Km_ckpt   Km/dual_residual_stabilized__composite_hinge/model.pth
```

## 4. Evaluate

```bash
python run.py eval
```

Re-reports every finished run into `<task>/<method>/eval/` and refreshes `runs/summary.csv`.
`python run.py summary` prints just the table.

## 5. 10-fold CV

```bash
python -m common.cv_prepare            # copies <task>/<method>/ to cv/<task>/fold0/<method>/
python run.py cv                       # folds 1..9 × {kcat, Km}
python run.py eval
python run.py cv-kok                   # folds 1..9, kcat/Km (needs `cv` above)
python run.py eval
python -m common.cv_summary            # fold mean ± std -> cv/CV_SUMMARY.md + cv/cv_summary.csv
```

## 6. One-axis-at-a-time ablations (10-fold)

```bash
python run.py cv-ablate                # kcat + Km, all variants
python run.py eval
python run.py cv-ablate-kok            # kcat/Km (needs cv-ablate above)
python run.py eval
```

Limit to a subset with `--variants dual_residual__composite_hinge`.

## run.py options

`--jobs N` runs in flight (default 1) · `--device cuda:0` · `--dry-run` prints the commands
without running them. Per-run logs: `runs/<task>__<method>__seed<seed>[__fold<k>].log`.

The full axis matrix, hyper-parameters and the canonical recipe are in `configs/variants.py`.
