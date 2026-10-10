"""Inference for the fine-tuned models (replicates evaluation/eval_v4.py + eval_v4_kok.py).

    python -m common.eval_model --task kcat --run_dir kcat/<method> \
        --test_csv ../../datasets_finetune/mut_kcat.csv --eval_cache eval_cache/mut_kcat_eval.pkl \
        --output kcat/<method>/eval/mut_kcat_post.csv [--delta_zero]

`--delta_zero` skips the residual stream -> emits the fold-0 start ("Before").
Group-mean centring of delta features is done over the whole test CSV, matching
eval_v4.py; a group split across chunks therefore keeps one consistent mean.
"""
import argparse
import json
import os
import sys

import numpy as np
import pandas as pd
import torch as th

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from common import data as D                                   # noqa: E402
from common.model import ActivityModel, DualStreamRegressor    # noqa: E402


def load_run_config(run_dir):
    with open(os.path.join(run_dir, "run_config.json")) as f:
        return json.load(f)


def build_model(task, cfg, device):
    use_delta = bool(cfg["use_delta"])
    stabilized = bool(cfg["stabilized"])
    delta_max = float(cfg.get("delta_max", 2.0))
    if task in ("kcat", "Km"):
        return DualStreamRegressor(use_delta=use_delta, stabilized=stabilized,
                                   delta_max=delta_max)
    kcat_m = DualStreamRegressor(use_delta=use_delta, stabilized=stabilized, delta_max=delta_max)
    Km_m = DualStreamRegressor(use_delta=use_delta, stabilized=stabilized, delta_max=delta_max)
    return ActivityModel(kcat_m, Km_m, alpha=float(cfg.get("alpha", 0.5)),
                         stabilized=stabilized, delta_max=delta_max)


def load_model_for_run(task, run_dir, device, weight=None, cfg=None):
    cfg = cfg or load_run_config(run_dir)
    weight = weight or os.path.join(run_dir, "model.pth")
    model = build_model(task, cfg, device)
    miss, unexp = model.load_state_dict(th.load(weight, map_location=device), strict=False)
    if miss or unexp:
        print(f"[Model] {os.path.basename(weight)}: missing={len(miss)} unexpected={len(unexp)}")
    return model.to(device).eval(), cfg


def test_features(eval_cache, test_csv):
    feats = pd.read_pickle(eval_cache).values.astype(np.float32)
    test_df = pd.read_csv(test_csv, index_col=0)
    assert len(test_df) == len(feats), (len(test_df), len(feats))
    assert feats.shape[1] == D.N_FEAT, feats.shape
    return test_df, feats


def _log10(col, df):
    return np.log10(np.clip(df[col].values.astype(np.float64), 1e-10, None))


def true_label(task, test_df):
    """Task column(s) only -- mut_kcat.csv has no Km(M) and mut_Km.csv no kcat(s^-1)."""
    if task == "kcat":
        return _log10("kcat(s^-1)", test_df)
    if task == "Km":
        return _log10("Km(M)", test_df)
    return _log10("kcat(s^-1)", test_df) - _log10("Km(M)", test_df)


def predict_test(model, task, test_df, feats, device, delta_zero=False):
    grp = (test_df["UniProtID"].astype(str) + "__" + test_df["Substrate"].astype(str)).values
    delta = None if delta_zero else D.val_delta_context(feats, grp)
    return D.predict_array(model, feats, delta, task, device=device)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", required=True, choices=["kcat", "Km", "kcat_over_Km"])
    ap.add_argument("--run_dir", required=True)
    ap.add_argument("--test_csv", required=True)
    ap.add_argument("--eval_cache", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--weight", default=None, help="default: <run_dir>/model.pth")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--delta_zero", action="store_true")
    a = ap.parse_args()

    test_df, feats = test_features(a.eval_cache, a.test_csv)
    model, cfg = load_model_for_run(a.task, a.run_dir, a.device, weight=a.weight)
    pred = predict_test(model, a.task, test_df, feats, a.device, a.delta_zero)

    true_col = {"kcat": "true_log10_kcat", "Km": "true_log10_Km",
                "kcat_over_Km": "true_log10_kcat_over_Km"}[a.task]
    pred_col = {"kcat": "pred_log10_kcat", "Km": "pred_log10_Km",
                "kcat_over_Km": "pred_log10_kcat_over_Km"}[a.task]
    out = pd.DataFrame({
        "Entry": test_df.index.values,
        "UniProtID": test_df["UniProtID"].values,
        "Substrate": test_df["Substrate"].values,
        "EnzymeType": test_df["EnzymeType"].values,
        pred_col: pred,
        true_col: true_label(a.task, test_df),
    })
    os.makedirs(os.path.dirname(os.path.abspath(a.output)), exist_ok=True)
    out.to_csv(a.output, index=False)
    print(f"[Done] {a.output}  columns={list(out.columns)}")


if __name__ == "__main__":
    main()
