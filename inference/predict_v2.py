"""CataPro-v2 inference: the fine-tuned, target-centred "delta" models.

CataPro-v2 keeps CataPro's frozen absolute stream and adds a trainable
mutation-effect residual stream. For each input it predicts

    out = out_abs(x) + scale * delta(x - group_mean(x))

where the group is the (enzyme, substrate) pair. The delta term is only
meaningful when several rows share one target+substrate group; for a group with
a single row delta == 0 and the prediction falls back to the frozen CataPro base
model (this mirrors how the models were trained/tested).

Input CSV must have the columns `Enzyme_id, type, sequence, smiles` (same layout
as `samples/sample_inp.csv`); `type` is the wild-type/mutant flag, and the group
(used to centre delta) is the `Enzyme_id` x `smiles` pair.

    python inference/predict_v2.py \
        -inp_fpath samples/sample_inp.csv \
        -out_fpath catapro_v2_predict_score.csv \
        -model_dpath models -device cuda:0

Writes `fasta_id, smiles, pred_log10[kcat(s^-1)], pred_log10[Km(M)],
pred_log10[kcat/Km(s^-1 M^-1)]` -- the same schema as the CataPro prediction file.

The two big frozen encoders (`ProtT5`, `MolT5`) are read from -model_dpath;
download them per the top-level README and place them under `models/`.
"""
import argparse
import os
import sys

import numpy as np
import pandas as pd
import torch as th

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
TRAIN_DIR = os.path.join(REPO, "finetune", "train")
sys.path.insert(0, HERE)          # inference/utils.py
sys.path.insert(0, TRAIN_DIR)     # common.* + configs.*

from utils import GetMACCSKeys, Seq_to_vec, get_molT5_embed   # noqa: E402
from common import data as D, eval_model as E                 # noqa: E402

# the canonical fine-tuned model family written up for CataPro-v2
METHOD = "dual_residual_stabilized__composite_hinge"
RUN_DIRS = {
    "kcat": os.path.join(TRAIN_DIR, "kcat", METHOD),
    "Km": os.path.join(TRAIN_DIR, "Km", METHOD),
    "kcat_over_Km": os.path.join(TRAIN_DIR, "kcat_over_Km", METHOD),
}


def extract_features(inp_df, model_dpath):
    """(sequence, smiles) -> [ProtT5 1024 | MolT5 768 | MACCS 167] float32."""
    sequences = inp_df["sequence"].values.copy()
    smiles = inp_df["smiles"].values
    prot = Seq_to_vec(sequences, os.path.join(model_dpath, "prot_t5_xl_uniref50"))
    molt5 = get_molT5_embed(smiles, os.path.join(model_dpath, "molt5-base-smiles2caption"))
    macc = GetMACCSKeys(smiles)
    feats = np.concatenate([prot, molt5, macc], axis=1).astype(np.float32)
    assert feats.shape[1] == D.N_FEAT, feats.shape
    return feats


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("-inp_fpath", default=os.path.join(REPO, "samples", "sample_inp.csv"),
                    help="input CSV: Enzyme_id, type, sequence, smiles")
    ap.add_argument("-out_fpath", default="catapro_v2_predict_score.csv",
                    help="where to write the predictions")
    ap.add_argument("-model_dpath", default=os.path.join(REPO, "models"),
                    help="directory holding prot_t5_xl_uniref50/ and molt5-base-smiles2caption/")
    ap.add_argument("-device", default="cuda:0", help="cuda:0 or cpu")
    a = ap.parse_args()

    inp_df = pd.read_csv(a.inp_fpath, index_col=0)
    for col in ("Enzyme_id", "type", "sequence", "smiles"):
        if col not in inp_df.columns:
            raise SystemExit(f"input is missing column '{col}'")

    # group = enzyme x substrate; delta is centred on this group. The input has no
    # substrate name, so the substrate's SMILES is used as its identity (this is how
    # training groups rows too: one group per (enzyme, substrate)).
    groups = np.array([f"{i}_{s}" for i, s in zip(inp_df["Enzyme_id"], inp_df["smiles"])])
    feats = extract_features(inp_df, a.model_dpath)
    delta = D.val_delta_context(feats, groups)
    n_single = sum(1 for g in np.unique(groups) if (groups == g).sum() < 2)
    if n_single:
        print(f"[predict_v2] {n_single} group(s) have a single row -> delta=0 for them "
              f"(base-model output)")

    # kcat / Km: one DualStreamRegressor each; kcat/Km: the fused ActivityModel
    kcat_model, _ = E.load_model_for_run("kcat", RUN_DIRS["kcat"], a.device)
    Km_model, _ = E.load_model_for_run("Km", RUN_DIRS["Km"], a.device)
    pred_kcat = D.predict_array(kcat_model, feats, delta, "kcat", device=a.device)
    pred_Km = D.predict_array(Km_model, feats, delta, "Km", device=a.device)

    kok_model, _ = E.load_model_for_run("kcat_over_Km", RUN_DIRS["kcat_over_Km"], a.device)
    pred_kok = D.predict_array(kok_model, feats, delta, "kcat_over_Km", device=a.device)

    out = pd.DataFrame({
        # same id convention as the base CataPro prediction file
        "fasta_id": [f"{i}_{t}" for i, t in zip(inp_df["Enzyme_id"], inp_df["type"])],
        "smiles": inp_df["smiles"].values,
        "pred_log10[kcat(s^-1)]": pred_kcat,
        "pred_log10[Km(M)]": pred_Km,
        "pred_log10[kcat/Km(s^-1 M^-1)]": pred_kok,
    })
    out.to_csv(a.out_fpath, index=False)
    print(f"[predict_v2] wrote {len(out)} row(s) -> {a.out_fpath}")


if __name__ == "__main__":
    main()
