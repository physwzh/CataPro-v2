"""Data plumbing, ported from training/kcat_v4/training.py (identical semantics).

The caches under cache/ are rebuilt on first run by common.gen_feats; once they
exist no re-extraction is needed. Layout of every cache: [1959 features | label col(s) | group_id | fold].
"""
import numpy as np
import pandas as pd
import torch as th
from torch.utils.data import DataLoader, Dataset, Sampler

N_FEAT = 1959


class MutantDataset(Dataset):
    def __init__(self, values, group_ids):
        self.values = values
        self.group_ids = group_ids

    def __getitem__(self, idx):
        return self.values[idx], self.group_ids[idx]

    def __len__(self):
        return len(self.values)


class GroupBatchSampler(Sampler):
    """Packs whole mutation groups into batches so ranking losses see full groups."""

    def __init__(self, group_ids, batch_size, shuffle=True):
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.group_to_indices = {}
        for idx, gid in enumerate(group_ids):
            self.group_to_indices.setdefault(gid, []).append(idx)
        self.group_list = sorted(self.group_to_indices.items(),
                                 key=lambda x: len(x[1]), reverse=True)

    def _pack(self):
        import random
        groups = list(self.group_list)
        if self.shuffle:
            random.shuffle(groups)
        batches, cur, cur_sz = [], [], 0
        for gid, indices in groups:
            if len(indices) > self.batch_size:
                for i in range(0, len(indices), self.batch_size):
                    batches.append(indices[i:i + self.batch_size])
                continue
            if cur_sz + len(indices) > self.batch_size and cur:
                batches.append([i for s in cur for i in s])
                cur, cur_sz = [], 0
            cur.append(indices)
            cur_sz += len(indices)
        if cur:
            batches.append([i for s in cur for i in s])
        if self.shuffle:
            random.shuffle(batches)
        return batches

    def __iter__(self):
        return iter(self._pack())

    def __len__(self):
        return len(self._pack())


def load_task_cache(pkl_path, label_col):
    """Return (feats[N,1959], label[N], group_ids[N] int, fold[N] int)."""
    df = pd.read_pickle(pkl_path)
    group_ids = df["group_id"].values.astype(np.int64)
    folds = df["fold"].values.astype(np.int64)
    arr = df.drop(columns=["group_id", "fold"]).values.astype(np.float32)
    assert arr.shape[1] > N_FEAT, arr.shape
    feats = arr[:, :N_FEAT]
    label = arr[:, label_col].astype(np.float32)
    return feats, label, group_ids, folds


def make_loaders(feats, labels, group_ids, folds, fold, batch_size, num_workers=4):
    train_mask = folds != fold
    val_mask = folds == fold
    train_ds = MutantDataset(th.from_numpy(np.concatenate([feats[train_mask], labels[train_mask, None]], axis=1)),
                             group_ids[train_mask])
    val_ds = MutantDataset(th.from_numpy(np.concatenate([feats[val_mask], labels[val_mask, None]], axis=1)),
                           group_ids[val_mask])
    train_loader = DataLoader(
        train_ds, batch_sampler=GroupBatchSampler(group_ids[train_mask], batch_size, True),
        num_workers=num_workers)
    val_loader = DataLoader(
        val_ds, batch_sampler=GroupBatchSampler(group_ids[val_mask], batch_size, False),
        num_workers=num_workers)
    return train_loader, val_loader, group_ids[val_mask], int(train_mask.sum()), int(val_mask.sum())


def compute_group_indices(group_ids_in_batch):
    return [th.where(group_ids_in_batch == g)[0] for g in th.unique(group_ids_in_batch)]


def compute_delta_feats(feats_all, group_idx_list):
    """delta = sample features - group-mean features (mutation-effect signal)."""
    deltas = th.zeros_like(feats_all)
    for idx in group_idx_list:
        if len(idx) < 2:
            continue
        deltas[idx] = feats_all[idx] - feats_all[idx].mean(dim=0, keepdim=True)
    return deltas


def split_batch(data, group_ids, device):
    """data layout: [1959 features | label] -> ezy(1024), sbt(935), label, gidx, delta."""
    data = data.to(device)
    group_ids = group_ids.to(device)
    feats_all = data[:, :N_FEAT]
    ezy = feats_all[:, :1024]
    sbt = feats_all[:, 1024:]
    label = data[:, N_FEAT].float()
    gidx = compute_group_indices(group_ids)
    delta = compute_delta_feats(feats_all, gidx)
    return ezy, sbt, label, gidx, delta


def val_delta_context(feats, group_ids):
    """Group-mean-centred delta features over a whole (validation) split, aligned to rows."""
    delta = np.zeros_like(feats)
    for g in np.unique(group_ids):
        idx = np.where(group_ids == g)[0]
        if len(idx) < 2:
            continue
        delta[idx] = feats[idx] - feats[idx].mean(axis=0, keepdims=True)
    return delta


def predict_array(model, feats, delta, task, device="cuda:0", bs=256):
    """Row-aligned predictions for a whole split (no batching artefacts).

    Group-centring is done globally by the caller, so a group split across two
    chunks still sees one consistent group mean -- this matches eval_model.py.
    `delta=None` skips the residual stream entirely -> the frozen fold-0 start.
    """
    model.eval()
    preds = []
    with th.no_grad():
        for i in range(0, len(feats), bs):
            f = th.from_numpy(np.ascontiguousarray(feats[i:i + bs])).to(device)
            d = None if delta is None else th.from_numpy(
                np.ascontiguousarray(delta[i:i + bs])).to(device)
            ezy = f[:, :1024]
            sbt = f[:, 1024:]
            if task == "kcat_over_Km":
                _, _, p = model(ezy, sbt, d)
            else:
                p = model(ezy, sbt, d)
            preds.append(p.detach().cpu().numpy().ravel())
    return np.concatenate(preds, axis=0)
