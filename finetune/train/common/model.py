"""Shared model definitions for the CataPro-v2 fine-tuning package.

`DualStreamRegressor` is the ONE architecture used for both kcat and Km (hard
constraint: the two tasks must share an identical model). It extends the released
CataPro head with a bounded mutation-effect residual stream:

    out = out_abs(x) + scale * delta(delta_feats)
    delta_feats = x - group_mean(x)          (mutation-effect signal)
    delta     = tanh(delta_out(delta_decoder(delta_norm(delta_feats))))   # stabilised
    scale     = delta_max * sigmoid(delta_gate)                          # stabilised

`delta_out` is zero-initialised, so at init delta = 0 and out == out_abs exactly
(the released checkpoint's behaviour is reproduced bit-for-bit).

Families (all share the class, so "same architecture" holds by construction):
  * dual_residual_stabilized : use_delta=True,  stabilized=True   <- the chosen family
  * dual_residual            : use_delta=True,  stabilized=False  (replicates the v4 recipe)
  * single_stream            : use_delta=False -> plain CataPro head, loss-only change

`ActivityModel` (kcat/Km) reuses two `DualStreamRegressor` sub-models as its frozen
base: pred_1 = new_kcat(x) - new_Km(x). Only the delta head on the fusion output is
trained, keeping kok's "non-original" prediction part on the new kcat/Km models.
"""
import torch as th
import torch.nn as nn

N_FEAT = 1959


class DualStreamRegressor(nn.Module):
    def __init__(self, hid=256, rate=0.0, use_delta=True, stabilized=True,
                 delta_hid=256, delta_max=2.0):
        super().__init__()
        self.use_delta = use_delta
        self.stabilized = stabilized
        self.delta_max = float(delta_max)

        # ---- absolute stream (keys match models/<task>_models/0_bestmodel.pth) ----
        self.prot_norm = nn.BatchNorm1d(1024)
        self.molt5_norm = nn.BatchNorm1d(768)
        self.decoder = nn.Sequential(
            nn.Linear(N_FEAT, hid), nn.BatchNorm1d(hid), nn.Dropout(p=rate), nn.ReLU())
        self.out = nn.Sequential(nn.Linear(hid, 1))

        # ---- mutation-effect residual stream (trainable) ----
        if use_delta:
            self.delta_norm = nn.BatchNorm1d(N_FEAT)
            self.delta_decoder = nn.Sequential(
                nn.Linear(N_FEAT, delta_hid), nn.BatchNorm1d(delta_hid),
                nn.Dropout(p=rate), nn.ReLU())
            self.delta_out = nn.Linear(delta_hid, 1)
            nn.init.zeros_(self.delta_out.weight)
            nn.init.zeros_(self.delta_out.bias)
            if stabilized:
                # scale = delta_max * sigmoid(gate); gate=0 -> scale == 1.0 at init
                self.delta_gate = nn.Parameter(th.zeros(1))
            else:
                self.delta_scale = nn.Parameter(th.ones(1))

    def _scale(self):
        """Differentiable delta scale tensor."""
        if self.stabilized:
            return self.delta_max * th.sigmoid(self.delta_gate)
        return self.delta_scale

    def scale_value(self):
        """Python float of the current delta scale (for logging)."""
        if not self.use_delta:
            return 0.0
        with th.no_grad():
            return float(self._scale().item())

    def _hidden(self, ezy_feats, sbt_feats):
        prot = self.prot_norm(ezy_feats[:, :1024])
        molt5 = self.molt5_norm(sbt_feats[:, :768])
        macc = sbt_feats[:, 768:]
        return self.decoder(th.cat([prot, molt5, macc], dim=1))

    def forward(self, ezy_feats, sbt_feats, delta_feats=None, return_feats=False):
        feats = self._hidden(ezy_feats, sbt_feats)
        out_abs = self.out(feats)
        out = out_abs
        if self.use_delta and delta_feats is not None:
            r = self.delta_out(self.delta_decoder(self.delta_norm(delta_feats)))
            if self.stabilized:
                r = th.tanh(r)
            out = out_abs + self._scale() * r
        if return_feats:
            return out, feats
        return out

    def freeze_absolute(self):
        """Freeze the released absolute stream; keep the delta stream trainable.

        single_stream (use_delta=False) has no delta stream, so the whole head is
        fine-tuned with the swapped loss instead.
        """
        if not self.use_delta:
            for p in self.parameters():
                p.requires_grad = True
            return
        for name, p in self.named_parameters():
            p.requires_grad = name.startswith("delta")
        # frozen BatchNorm keeps its pretrained running stats
        self.prot_norm.eval()
        self.molt5_norm.eval()
        self.decoder.eval()

    def trainable_names(self):
        return [n for n, p in self.named_parameters() if p.requires_grad]


class ActivityModel(nn.Module):
    """kcat/Km fusion: frozen new kcat/Km sub-models + trainable residual head."""

    def __init__(self, kcat_model, Km_model, rate=0.0, alpha=0.5, stabilized=True,
                 delta_max=2.0):
        super().__init__()
        self.alpha = float(alpha)
        self.stabilized = stabilized
        self.delta_max = float(delta_max)
        self.kcat_model = kcat_model
        self.Km_model = Km_model

        # ---- frozen fusion head (keys from models/act_models/0_bestmodel.pth) ----
        self.prot_norm = nn.BatchNorm1d(1024)
        self.molt5_norm = nn.BatchNorm1d(768)
        self.decoder = nn.Sequential(
            nn.Linear(N_FEAT, 256), nn.BatchNorm1d(256), nn.Dropout(p=rate), nn.ReLU())
        self.attn = nn.Sequential(nn.Linear(256, 256), nn.Softmax(dim=1))
        self.out = nn.Linear(256, 1)

        # ---- trainable residual head on the fusion output (same form as the single-task one) ----
        self.delta_norm = nn.BatchNorm1d(N_FEAT)
        self.delta_decoder = nn.Sequential(
            nn.Linear(N_FEAT, 256), nn.BatchNorm1d(256),
            nn.Dropout(p=rate), nn.ReLU())
        self.delta_out = nn.Linear(256, 1)
        nn.init.zeros_(self.delta_out.weight)
        nn.init.zeros_(self.delta_out.bias)
        if stabilized:
            self.delta_gate = nn.Parameter(th.zeros(1))
        else:
            self.delta_scale = nn.Parameter(th.ones(1))

    def _scale(self):
        if self.stabilized:
            return self.delta_max * th.sigmoid(self.delta_gate)
        return self.delta_scale

    def scale_value(self):
        with th.no_grad():
            return float(self._scale().item())

    def _fusion(self, ezy_feats, sbt_feats):
        prot = self.prot_norm(ezy_feats[:, :1024])
        molt5 = self.molt5_norm(sbt_feats[:, :768])
        macc = sbt_feats[:, 768:]
        feats = self.decoder(th.cat([prot, molt5, macc], dim=1))
        return self.out(self.attn(feats) * feats)

    def forward(self, ezy_feats, sbt_feats, delta_feats=None):
        pred_kcat = self.kcat_model(ezy_feats, sbt_feats, delta_feats).ravel()
        pred_Km = self.Km_model(ezy_feats, sbt_feats, delta_feats).ravel()
        pred_1 = (pred_kcat - pred_Km).detach()
        pred_2 = self._fusion(ezy_feats, sbt_feats).ravel()
        pred = pred_1 * (1 - self.alpha) + pred_2 * self.alpha
        if delta_feats is not None:
            r = self.delta_out(self.delta_decoder(self.delta_norm(delta_feats))).ravel()
            if self.stabilized:
                r = th.tanh(r)
            pred = pred + self._scale() * r
        return pred_kcat, pred_Km, pred

    def freeze_absolute(self):
        """Freeze sub-models + fusion head; keep only the fusion delta head trainable."""
        for name, p in self.named_parameters():
            p.requires_grad = name.startswith("delta")
        for m in (self.kcat_model, self.Km_model):
            m.eval()
        self.prot_norm.eval()
        self.molt5_norm.eval()
        self.decoder.eval()

    def trainable_names(self):
        return [n for n, p in self.named_parameters() if p.requires_grad]
