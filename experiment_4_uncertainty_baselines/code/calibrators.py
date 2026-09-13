# -*- coding: utf-8 -*-
"""
Post-hoc calibrators for semantic segmentation — adapted from SelectiveCal
(Wang et al., CVPR 2023, "On Calibrating Semantic Segmentation Models").
Upstream source vendored in ./selectivecal_src/.

Key adaptations for the pepper -> tomato BINARY calibration transfer:
  * num_class is a constructor argument (= 2 here: background vs fruit), not a
    hard-coded global (upstream hard-codes 150 for ADE20K).
  * All hard-coded .cuda() calls replaced with device-agnostic ops.
  * A uniform fit()/calibrate() interface added around each nn.Module so the
    driver can train + apply every method the same way.
  * Calibration is done at a reduced feature resolution (upstream uses the
    "_inp56" 56px maps) — the driver caches logits/images/labels at low res.

Six methods (matching the requested set):
  TS        Temperature_Scaling         single scalar T
  Logistic  Vector_Scaling              per-class affine (multinomial logistic)
  Dirichlet Dirichlet_Scaling           linear map on log-softmax
  LTS       LTS_CamVid_With_Image       pixel-adaptive temperature (uses image)
  Meta      Meta_Scaling                entropy-gated temperature (Meta-Cal)
  Selective SelectiveScaling            Binary_Classifier correctness gate + TS

All modules consume logits shaped (B, C, H, W) and labels (B, H, W) with
ignore_index = 255. calibrate() returns calibrated logits (B, C, H, W).
"""

import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

IGNORE_INDEX = 255


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────
def _init_conv(m):
    if isinstance(m, nn.Conv2d):
        n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
        m.weight.data.normal_(0, math.sqrt(2. / n))
        if m.bias is not None:
            m.bias.data.zero_()
    elif isinstance(m, nn.BatchNorm2d):
        m.weight.data.fill_(1)
        m.bias.data.zero_()


def pixel_entropy(logits):
    """Per-pixel Shannon entropy of softmax(logits). logits: (B,C,H,W)->(B,H,W)."""
    p = F.softmax(logits, dim=1)
    return torch.sum(-p * torch.log(p + 1e-12), dim=1)


# ─────────────────────────────────────────────────────────────────────────────
# Calibrator modules
# ─────────────────────────────────────────────────────────────────────────────
class Temperature_Scaling(nn.Module):
    def __init__(self, num_class=2):
        super().__init__()
        self.num_class = num_class
        self.temperature_single = nn.Parameter(torch.ones(1))

    def weights_init(self):
        self.temperature_single.data.fill_(1)

    def forward(self, logits):
        T = self.temperature_single.to(logits.device)
        return logits / T


class Vector_Scaling(nn.Module):
    """Logistic / vector scaling: per-class affine transform of the logits."""
    def __init__(self, num_class=2):
        super().__init__()
        self.num_class = num_class
        self.vector_parameters = nn.Parameter(torch.ones(1, num_class, 1, 1))
        self.vector_offset = nn.Parameter(torch.zeros(1, num_class, 1, 1))

    def weights_init(self):
        self.vector_offset.data.fill_(0)
        self.vector_parameters.data.fill_(1)

    def forward(self, logits):
        return logits * self.vector_parameters.to(logits.device) \
            + self.vector_offset.to(logits.device)


class Dirichlet_Scaling(nn.Module):
    def __init__(self, num_class=2):
        super().__init__()
        self.num_class = num_class
        self.dirichlet_linear = nn.Linear(num_class, num_class)

    def weights_init(self):
        self.dirichlet_linear.weight.data.copy_(torch.eye(self.num_class))
        self.dirichlet_linear.bias.data.copy_(torch.zeros(self.num_class))

    def forward(self, logits):
        logits = logits.permute(0, 2, 3, 1)
        probs = F.softmax(logits, dim=-1)
        ln_probs = torch.log(probs + 1e-10)
        return self.dirichlet_linear(ln_probs).permute(0, 3, 1, 2)


class LTS_CamVid_With_Image(nn.Module):
    """Local Temperature Scaling — pixel-adaptive temperature from logits+image."""
    def __init__(self, num_class=2):
        super().__init__()
        nc = num_class
        k = dict(kernel_size=5, stride=1, padding=4, padding_mode='reflect',
                 dilation=2, bias=True)
        self.c1 = nn.Conv2d(nc, 1, **k)
        self.c2 = nn.Conv2d(nc, 1, **k)
        self.c3 = nn.Conv2d(nc, 1, **k)
        self.c4 = nn.Conv2d(nc, 1, **k)
        self.p1 = nn.Conv2d(nc, 1, **k)
        self.p2 = nn.Conv2d(nc, 1, **k)
        self.p3 = nn.Conv2d(nc, 1, **k)
        self.c_img = nn.Conv2d(3, 1, **k)
        self.p_img = nn.Conv2d(nc, 1, **k)
        self.num_class = num_class

    def weights_init(self):
        for m in [self.c1, self.c2, self.c3, self.c4, self.p1, self.p2, self.p3,
                  self.c_img, self.p_img]:
            nn.init.zeros_(m.weight.data)
            nn.init.zeros_(m.bias.data)

    def forward(self, logits, image):
        dev = logits.device
        one = torch.ones(1, device=dev)
        t1 = self.c1(logits) + one
        t2 = self.c2(logits) + one
        t3 = self.c3(logits) + one
        t4 = self.c4(logits) + one
        pp1 = self.p1(logits)
        pp2 = self.p2(logits)
        pp3 = self.p3(logits)
        lvl11 = t1 * torch.sigmoid(pp1) + t2 * (1.0 - torch.sigmoid(pp1))
        lvlnc = t3 * torch.sigmoid(pp2) + t4 * (1.0 - torch.sigmoid(pp2))
        temp1 = lvl11 * torch.sigmoid(pp3) + lvlnc * (1.0 - torch.sigmoid(pp3))
        temp2 = self.c_img(image) + one
        tparam = self.p_img(logits)
        temperature = temp1 * torch.sigmoid(tparam) + temp2 * (1.0 - torch.sigmoid(tparam))
        temperature = F.relu(temperature + one) + 1e-8
        temperature = temperature.repeat(1, self.num_class, 1, 1)
        return logits / temperature


class Meta_Scaling(nn.Module):
    """Meta-Cal: entropy-gated temperature scaling.

    Low-entropy (confident) pixels get temperature-scaled; high-entropy pixels
    are pushed to uniform (near chance confidence) at inference.
    """
    def __init__(self, num_class=2, alpha=0.05):
        super().__init__()
        self.num_class = num_class
        self.alpha = alpha
        self.temperature_single = nn.Parameter(torch.ones(1))

    def weights_init(self):
        self.temperature_single.data.fill_(1)

    def forward(self, logits, threshold):
        """Inference-time transform on full spatial logits (B,C,H,W)."""
        T = self.temperature_single.to(logits.device)
        ent = pixel_entropy(logits).unsqueeze(1)          # (B,1,H,W)
        gate = (ent < threshold).float()                  # confident pixels
        scaled = logits / T
        uniform = torch.zeros_like(logits)                # softmax -> uniform
        return gate * scaled + (1.0 - gate) * uniform


class Binary_Classifier(nn.Module):
    """Correctness predictor used by Selective Scaling.

    Predicts, per pixel, whether the base model's argmax equals the GT label
    (2-way: incorrect=0 / correct=1). Trained with CE on the fit set.
    """
    def __init__(self, num_class=2):
        super().__init__()
        self.num_class = num_class
        self.dirichlet_linear = nn.Linear(num_class, num_class)
        self.bn0 = nn.BatchNorm2d(num_class)
        self.linear_1 = nn.Linear(num_class, num_class * 2)
        self.bn1 = nn.BatchNorm2d(num_class * 2)
        self.linear_2 = nn.Linear(num_class * 2, num_class)
        self.bn2 = nn.BatchNorm2d(num_class)
        self.binary_linear = nn.Linear(num_class, 2)
        self.relu = nn.ReLU()

    def weights_init(self):
        self.dirichlet_linear.weight.data.copy_(torch.eye(self.num_class))
        self.dirichlet_linear.bias.data.copy_(torch.zeros(self.num_class))

    def forward(self, logits, gt=None):
        x = logits.permute(0, 2, 3, 1)
        probs = F.softmax(x, dim=-1)
        ln_probs = torch.log(probs + 1e-16)
        out = self.dirichlet_linear(ln_probs)
        out = self.relu(self.bn0(out.permute(0, 3, 1, 2)))
        out = self.linear_1(out.permute(0, 2, 3, 1))
        out = self.relu(self.bn1(out.permute(0, 3, 1, 2)))
        out = self.linear_2(out.permute(0, 2, 3, 1))
        out = self.relu(self.bn2(out.permute(0, 3, 1, 2)))
        tf_positive = self.binary_linear(out.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
        if gt is None:
            return tf_positive
        pred = torch.max(probs, dim=-1)[1]
        mask = (pred == gt).long()      # 1 = correct, 0 = incorrect
        return tf_positive, mask


# ─────────────────────────────────────────────────────────────────────────────
# Fit / apply wrappers — uniform interface over all six methods
# ─────────────────────────────────────────────────────────────────────────────
NAMES = ["TS", "Logistic", "Dirichlet", "LTS", "Meta", "Selective"]


def _iter_batches(cache, batch_size, device, need_image=False):
    """cache: dict with 'logits'(N,C,h,w),'labels'(N,h,w),'image'(N,3,h,w)."""
    N = cache["logits"].shape[0]
    for i in range(0, N, batch_size):
        lo = cache["logits"][i:i + batch_size].float().to(device)
        la = cache["labels"][i:i + batch_size].long().to(device)
        im = cache["image"][i:i + batch_size].float().to(device) if need_image else None
        yield im, lo, la


class Calibrator:
    """Wraps one method with fit(cache)/calibrate(logits,image)."""

    def __init__(self, name, num_class=2, alpha=0.05):
        self.name = name
        self.num_class = num_class
        self.alpha = alpha
        self.meta_threshold = None
        self.temperature = None          # for Selective: a fitted TS module
        if name == "TS":
            self.model = Temperature_Scaling(num_class)
        elif name == "Logistic":
            self.model = Vector_Scaling(num_class)
        elif name == "Dirichlet":
            self.model = Dirichlet_Scaling(num_class)
        elif name == "LTS":
            self.model = LTS_CamVid_With_Image(num_class)
        elif name == "Meta":
            self.model = Meta_Scaling(num_class, alpha)
        elif name == "Selective":
            self.model = Binary_Classifier(num_class)
        else:
            raise ValueError(f"unknown calibrator {name}")
        self.model.weights_init()

    # ── training ─────────────────────────────────────────────────────────────
    def fit(self, cache, device, epochs=40, batch_size=20, lr=1e-3):
        self.model.to(device).train()
        need_image = self.name == "LTS"

        if self.name == "Meta":
            self._fit_meta(cache, device, epochs, batch_size, lr)
            return self
        if self.name == "Selective":
            self._fit_selective(cache, device, epochs, batch_size, lr)
            return self

        crit = nn.CrossEntropyLoss(ignore_index=IGNORE_INDEX)
        opt = torch.optim.AdamW(self.model.parameters(), lr=lr, weight_decay=1e-6)
        for _ in range(epochs):
            for im, lo, la in _iter_batches(cache, batch_size, device, need_image):
                opt.zero_grad()
                cal = self.model(lo, im) if need_image else self.model(lo)
                loss = crit(cal, la)
                loss.backward()
                opt.step()
        self.model.eval()
        return self

    def _fit_meta(self, cache, device, epochs, batch_size, lr):
        # threshold: entropy quantile leaving alpha fraction of pixels "uncertain"
        ents = []
        with torch.no_grad():
            for _, lo, la in _iter_batches(cache, batch_size, device):
                e = pixel_entropy(lo)[la != IGNORE_INDEX]
                ents.append(e.cpu())
        ents = torch.cat(ents)
        # torch.quantile caps at ~2**24 elements; fall back to numpy for big fits
        # (e.g. hi-res calibration maps). Linear interp -> identical to torch below.
        q = 1.0 - self.alpha
        if ents.numel() > 16_000_000:
            self.meta_threshold = float(np.quantile(ents.numpy(), q))
        else:
            self.meta_threshold = float(torch.quantile(ents, q))
        # fit temperature on the confident (below-threshold) pixels
        crit = nn.CrossEntropyLoss(ignore_index=IGNORE_INDEX)
        opt = torch.optim.AdamW(self.model.parameters(), lr=lr, weight_decay=1e-6)
        thr = self.meta_threshold
        for _ in range(epochs):
            for _, lo, la in _iter_batches(cache, batch_size, device):
                opt.zero_grad()
                T = self.model.temperature_single.to(device)
                scaled = lo / T
                ent = pixel_entropy(lo)
                mask = (ent < thr) & (la != IGNORE_INDEX)
                if mask.sum() == 0:
                    continue
                cal = scaled.permute(0, 2, 3, 1)[mask]     # (M,C)
                tgt = la[mask]
                loss = crit(cal, tgt)
                loss.backward()
                opt.step()
        self.model.eval()

    def _fit_selective(self, cache, device, epochs, batch_size, lr):
        # 1) fit a temperature (reuse TS) for the "correct" pixels
        ts = Calibrator("TS", self.num_class).fit(cache, device, epochs, batch_size, lr)
        self.temperature = ts.model.temperature_single.detach().to(device)
        # 2) train the binary correctness classifier (CE + neg-sample duplication)
        crit = nn.CrossEntropyLoss()
        opt = torch.optim.AdamW(self.model.parameters(), lr=lr, weight_decay=1e-6)
        self.model.train()
        for _ in range(epochs):
            for _, lo, la in _iter_batches(cache, batch_size, device):
                opt.zero_grad()
                tf, bl = self.model(lo, la)               # (B,2,h,w),(B,h,w)
                tf = tf.permute(0, 2, 3, 1).reshape(-1, 2)
                bl = bl.reshape(-1)
                valid = la.reshape(-1) != IGNORE_INDEX
                tf, bl = tf[valid], bl[valid]
                # duplicate the minority (incorrect) class once to fight imbalance
                neg = (bl == 0).nonzero(as_tuple=True)[0]
                if neg.numel() > 0:
                    tf = torch.cat([tf, tf[neg]], dim=0)
                    bl = torch.cat([bl, bl[neg]], dim=0)
                loss = crit(tf, bl)
                loss.backward()
                opt.step()
        self.model.eval()

    # ── inference ────────────────────────────────────────────────────────────
    @torch.no_grad()
    def calibrate(self, logits, image=None):
        """logits (B,C,H,W) -> calibrated logits (B,C,H,W)."""
        dev = logits.device
        self.model.to(dev).eval()
        if self.name == "LTS":
            return self.model(logits, image)
        if self.name == "Meta":
            return self.model(logits, self.meta_threshold)
        if self.name == "Selective":
            T = self.temperature.to(dev)
            scaled = logits / T
            tf = self.model(logits)                        # (B,2,H,W)
            correct = torch.max(tf, dim=1)[1].unsqueeze(1).float()  # 1=correct
            uniform = torch.zeros_like(logits)
            return correct * scaled + (1.0 - correct) * uniform
        return self.model(logits)                          # TS/Logistic/Dirichlet

    def state(self):
        d = {"name": self.name, "num_class": self.num_class, "alpha": self.alpha,
             "model": self.model.state_dict(), "meta_threshold": self.meta_threshold}
        if self.temperature is not None:
            d["temperature"] = self.temperature.cpu()
        return d
