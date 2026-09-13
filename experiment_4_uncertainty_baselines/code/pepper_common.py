# -*- coding: utf-8 -*-
"""
Shared utilities for the PEPPER (in-domain, 8-class) post-hoc calibration study.

Mirrors the tomato study (code/posthoc_calibration/) but:
  * multi-class (NUM=8): bg + 7 pepper subtypes, NO binary reduction
  * pepper COCOSegDataset (COCO polygons -> semantic map, remapped to 0..7)
  * fit calibrators on the fixed cal(30) subset of `valid`; eval on val(33)/test(93)

Everything runs via SLURM (ssl env) — never python directly on login nodes.
"""

import os
import sys
import json
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms

import skimage.draw
from transformers import Mask2FormerForUniversalSegmentation, Mask2FormerImageProcessor

# ── paths ─────────────────────────────────────────────────────────────────────
ROOT = "/lustre/scratch/data/mibrahi2_hpc-my_research/mibrahi2_hpc-my_research-1783386603/mibrahi2_hpc-my_research-1775524204"
COCO_FILE = f"{ROOT}/data/bup_20/CKA_sweet_pepper_2020_summer/CKA_sweet_pepper_2020_summer.json"
DATA_ROOT = f"{ROOT}/data/bup_20"                       # rel paths in coco are under here
OUT_DIR   = f"{ROOT}/results/posthoc_calibration_pepper"
MODELS_DIR = f"{OUT_DIR}/models"                        # seed_<S>/model_shot_5.pt
SPLIT_MANIFEST = f"{OUT_DIR}/split_manifest.json"
TOMATO_ROOT = f"{ROOT}/data/tomato_esra"                # tomato OoD source
GROWLI_ROOT = f"{ROOT}/data/growliflower_l"             # growliflower OoD source

# reuse the shared 6 calibrators from the tomato study (num_class-parametrized)
sys.path.insert(0, f"{ROOT}/code/posthoc_calibration")
sys.path.insert(0, f"{ROOT}/code/bup_20_trials/elora")

ADE_MEAN = np.array([123.675, 116.280, 103.530]) / 255
ADE_STD  = np.array([58.395,  57.120,  57.375])  / 255

# ── pepper 8-class label map (identical to fullft_evaluation_seeded) ───────────
ID2LABEL_ORIG = {0: "bg", 11: "pepper_kp", 12: "pepper red", 13: "pepper yellow",
                 14: "pepper green", 15: "pepper mixed", 17: "pepper mixed_red",
                 18: "pepper mixed_yellow"}
LABEL2ID = {old: new for new, old in enumerate(sorted(ID2LABEL_ORIG.keys()))}
ID2LABEL = {new: ID2LABEL_ORIG[old] for old, new in LABEL2ID.items()}
NUM = len(ID2LABEL)                                     # 8
_REMAP_LUT = np.zeros(256, dtype=np.int64)
for _old, _new in LABEL2ID.items():
    _REMAP_LUT[_old] = _new

SPLIT_IDS = {
    "train": list(range(283, 345)) + list(range(408, 471)),   # 125
    "valid": list(range(345, 377)) + list(range(533, 564)),   # 63
    "test":  list(range(377, 408)) + list(range(471, 533)),   # 93
}


# ── dataset ───────────────────────────────────────────────────────────────────
class COCOSegDataset(torch.utils.data.Dataset):
    """Returns (pil_img, sem_np_remapped[HxW int64], image_id).

    Provide either `split` (train/valid/test) OR an explicit `image_ids` list
    (used for the cal(30)/val(33) sub-splits).
    """
    def __init__(self, split=None, image_ids=None,
                 coco_file=COCO_FILE, root_dir=DATA_ROOT):
        with open(coco_file) as f:
            data = json.load(f)
        self.root_dir = root_dir
        if image_ids is not None:
            keep = set(int(i) for i in image_ids)
        else:
            keep = set(SPLIT_IDS[split])
        self.images = [im for im in data["images"] if im["id"] in keep]
        self.images.sort(key=lambda im: im["id"])          # deterministic order
        self.ann_lookup = {}
        for ann in data["annotations"]:
            self.ann_lookup.setdefault(ann["image_id"], []).append(ann)

    def __len__(self):
        return len(self.images)

    def image_ids(self):
        return [im["id"] for im in self.images]

    def __getitem__(self, idx):
        info = self.images[idx]
        H, W = info["height"], info["width"]
        rel = info["path"].lstrip("/datasets/")
        pil = Image.open(os.path.join(self.root_dir, rel)).convert("RGB")
        sem = np.zeros((H, W), dtype=np.uint8)
        for ann in self.ann_lookup.get(info["id"], []):
            for poly in ann.get("segmentation", []):
                pts = np.array(poly).reshape(-1, 2)
                rr, cc = skimage.draw.polygon(pts[:, 1], pts[:, 0], sem.shape)
                sem[rr, cc] = ann["category_id"]
        sem = _REMAP_LUT[sem.astype(np.int64)]             # -> 0..7
        return pil, sem, info["id"]


# ── split manifest (valid -> cal + val), fixed across model seeds ─────────────
def make_split(split_seed=2024, n_cal=30):
    """Deterministically partition the 63 valid ids into cal(n_cal)+val(rest)."""
    vids = sorted(SPLIT_IDS["valid"])
    rng = np.random.RandomState(split_seed)
    perm = rng.permutation(len(vids))
    cal = sorted(int(vids[i]) for i in perm[:n_cal])
    val = sorted(int(vids[i]) for i in perm[n_cal:])
    return {"split_seed": split_seed, "n_cal": len(cal), "n_val": len(val),
            "cal_ids": cal, "val_ids": val,
            "train_ids": sorted(SPLIT_IDS["train"]),
            "test_ids": sorted(SPLIT_IDS["test"])}


def load_split():
    with open(SPLIT_MANIFEST) as f:
        return json.load(f)


# ── model ─────────────────────────────────────────────────────────────────────
def load_pepper_model(ckpt_path, device, pretrained="facebook/mask2former-swin-base-ade-semantic"):
    model = Mask2FormerForUniversalSegmentation.from_pretrained(
        pretrained, id2label=ID2LABEL, ignore_mismatched_sizes=True)
    model.load_state_dict(torch.load(ckpt_path, map_location="cpu"))
    model.eval().to(device)
    for p in model.parameters():
        p.requires_grad = False
    return model


def build_seg_probs(outputs, H, W):
    """8-class semantic probs (B,C,H,W), same recipe as calibration_shift_eval."""
    cp = outputs.class_queries_logits.softmax(dim=-1)[..., :-1]
    mp = outputs.masks_queries_logits.sigmoid()
    seg = torch.einsum("bqc,bqhw->bchw", cp, mp)
    seg = seg / (seg.sum(dim=1, keepdim=True) + 1e-6)
    return F.interpolate(seg, size=(H, W), mode="bilinear", align_corners=False)


def to_pseudologits(seg):
    """(B,NUM,H,W) prob -> pseudo-logits log(p). No reduction (multi-class)."""
    seg = seg / (seg.sum(dim=1, keepdim=True) + 1e-6)
    return torch.log(seg + 1e-6)


def _tfm():
    return transforms.Compose([transforms.ToTensor(),
                               transforms.Normalize(mean=ADE_MEAN, std=ADE_STD)])


@torch.no_grad()
def predict_cache(model, dataset, indices, device, cal_size,
                  max_images=None, corrupt_fn=None, corrupt_param=None,
                  binary_anomaly=False):
    """Run model on dataset[indices]; cache low-res 8-class pseudo-logits + GT + image.

    binary_anomaly=True: returns GT as (sem>0) anomaly mask (used for pixel-OoD on
    tomato/growliflower). Otherwise returns the remapped multi-class label map.
    """
    from tqdm import tqdm
    proc = Mask2FormerImageProcessor(ignore_index=255, reduce_labels=False,
                                     do_resize=False, do_rescale=False,
                                     do_normalize=False, num_labels=NUM)
    tfm = _tfm()
    logit_list, label_list, img_list = [], [], []
    if max_images is not None:
        indices = indices[:max_images]
    for idx in tqdm(indices, desc="predict"):
        pil, sem_np, _ = dataset[idx]
        W0, H0 = pil.size
        if corrupt_fn is not None:
            pil = corrupt_fn(pil, corrupt_param)
            if pil.size != (W0, H0):
                pil = pil.resize((W0, H0), Image.BILINEAR)
        W, H = pil.size
        pv = proc([tfm(pil)], return_tensors="pt")["pixel_values"].to(device)
        seg = build_seg_probs(model(pixel_values=pv), H, W)      # (1,NUM,H,W)
        plog = to_pseudologits(seg)
        if binary_anomaly:
            gt = torch.from_numpy((sem_np > 0).astype(np.int64))[None, None].float()
        else:
            gt = torch.from_numpy(sem_np.astype(np.int64))[None, None].float()
        im = tfm(pil)[None]
        h = cal_size
        w = int(round(cal_size * W / H))
        plog = F.interpolate(plog, size=(h, w), mode="bilinear", align_corners=False)
        gt = F.interpolate(gt, size=(h, w), mode="nearest").long()[:, 0]
        im = F.interpolate(im.to(device), size=(h, w), mode="bilinear", align_corners=False)
        logit_list.append(plog.squeeze(0).half().cpu())
        label_list.append(gt.squeeze(0).cpu())
        img_list.append(im.squeeze(0).half().cpu())
    return {"logits": torch.stack(logit_list),
            "labels": torch.stack(label_list),
            "image":  torch.stack(img_list)}


# ── metrics (multi-class) ─────────────────────────────────────────────────────
def compute_metrics(cal_logits, labels, num=NUM, n_bins=10):
    """cal_logits (N,num,h,w), labels (N,h,w) -> dict(ECE, ACE, mIoU, acc)."""
    from torchmetrics.classification import MulticlassCalibrationError
    from torch_uncertainty.metrics.classification.adaptive_calibration_error import AdaptiveCalibrationError
    import evaluate as hf_evaluate
    probs = F.softmax(cal_logits.float(), dim=1)
    preds = probs.argmax(1)
    valid = labels != 255
    pf = probs.permute(0, 2, 3, 1).reshape(-1, num)
    tf = labels.reshape(-1)
    m = tf != 255
    pf, tf = pf[m], tf[m]
    ece = MulticlassCalibrationError(num_classes=num, n_bins=n_bins, norm="l1")(pf, tf).item()
    ace = AdaptiveCalibrationError(task="multiclass", num_classes=num, num_bins=n_bins, norm="l1")(pf, tf).item()
    metric = hf_evaluate.load("mean_iou")
    metric.add_batch(predictions=[preds[i].numpy() for i in range(len(preds))],
                     references=[labels[i].numpy() for i in range(len(labels))])
    miou = metric.compute(num_labels=num, ignore_index=255)["mean_iou"]
    acc = (preds[valid] == labels[valid]).float().mean().item()
    return {"ECE": ece, "ACE": ace, "mIoU": float(miou), "acc": acc}
