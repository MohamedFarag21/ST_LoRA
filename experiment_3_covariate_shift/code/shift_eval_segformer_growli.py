# -*- coding: utf-8 -*-
"""
Experiment 3a — SegFormer-B2 / GrowliFlower-L, homogeneous vs heterogeneous ensembles UNDER SHIFT.

Evaluates one model's LAST-FOUR-snapshot ST-LoRA (or FRE) ensemble (shots 2,3,4,5) on the
GrowliFlower-L test split under the continuous 20-level shift grid (brightness/contrast/noise +
geometric rotation/zoom). Photometric shifts corrupt the image only; geometric shifts corrupt image
AND mask (nearest-neighbour). Binary plant-vs-background; reports mIoU + ECE (float64
StreamBinMetrics) per condition. SLURM/wrapped only (ssl env).

Usage:
    python shift_eval_segformer_growli.py --method stlora --variant heterogeneous --seed 42 \
        --results_dir <segformer growli output base> --root_dir /path/to/growliflower_l
"""
import os
import sys
import json
import argparse

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import segformer_growli_eval_seeded as ge                 # base/single-model loaders
from run_seed_calibration_stream import StreamBinMetrics
import shift_corruptions as sc

ADE_MEAN = np.array([123.675, 116.280, 103.530]) / 255
ADE_STD  = np.array([58.395,  57.120,  57.375])  / 255
NUM_LABELS = 2
IGNORE = 255
SPLITS = {"train": "Train", "valid": "Val", "test": "Test"}


class GrowliShiftDataset(Dataset):
    def __init__(self, root_dir, split, corr_name, param, resize=None):
        self.img_dir = os.path.join(root_dir, "images", SPLITS[split])
        self.lbl_dir = os.path.join(root_dir, "labels", SPLITS[split])
        self.corr_name, self.param, self.resize = corr_name, param, resize
        self.stems = sorted(os.path.splitext(f)[0] for f in os.listdir(self.img_dir)
                            if f.lower().endswith((".jpg", ".jpeg", ".png")))
        self.normalize = transforms.Compose([
            transforms.ToTensor(), transforms.Normalize(mean=ADE_MEAN, std=ADE_STD)])

    def __len__(self):
        return len(self.stems)

    def _mask(self, mt, stem):
        p = os.path.join(self.lbl_dir, mt, f"{stem}_Label_{mt}.png")
        return (np.array(Image.open(p)) > 0) if os.path.exists(p) else None

    def __getitem__(self, idx):
        stem = self.stems[idx]
        img = Image.open(os.path.join(self.img_dir, f"{stem}.jpg")).convert("RGB")
        W, H = img.size
        seg = np.zeros((H, W), dtype=np.uint8)
        plant = self._mask("maskPlants", stem)
        if plant is not None: seg[plant] = 1
        void = self._mask("maskVoid", stem)
        if void is not None: seg[void] = IGNORE
        if self.resize is not None:
            rh, rw = self.resize
            img = img.resize((rw, rh), Image.BILINEAR)
            seg = np.array(Image.fromarray(seg).resize((rw, rh), Image.NEAREST))
            W, H = rw, rh

        photometric = self.corr_name in sc.PHOTOMETRIC
        img, seg2 = sc.apply(self.corr_name, img, None if photometric else seg, self.param)
        if not photometric:
            seg = seg2
        if img.size != (W, H):
            img = img.resize((W, H), Image.BILINEAR)
        return self.normalize(img), torch.from_numpy(seg.astype(np.int64)).long()


def build_members(method, pretrained, results_dir, seed, shots, dev):
    seed_dir = os.path.join(results_dir, f"seed_{seed}")
    return [ge.load_single_model(method, pretrained, seed_dir, s, dev) for s in shots]


@torch.no_grad()
def eval_condition(members, loader, dev):
    acc = StreamBinMetrics(num_classes=NUM_LABELS)
    for images, seg in loader:
        pv = images.to(dev)
        H, W = images.shape[2], images.shape[3]
        probs = None
        for m in members:
            logits = F.interpolate(m(pixel_values=pv).logits, size=(H, W),
                                   mode="bilinear", align_corners=False)
            p = logits.softmax(dim=1)
            probs = p if probs is None else probs + p
        probs = (probs / len(members)).cpu()
        for i in range(probs.shape[0]):
            acc.update(probs[i:i + 1], seg[i:i + 1])
    return acc.compute()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--method", required=True, choices=["stlora", "fre"])
    p.add_argument("--variant", default="", help="homogeneous|heterogeneous (naming only)")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--results_dir", required=True, help="base dir containing seed_<seed>/model_shot_*")
    p.add_argument("--root_dir", required=True, help="GrowliFlower-L root")
    p.add_argument("--pretrained", default="nvidia/segformer-b2-finetuned-ade-512-512")
    p.add_argument("--shots", nargs="+", type=int, default=[2, 3, 4, 5])
    p.add_argument("--resize", nargs=2, type=int, default=None, metavar=("H", "W"))
    p.add_argument("--batch_size", type=int, default=8)
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--gpu", type=str, default="0")
    p.add_argument("--out_dir", default=None)
    args = p.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    resize = tuple(args.resize) if args.resize else None

    print(f"=== EXP3a shift eval: segformer/{args.method} variant={args.variant} "
          f"seed={args.seed} shots={args.shots} ===", flush=True)
    members = build_members(args.method, args.pretrained, args.results_dir, args.seed, args.shots, dev)

    results = {"arch": "segformer_b2", "method": args.method, "variant": args.variant,
               "seed": args.seed, "shots": args.shots, "eval": "GrowliFlower_test_shift",
               "active_corruptions": sc.GROWLI_ACTIVE, "conditions": {}}
    for cname, level, param in sc.iter_conditions(sc.GROWLI_SHIFTS, active=sc.GROWLI_ACTIVE):
        ds = GrowliShiftDataset(args.root_dir, "test", cname, param, resize=resize)
        loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
        m = eval_condition(members, loader, dev)
        key = f"{cname}_l{level}"
        results["conditions"][key] = {"corruption": cname, "level": level, "param": param,
                                      "mIoU": m["mIoU"], "ECE": m["ECE"]}
        print(f"  [{key}] param={param} mIoU={m['mIoU']:.4f} ECE={m['ECE']:.4f}", flush=True)

    out_dir = args.out_dir or os.path.join(args.results_dir, f"seed_{args.seed}")
    os.makedirs(out_dir, exist_ok=True)
    outp = os.path.join(out_dir, f"shift_eval_growli_{args.method}.json")
    json.dump(results, open(outp, "w"), indent=2)
    print(f"[done] wrote {outp}", flush=True)


if __name__ == "__main__":
    main()
