# -*- coding: utf-8 -*-
"""
Experiment 3b — Mask2Former / BUP20 component sensitivity UNDER SHIFT.

Takes one Experiment-2 catalog config (index into configs/m2f.json), loads its LAST-FOUR-snapshot
ST-LoRA ensemble (shots 2,3,4,5) for one seed, and evaluates it on the BUP20 test split under the
19 discrete shift conditions (Exp3 table). Photometric shifts corrupt the image only; geometric
shifts corrupt image AND mask (nearest-neighbour). Reports mIoU + ECE (float64 StreamBinMetrics)
per condition. SLURM/wrapped only (ssl env).

Usage:
    python shift_eval_m2f_bup20.py --index 0 --seed 42 \
        --results_dir <exp2 m2f output base> --coco_file ... --root_dir ...
"""
import os
import sys
import json
import argparse

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import skimage.draw

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import calibration_shift_eval as cse                      # constants + model loaders + build_seg_probs
from run_seed_calibration_stream import StreamBinMetrics  # float64 ECE/mIoU
import shift_corruptions as sc

TEST_IDS = list(range(377, 408)) + list(range(471, 533))   # 93 BUP20 test images


class BUP20ShiftDataset(Dataset):
    """BUP20 test images + polygon masks, with one corruption applied (geometric -> image+mask)."""
    def __init__(self, coco_file, root_dir, corr_name, param):
        data = json.load(open(coco_file))
        ids = set(TEST_IDS)
        self.images = [im for im in data["images"] if im["id"] in ids]
        self.root_dir = root_dir
        self.corr_name = corr_name
        self.param = param
        self.ann = {}
        for a in data["annotations"]:
            self.ann.setdefault(a["image_id"], []).append(a)
        self.normalize = transforms.Compose([
            transforms.ToTensor(), transforms.Normalize(mean=cse.ADE_MEAN, std=cse.ADE_STD)])

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        info = self.images[idx]
        H, W = info["height"], info["width"]
        rel = info["path"].lstrip("/datasets/")
        img = Image.open(os.path.join(self.root_dir, rel)).convert("RGB")
        sem = np.zeros((H, W), dtype=np.uint8)
        for a in self.ann.get(info["id"], []):
            for poly in a.get("segmentation", []):
                pts = np.array(poly).reshape(-1, 2)
                rr, cc = skimage.draw.polygon(pts[:, 1], pts[:, 0], sem.shape)
                sem[rr, cc] = a["category_id"]
        seg = cse._REMAP_LUT[sem.astype(np.int64)].astype(np.uint8)   # 8-class remap

        photometric = self.corr_name in sc.PHOTOMETRIC
        img, seg2 = sc.apply(self.corr_name, img, None if photometric else seg, self.param)
        if not photometric:
            seg = seg2
        if img.size != (W, H):
            img = img.resize((W, H), Image.BILINEAR)
        return self.normalize(img), torch.from_numpy(seg.astype(np.int64)).long()


def build_members(results_dir, config_name, seed, shots, pretrained, dev):
    """Load the last-four LoRA snapshots as ready eval models (reuses cse loaders)."""
    seed_dir = os.path.join(results_dir, config_name, f"seed_{seed}")
    members = []
    for s in shots:
        adapter = os.path.join(seed_dir, f"model_shot_{s}")
        if not os.path.isdir(adapter):
            raise FileNotFoundError(adapter)
        members.append(cse.load_lora_adapter(cse.load_base_model(pretrained, dev), adapter, dev))
    return members


@torch.no_grad()
def eval_condition(members, loader, H, W, num, dev):
    """Mean-softmax ensemble over members, streamed; return metrics dict for one condition."""
    acc = StreamBinMetrics(num_classes=num)
    for images, seg in loader:
        pv = images.to(dev)
        probs = None
        for m in members:
            p = cse.build_seg_probs(m(pv), H, W)
            probs = p if probs is None else probs + p
        probs = (probs / len(members)).cpu()
        for i in range(probs.shape[0]):
            acc.update(probs[i:i + 1], seg[i:i + 1])
    return acc.compute()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--index", type=int, required=True, help="index into configs/m2f.json")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--results_dir", required=True, help="Exp-2 M2F output base (contains <config>/seed_X)")
    p.add_argument("--catalog", default=os.path.join(os.path.dirname(HERE), "configs", "m2f.json"))
    p.add_argument("--coco_file", required=True)
    p.add_argument("--root_dir", required=True)
    p.add_argument("--pretrained", default="facebook/mask2former-swin-base-ade-semantic")
    p.add_argument("--shots", nargs="+", type=int, default=[2, 3, 4, 5])
    p.add_argument("--height", type=int, default=1280)
    p.add_argument("--width", type=int, default=720)
    p.add_argument("--batch_size", type=int, default=2)
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--out_dir", default=None)
    args = p.parse_args()

    cfgs = json.load(open(args.catalog))
    cfg = cfgs[args.index]
    name = cfg["name"]
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    num = cse.NUM_LABELS

    print(f"=== EXP3b shift eval: m2f/{name} seed={args.seed} shots={args.shots} ===", flush=True)
    members = build_members(args.results_dir, name, args.seed, args.shots, args.pretrained, dev)

    results = {"arch": "m2f", "config": name, "seed": args.seed, "shots": args.shots,
               "eval": "BUP20_test_shift", "conditions": {}}
    for cname, level, param in sc.iter_conditions(sc.BUP20_SHIFTS):
        ds = BUP20ShiftDataset(args.coco_file, args.root_dir, cname, param)
        loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False,
                            num_workers=args.num_workers)
        m = eval_condition(members, loader, args.height, args.width, num, dev)
        key = f"{cname}_l{level}"
        results["conditions"][key] = {"corruption": cname, "level": level, "param": param,
                                      "mIoU": m["mIoU"], "ECE": m["ECE"]}
        print(f"  [{key}] param={param} mIoU={m['mIoU']:.4f} ECE={m['ECE']:.4f}", flush=True)

    out_dir = args.out_dir or os.path.join(args.results_dir, name, f"seed_{args.seed}")
    os.makedirs(out_dir, exist_ok=True)
    outp = os.path.join(out_dir, "shift_eval_bup20.json")
    json.dump(results, open(outp, "w"), indent=2)
    print(f"[done] wrote {outp}", flush=True)


if __name__ == "__main__":
    main()
