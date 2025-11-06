#!/usr/bin/env python3
import argparse, json, os, math, colorsys
import numpy as np
import cv2 as cv
from PIL import Image
from scipy import ndimage as ndi

import torch
from transformers import AutoImageProcessor, SegformerForSemanticSegmentation

# ========================== Segmentation (planes) ==========================

MODEL_ID = "nvidia/segformer-b5-finetuned-ade-640-640"
FLOOR_NAME, WALL_NAME, CEILING_NAME = "floor", "wall", "ceiling"

def make_color(i, total, v=0.85):
    h = (i % total) / float(max(total, 1)); s = 0.65
    r, g, b = colorsys.hsv_to_rgb(h, s, v)
    return (int(255*r), int(255*g), int(255*b))

def imwrite_ok(path, img, tag=""):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    ok = img is not None and cv.imwrite(path, img)
    print(f"[SAVE {'OK' if ok else 'FAIL'}] {tag} -> {os.path.abspath(path)}")
    return ok

def dump_unique(pred, path_txt):
    vals, cnt = np.unique(pred, return_counts=True)
    with open(path_txt, "w") as f:
        for v, c in sorted(zip(vals, cnt), key=lambda x: -x[1])[:200]:
            f.write(f"id={int(v)}  pixels={int(c)}\n")
    print(f"[STATS] Wrote {path_txt}")

def load_model(device):
    processor = AutoImageProcessor.from_pretrained(MODEL_ID)
    model = SegformerForSemanticSegmentation.from_pretrained(MODEL_ID)
    model.to(device); model.eval()
    id2label = model.config.id2label
    label2id = {v.lower(): int(k) for k, v in id2label.items()}
    return processor, model, id2label, label2id

def run_segmentation(img_bgr, processor, model, device):
    img_rgb = cv.cvtColor(img_bgr, cv.COLOR_BGR2RGB)
    pil = Image.fromarray(img_rgb)
    with torch.no_grad():
        inputs = processor(images=pil, return_tensors="pt").to(device)
        logits = model(**inputs).logits
        up = torch.nn.functional.interpolate(
            logits, size=pil.size[::-1], mode="bilinear", align_corners=False
        )
        pred = up.argmax(dim=1)[0].cpu().numpy().astype(np.int32)
    return pred

def clean_mask(mask, k=5, min_area=400):
    if mask is None: return None
    m = cv.morphologyEx(mask.astype(np.uint8), cv.MORPH_OPEN,
                        cv.getStructuringElement(cv.MORPH_ELLIPSE, (k, k)))
    n, lb, stats, _ = cv.connectedComponentsWithStats(m, connectivity=4)
    out = np.zeros_like(m)
    for i in range(1, n):
        if stats[i, cv.CC_STAT_AREA] >= min_area:
            out[lb == i] = 255
    return out

def split_instances(mask, min_area=800):
    if mask is None: return []
    n, lb, stats, _ = cv.connectedComponentsWithStats(mask, connectivity=4)
    inst = []
    for i in range(1, n):
        if stats[i, cv.CC_STAT_AREA] >= min_area:
            m = np.zeros_like(mask); m[lb == i] = 255
            inst.append(m)
    return inst

def overlay_planes(img_bgr, floor_m, wall_ms, ceil_m, alpha=0.35, edge=2):
    canvas = img_bgr.copy()
    layers, colors = [], []
    if floor_m is not None: layers.append(floor_m); colors.append((120,255,120))
    for i, wm in enumerate(wall_ms): layers.append(wm); colors.append(make_color(i, max(3, len(wall_ms)+2)))
    if ceil_m is not None: layers.append(ceil_m); colors.append((200,200,255))
    for m, col in zip(layers, colors):
        col_img = np.zeros_like(canvas); col_img[:] = col
        mask_bool = (m > 0)
        canvas[mask_bool] = (canvas[mask_bool].astype(np.float32)*(1-alpha) +
                             col_img[mask_bool].astype(np.float32)*alpha).astype(np.uint8)
        cnts,_ = cv.findContours(m, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
        cv.drawContours(canvas, cnts, -1, (40,220,40), edge)
    return canvas

# ============================ Script main ================================

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--image", required=True)
    ap.add_argument("--out_dir", default="outputs")
    ap.add_argument("--cpu", action="store_true")
    ap.add_argument("--min_wall_area", type=int, default=1200)
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    device = "cpu" if (args.cpu or not torch.cuda.is_available()) else "cuda"

    img = cv.imread(args.image, cv.IMREAD_COLOR)
    if img is None: raise SystemExit(f"Cannot read image: {args.image}")

    processor, model, id2label, label2id = load_model(device)
    pred = run_segmentation(img, processor, model, device)

    dump_unique(pred, os.path.join(args.out_dir, "pred_ids.txt"))
    pred_vis = np.stack([(pred*37)%255, (pred*17)%255, (pred*97)%255], axis=-1).astype(np.uint8)
    imwrite_ok(os.path.join(args.out_dir, "pred_vis.png"), pred_vis, "pred_vis")

    def find_ids_by_prefix(prefix, label2id, id2label):
        p = prefix.lower()
        ids = [i for name,i in label2id.items() if name.startswith(p)]
        if not ids:
            for k,v in id2label.items():
                if v.lower().startswith(p): ids.append(int(k))
        return sorted(set(int(i) for i in ids))

    floor_ids = find_ids_by_prefix("floor", label2id, id2label)
    wall_ids  = find_ids_by_prefix("wall",  label2id, id2label)
    ceil_ids  = find_ids_by_prefix("ceiling", label2id, id2label)
    print("[IDS]", "floor:", floor_ids, "wall:", wall_ids, "ceiling:", ceil_ids)

    floor_raw = (np.isin(pred, floor_ids).astype(np.uint8)*255) if floor_ids else None
    wall_raw  = (np.isin(pred, wall_ids ).astype(np.uint8)*255) if wall_ids  else None
    ceil_raw  = (np.isin(pred, ceil_ids).astype(np.uint8)*255) if ceil_ids else None
    imwrite_ok(os.path.join(args.out_dir, "floor_raw.png"), floor_raw, "floor_raw") if floor_raw is not None else None
    imwrite_ok(os.path.join(args.out_dir, "wall_raw.png" ), wall_raw , "wall_raw" ) if wall_raw  is not None else None
    imwrite_ok(os.path.join(args.out_dir, "ceiling_raw.png"), ceil_raw, "ceiling_raw") if ceil_raw is not None else None

    floor_m = clean_mask(floor_raw) if floor_raw is not None else None
    wall_m  = clean_mask(wall_raw ) if wall_raw  is not None else None
    ceil_m  = clean_mask(ceil_raw ) if ceil_raw  is not None else None
    imwrite_ok(os.path.join(args.out_dir, "floor_clean.png"), floor_m, "floor_clean") if floor_m is not None else None
    imwrite_ok(os.path.join(args.out_dir, "wall_clean.png" ), wall_m , "wall_clean" ) if wall_m  is not None else None
    imwrite_ok(os.path.join(args.out_dir, "ceiling_clean.png"), ceil_m, "ceiling_clean") if ceil_m is not None else None

    wall_instances = []
    if wall_m is not None:
        wall_instances = split_instances(wall_m, min_area=args.min_wall_area)
    planes_overlay = overlay_planes(img, floor_m, wall_instances, ceil_m, alpha=0.35, edge=2)
    imwrite_ok(os.path.join(args.out_dir, "planes_overlay.png"), planes_overlay, "planes_overlay")

if __name__ == "__main__":
    main()
