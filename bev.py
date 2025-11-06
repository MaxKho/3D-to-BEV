#!/usr/bin/env python3
import argparse, json, os, math
import numpy as np
import cv2 as cv

def _normH(p): return p/(p[...,2:3]+1e-12) if p.shape[-1]==3 else p
def _intersect(L1, L2): return _normH(np.cross(L1, L2))

def imwrite_ok(path, img, tag=""):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    ok = img is not None and cv.imwrite(path, img)
    print(f"[SAVE {'OK' if ok else 'FAIL'}] {tag} -> {os.path.abspath(path)}")
    return ok

def bev_from_floor_quad(img, floor_mask, vps, out_w=1500):
    fm = (floor_mask > 127).astype(np.uint8)
    cnts,_ = cv.findContours(fm, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    if not cnts: return None, None, None
    hull = cv.convexHull(max(cnts, key=cv.contourArea)).reshape(-1,2).astype(np.float64)
    if len(hull) < 4: return None, None, None
    def line_through(vp, p): return np.cross(vp, np.array([p[0], p[1], 1.0], float))
    def _finite_all(A): return np.isfinite(A).all()
    best, best_area = None, -1.0
    idxs = np.linspace(0, len(hull)-1, min(48, len(hull)), dtype=int)
    for a in idxs:
        for b in idxs:
            if (b - a) % len(hull) < len(hull)//4: continue
            p0, p1 = hull[a], hull[b]
            Lx0, Ly0 = line_through(vps['x'], p0), line_through(vps['y'], p0)
            Lx1, Ly1 = line_through(vps['x'], p1), line_through(vps['y'], p1)
            q00 = _intersect(Lx0, Ly0).ravel()[:2]
            q10 = _intersect(Lx1, Ly0).ravel()[:2]
            q11 = _intersect(Lx1, Ly1).ravel()[:2]
            q01 = _intersect(Lx0, Ly1).ravel()[:2]
            Q = np.array([q00,q10,q11,q01], np.float64)
            if not _finite_all(Q): continue
            x,y = Q[:,0], Q[:,1]
            area = 0.5*abs(np.dot(x, np.roll(y,-1)) - np.dot(y, np.roll(x,-1)))
            if area > best_area: best, best_area = Q, area
    if best is None or best_area < 1.0: return None, None, None
    a = np.linalg.norm(best[1]-best[0]); b = np.linalg.norm(best[3]-best[0])
    out_h = int(max(200, min(3000, out_w * (b/(a+1e-9)))))
    Q = best.astype(np.float32)
    R = np.array([[0,0],[out_w,0],[out_w,out_h],[0,out_h]], np.float32)
    H = cv.getPerspectiveTransform(Q, R)
    bev = cv.warpPerspective(img, H, (out_w, out_h), flags=cv.INTER_LINEAR, borderMode=cv.BORDER_REPLICATE)
    fm255 = (floor_mask>127).astype(np.uint8)*255
    bev_floor = cv.warpPerspective(fm255, H, (out_w, out_h), flags=cv.INTER_NEAREST, borderMode=cv.BORDER_CONSTANT, borderValue=0)
    ones = np.ones(img.shape[:2], np.uint8)*255
    bev_foot = cv.warpPerspective(ones, H, (out_w, out_h), flags=cv.INTER_NEAREST, borderMode=cv.BORDER_CONSTANT, borderValue=0)
    free = (bev_floor>127).astype(np.uint8)
    occ = ((bev_foot>127) & (1-free)).astype(np.uint8)*255
    return bev, occ, H

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--image", required=True)
    ap.add_argument("--out_dir", default="outputs")
    ap.add_argument("--bev_width", type=int, default=1500)
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    img = cv.imread(args.image, cv.IMREAD_COLOR)
    if img is None: raise SystemExit(f"Cannot read image: {args.image}")

    floor_mask = cv.imread(os.path.join(args.out_dir, "empty_floor.png"), cv.IMREAD_GRAYSCALE)
    if floor_mask is None or (floor_mask>0).sum()==0:
        raise SystemExit("[ERROR] empty_floor.png missing. Run step2 first.")

    with open(os.path.join(args.out_dir, "vps.json"), "r") as f:
        vp_data = json.load(f)

    def _from_list(v):
        return None if v is None else np.array([float(v[0]), float(v[1]), float(v[2])], float)
    vps = {'x': _from_list(vp_data.get('x')),
           'y': _from_list(vp_data.get('y')),
           'z': _from_list(vp_data.get('z'))}

    def _finite_xy(v):
        return (v is not None) and np.all(np.isfinite(v[:2])) and (abs(v[0]) < 1e5) and (abs(v[1]) < 1e5)
    if not (_finite_xy(vps['x']) and _finite_xy(vps['y'])):
        raise SystemExit("[BEV] skipped: floor VPs ill-conditioned or missing")

    bev, occ, _ = bev_from_floor_quad(img, floor_mask, vps, out_w=args.bev_width)
    imwrite_ok(os.path.join(args.out_dir, "bev.png"), bev, "bev")
    imwrite_ok(os.path.join(args.out_dir, "occ.png"), occ, "occ")
    print("[DONE] outputs ->", os.path.abspath(args.out_dir))

if __name__ == "__main__":
    main()
