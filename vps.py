#!/usr/bin/env python3
# vps.py — uses plane masks to compute vanishing points and save vp_overlay.png + vps.json

import argparse, json, os, math
import numpy as np
import cv2 as cv
from PIL import Image  # not used directly but kept for parity
from scipy import ndimage as ndi

# ========================== small I/O utils ==========================

def imwrite_ok(path, img, tag=""):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    ok = img is not None and cv.imwrite(path, img)
    print(f"[SAVE {'OK' if ok else 'FAIL'}] {tag} -> {os.path.abspath(path)}")
    return ok

def _vp_px(v):
    return None if v is None else (float(v[0]/(v[2]+1e-12)), float(v[1]/(v[2]+1e-12)))

def _normH(p): return p/(p[...,2:3]+1e-12) if p.shape[-1]==3 else p
def _intersect(L1, L2): return _normH(np.cross(L1, L2))

# ===================== helpers reused from original =====================

def make_empty_room_floor(floor_mask, walls_mask=None, feet_mask=None):
    fm = (floor_mask > 127).astype(np.uint8) * 255
    h, w = fm.shape
    inv = cv.morphologyEx(cv.bitwise_not(fm), cv.MORPH_OPEN,
                          cv.getStructuringElement(cv.MORPH_ELLIPSE, (max(3,int(0.004*min(h,w))),)*2))
    fm = cv.bitwise_not(inv)
    cnts,_ = cv.findContours(fm, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    if cnts:
        cnt = max(cnts, key=cv.contourArea)
        canvas = np.zeros_like(fm); cv.drawContours(canvas, [cnt], -1, 255, cv.FILLED)
        fm = cv.morphologyEx(canvas, cv.MORPH_CLOSE,
                             cv.getStructuringElement(cv.MORPH_ELLIPSE, (max(5,int(0.012*min(h,w)))*3,)*2))
    if feet_mask is not None:
        fm = cv.bitwise_and(fm, cv.bitwise_not((feet_mask>127).astype(np.uint8)*255))
    if walls_mask is not None:
        wm_er = cv.erode((walls_mask>127).astype(np.uint8)*255,
                         cv.getStructuringElement(cv.MORPH_RECT, (max(3,int(0.006*min(h,w))),)*2))
        fm = cv.bitwise_and(fm, cv.bitwise_not(wm_er))
    k3 = max(3, int(0.004 * min(h, w)))
    fm = cv.morphologyEx(fm, cv.MORPH_OPEN, cv.getStructuringElement(cv.MORPH_ELLIPSE, (k3,k3)))
    fm = cv.morphologyEx(fm, cv.MORPH_CLOSE, cv.getStructuringElement(cv.MORPH_ELLIPSE, (3*k3,3*k3)))
    edges = cv.Canny(fm, 50, 120)
    L = cv.HoughLinesP(edges, 1, np.pi/180, threshold=40,
                       minLineLength=int(0.06*min(h,w)), maxLineGap=int(0.02*min(h,w)))
    if L is not None:
        mask_lines = np.zeros_like(fm); thick = max(3, int(0.004 * min(h, w)))
        for x1,y1,x2,y2 in L[:,0]: cv.line(mask_lines, (x1,y1), (x2,y2), 255, thick)
        fm = cv.bitwise_or(fm, mask_lines)
    return fm

def build_boundary_distance(floor_mask, walls_mask, img_shape, dilate_px=2):
    h, w = img_shape[:2]
    bound = np.zeros((h, w), np.uint8)
    if floor_mask is not None: bound |= cv.Canny((floor_mask>127).astype(np.uint8)*255, 50, 150)
    if walls_mask is not None: bound |= cv.Canny((walls_mask>127).astype(np.uint8)*255, 50, 150)
    if dilate_px>0:
        bound = cv.dilate(bound, cv.getStructuringElement(cv.MORPH_ELLIPSE, (2*dilate_px+1,)*2))
    return ndi.distance_transform_edt((bound == 0).astype(np.uint8))

def _lsd_lines_global(bgr, floor_mask=None, walls_mask=None, max_keep=600, min_len_frac=0.05, score_bias=0.35):
    h, w = bgr.shape[:2]; min_len = float(min_len_frac) * np.hypot(h, w)
    gray = cv.cvtColor(bgr, cv.COLOR_BGR2GRAY)
    lsd = cv.createLineSegmentDetector(cv.LSD_REFINE_STD)
    segs, _, _, _ = lsd.detect(gray)
    if segs is None:
        print("[LSD] none"); return [], np.empty((0,4), np.float32), np.empty((0,), np.float32)
    segs = segs.reshape(-1,4).astype(np.float32)
    lens = np.hypot(segs[:,2]-segs[:,0], segs[:,3]-segs[:,1])
    keep = lens >= min_len
    segs, lens = segs[keep], lens[keep]
    print(f"[LSD] kept {len(segs)} segs")
    if len(segs) == 0:
        return [], np.empty((0,4), np.float32), np.empty((0,), np.float32)

    pri = np.zeros(len(segs), np.float32)
    if floor_mask is not None:
        fm = (floor_mask > 127).astype(np.uint8); edges = cv.Canny(fm*255, 50, 150)
        dist = ndi.distance_transform_edt(1 - (edges > 0))
        k = max(1, int(0.004 * min(h, w)))
        for i,(x1,y1,x2,y2) in enumerate(segs):
            H, W = dist.shape
            x1i, y1i = int(np.clip(x1, 0, W-1)), int(np.clip(y1, 0, H-1))
            x2i, y2i = int(np.clip(x2, 0, W-1)), int(np.clip(y2, 0, H-1))
            d = min(dist[y1i, x1i], dist[y2i, x2i]); pri[i] += np.exp(-d/(6.0*k))
    if walls_mask is not None:
        wm = (walls_mask > 127).astype(np.uint8)
        for i,(x1,y1,x2,y2) in enumerate(segs):
            if wm[int(np.clip(y1,0,h-1)), int(np.clip(x1,0,w-1))] or wm[int(np.clip(y2,0,h-1)), int(np.clip(x2,0,w-1))]:
                pri[i] += 1.0

    score = lens + score_bias * (pri * min_len)
    idx = np.argsort(-score)[:min(max_keep, len(segs))]
    segs, lens = segs[idx], lens[idx]

    lines = []
    for x1,y1,x2,y2 in segs:
        p1, p2 = np.array([x1,y1,1.0]), np.array([x2,y2,1.0])
        L = np.cross(p1, p2).astype(np.float64); L /= (np.linalg.norm(L[:2]) + 1e-12)
        lines.append(L)
    return lines, segs, lens

def _dominant_floor_axes_from_mask(floor_mask):
    fm = (floor_mask > 127).astype(np.uint8) * 255
    e = cv.Canny(fm, 50, 150); h, w = fm.shape
    L = cv.HoughLines(e, 1, np.pi/180, threshold=max(60, int(0.08*min(h, w))))
    if L is None: return []
    angs = np.mod(L[:,0,1], np.pi); dirs = np.mod(angs + np.pi/2.0, np.pi)
    hist, bins = np.histogram(dirs, bins=90, range=(0,np.pi))
    peaks = hist.argsort()[-2:][::-1]
    ax = []
    for p in peaks:
        a0,a1 = bins[p], bins[p+1]; sel = (dirs >= a0) & (dirs < a1)
        if np.any(sel): ax.append(np.median(dirs[sel]))
    return ax

def _dominant_wall_axis_from_mask(walls_mask):
    if walls_mask is None: return None
    wm = (walls_mask > 127).astype(np.uint8)*255
    wminv = cv.bitwise_not(wm)
    def axis_from_bin(binimg):
        edges = cv.Canny(binimg, 50, 150)
        k = max(3, int(0.006 * min(binimg.shape[:2])))
        edges = cv.dilate(edges, cv.getStructuringElement(cv.MORPH_RECT,(k,k)))
        L = cv.HoughLinesP(edges, 1, np.pi/180, threshold=30,
                           minLineLength=int(0.05*min(*binimg.shape)),
                           maxLineGap=int(0.02*min(*binimg.shape)))
        if L is None: return None, 0
        dirs = []
        for x1,y1,x2,y2 in L[:,0]:
            a = math.atan2(y2-y1, x2-x1);  a = a + np.pi if a < 0 else a
            dirs.append(a)
        if not dirs: return None, 0
        return float(np.median(np.array(dirs))), len(dirs)
    a1,n1 = axis_from_bin(wm); a2,n2 = axis_from_bin(wminv)
    return a1 if n1 >= n2 else a2

def compute_vps_and_lines(
    bgr, floor_mask=None, walls_mask=None,
    draw_n=10, draw_wall_n=None,
    max_keep=700, min_len_frac=0.045, score_bias=0.35,
    max_boundary_px=None, max_boundary_frac=0.00, boundary_dilate_px=2,
    min_pair_angle_deg=10.0, cluster_radius_frac=0.035, min_intersections_per_vp=20,
    axis_valid_frac=0.60
):
    # --- LSD and intersections
    lines, segs, lens = _lsd_lines_global(
        bgr, floor_mask=floor_mask, walls_mask=walls_mask,
        max_keep=max_keep, min_len_frac=min_len_frac, score_bias=score_bias
    )
    if len(lines) < 8:
        print("[INT] abort: not enough lines")
        return {'x': None, 'y': None, 'z': None}, []

    h, w = bgr.shape[:2]; diag = float(np.hypot(h, w))
    min_pair_angle = math.radians(min_pair_angle_deg)

    L = np.asarray(lines, float)
    P, W = [], []
    for i in range(len(L)):
        a1,b1,_ = L[i]; v1 = np.array([-b1, a1]); v1 /= (np.linalg.norm(v1)+1e-12)
        for j in range(i+1, len(L)):
            a2,b2,_ = L[j]; v2 = np.array([-b2, a2]); v2 /= (np.linalg.norm(v2)+1e-12)
            ang = math.acos(np.clip(abs(v1 @ v2), 0.0, 1.0))
            if ang < min_pair_angle or (math.pi - ang) < min_pair_angle: continue
            p = np.cross(L[i], L[j])
            if not np.isfinite(p).all() or abs(p[2]) < 1e-12: continue
            p = p[:2] / p[2]
            wt = float(lens[i] * lens[j] * math.sin(ang))
            P.append(p); W.append(wt)
    if not P:
        print("[INT] none"); return {'x': None, 'y': None, 'z': None}, []
    print(f"[INT] intersections {len(P)}")

    # --- greedy radius clustering
    P = np.asarray(P, float); W = np.asarray(W, float)
    rad = max(8.0, cluster_radius_frac * diag)
    order = np.argsort(-W); used = np.zeros(len(P), np.uint8)
    centers, weights = [], []
    for idx in order:
        if used[idx]: continue
        d = np.hypot(P[:,0]-P[idx,0], P[:,1]-P[idx,1])
        mask = (d <= rad) & (~used.astype(bool))
        if mask.sum() < min_intersections_per_vp:
            used[mask] = 1; continue
        ww = W[mask]; pp = P[mask]
        c = (pp * ww[:,None]).sum(axis=0) / (ww.sum() + 1e-12)
        centers.append(c); weights.append(ww.sum()); used[mask] = 1
    if len(centers) < 1:
        print("[CLUST] none"); return {'x': None, 'y': None, 'z': None}, []
    print(f"[CLUST] centers {len(centers)}")

    keep = np.argsort(-np.asarray(weights))[:min(3, len(centers))]
    centers = [centers[i] for i in keep]
    vps = [np.array([c[0], c[1], 1.0], float) for c in centers]

    # --- assign lines to nearest VP
    Lnorm = np.asarray([l / (np.linalg.norm(l[:2]) + 1e-12) for l in lines], float)
    M = np.stack([np.abs(Lnorm @ vp) for vp in vps], axis=1)
    lab = np.argmin(M, axis=1)

    # --- mask-driven axis hints
    floor_axes = _dominant_floor_axes_from_mask(floor_mask) if floor_mask is not None else []
    wall_axis = _dominant_wall_axis_from_mask(walls_mask)

    dirs = np.arctan2(segs[:,3]-segs[:,1], segs[:,2]-segs[:,0]); dirs[dirs<0]+=np.pi
    def angsep(a,b): d = abs(a-b);  return min(d, np.pi-d)

    have_wall_bin = wall_axis is not None
    num_bins = 2 + (1 if have_wall_bin else 0)
    vp_axis = [-1]*len(vps); vp_axis_frac = [0.0]*len(vps)
    for k in range(len(vps)):
        ids = np.where(lab==k)[0]
        if len(ids)==0: continue
        dsub = dirs[ids]
        bins_angles = (floor_axes[:2] + [wall_axis]) if have_wall_bin else floor_axes[:2]
        nearest = []
        for dth in dsub:
            ad = [angsep(dth, ax) for ax in bins_angles]
            nearest.append(int(np.argmin(ad)))
        counts = np.bincount(np.array(nearest), minlength=num_bins)
        maj = int(np.argmax(counts)); frac = counts[maj]/max(1,counts.sum())
        vp_axis[k] = maj; vp_axis_frac[k] = float(frac)

    vertical_bin = 2 if have_wall_bin else (
        0 if len(floor_axes)<2 else int(np.argmax([min(angsep(a,floor_axes[0]), angsep(a,floor_axes[1])) for a in floor_axes]))
    )

    idxs_for_bin = {0:None, 1:None, 2:None}
    for k in range(len(vps)):
        b = vp_axis[k]
        if b == -1: continue
        if vp_axis_frac[k] >= axis_valid_frac:
            if idxs_for_bin.get(b) is None: idxs_for_bin[b] = k
            else:
                old = idxs_for_bin[b]
                if np.sum(lab==k) > np.sum(lab==old): idxs_for_bin[b] = k

    vz_idx = idxs_for_bin.get(vertical_bin)
    floor_bins = [0,1] if vertical_bin!=0 else [1,0]
    fx_idx, fy_idx = idxs_for_bin.get(floor_bins[0]), idxs_for_bin.get(floor_bins[1])

    def _px(i): return None if i is None else vps[i][0] / (vps[i][2] + 1e-12)
    if fx_idx is not None and fy_idx is not None and _px(fx_idx) > _px(fy_idx):
        fx_idx, fy_idx = fy_idx, fx_idx

    tag_to_idx = {'X': fx_idx, 'Y': fy_idx, 'Z': vz_idx}

    # --- pruning by VP-shared
    R = np.stack([np.abs(Lnorm @ vp) for vp in vps], axis=1) if len(vps)>0 else np.empty((len(Lnorm),0),float)
    NEAR_Q = 35.0
    def vp_shared_fraction(idx):
        if idx is None or R.shape[1] == 0: return float('nan')
        r_idx = R[:, idx]; thr_idx = np.percentile(r_idx, NEAR_Q)
        near = r_idx <= thr_idx
        if not np.any(near): return 0.0
        others = []
        for j in range(R.shape[1]):
            if j == idx: continue
            thr_j = np.percentile(R[:, j], NEAR_Q)
            others.append(R[:, j] <= thr_j)
        if not others: return 0.0
        shared = np.any(np.stack(others, axis=1), axis=1)
        return float((shared & near).sum()) / float(near.sum())
    for tag in ('X','Y','Z'):
        idx = tag_to_idx[tag]
        if idx is None: continue
        fs = vp_shared_fraction(idx)
        print(f"[VP shared] {tag}={0.0 if np.isnan(fs) else fs:.3f}")
        if not np.isnan(fs) and (fs > axis_valid_frac): tag_to_idx[tag] = None
    fx_idx, fy_idx, vz_idx = tag_to_idx['X'], tag_to_idx['Y'], tag_to_idx['Z']

    # ================== FALLBACKS KEPT ==================
    # 1) Synthesize missing floor VP by SVD or geometric backup
    if (fx_idx is None) ^ (fy_idx is None):
        have_wall_bin = wall_axis is not None
        bins_angles = (floor_axes[:2] + [wall_axis]) if have_wall_bin else floor_axes[:2]
        if len(bins_angles) >= 2:
            known_idx = fx_idx if fx_idx is not None else fy_idx
            known_bin = vp_axis[known_idx] if 0 <= vp_axis[known_idx] <= 1 else 0
            other_bin = 1 - known_bin
            def _angsep(a,b): d = abs(a-b); return min(d, np.pi-d)
            def nearest_bin(theta):
                ad = [_angsep(theta, ax) for ax in bins_angles]
                return int(np.argmin(ad))
            all_bins = np.array([nearest_bin(th) for th in dirs])
            cand_ids = np.where(all_bins == other_bin)[0]

            synth = None
            if len(cand_ids) >= 2:
                LS = np.asarray(lines, float); A = LS[cand_ids].astype(np.float64)
                try:
                    _, _, VT = np.linalg.svd(A, full_matrices=False)
                    v = VT[-1, :]
                    if abs(v[2]) >= 1e-12 and np.isfinite(v).all():
                        synth = np.array([v[0]/v[2], v[1]/v[2], 1.0], float)
                except np.linalg.LinAlgError:
                    synth = None
            if synth is None:
                xpix = vps[known_idx][:2]/(vps[known_idx][2]+1e-12)
                if len(cand_ids) >= 2:
                    angs = dirs[cand_ids]; c = np.cos(2*angs).mean(); s = np.sin(2*angs).mean()
                    theta_h = 0.5 * math.atan2(s, c);  theta_h = theta_h + np.pi if theta_h < 0 else theta_h
                else:
                    md = np.median(dirs[lab==known_idx]) if np.any(lab==known_idx) else 0.0
                    theta_h = (md + np.pi/2.0) % np.pi
                hdir = np.array([math.cos(theta_h), math.sin(theta_h)], float)
                y_est = xpix + 2000.0*hdir
                synth = np.array([y_est[0], y_est[1], 1.0], float)
            if fx_idx is None: vps.append(synth); fx_idx = len(vps)-1
            else: vps.append(synth); fy_idx = len(vps)-1

    # 2) If both floor VPs missing but Z and at least one floor axis exist
    if (fx_idx is None) and (fy_idx is None) and (vz_idx is not None) and (len(floor_axes) >= 1):
        h_, w_ = bgr.shape[:2]; cx, cy = w_*0.5, h_*0.5
        vz_xy = vps[vz_idx][:2]/(vps[vz_idx][2]+1e-12)
        r = np.array([vz_xy[0]-cx, vz_xy[1]-cy], float); r /= (np.linalg.norm(r)+1e-12)
        hdir = np.array([r[1], -r[0]])
        def line_from_point_dir(pt, v): a, b = v[1], -v[0]; c = -(a*pt[0] + b*pt[1]); return np.array([a,b,c], float)
        Lh = line_from_point_dir((cx, cy), hdir)
        def line_from_center_angle(theta):
            v = np.array([math.cos(theta), math.sin(theta)], float)
            return line_from_point_dir((cx, cy), v)
        ax_list = [floor_axes[0], (floor_axes[0]+np.pi/2.0)%np.pi] if len(floor_axes)==1 else floor_axes[:2]
        p_first = _intersect(Lh, line_from_center_angle(ax_list[0])).ravel()
        p_second = _intersect(Lh, line_from_center_angle(ax_list[1])).ravel()
        if p_first[2] != 0 and p_second[2] != 0:
            vps.append(p_first); fx_idx = len(vps)-1
            vps.append(p_second); fy_idx = len(vps)-1

    # 3) If Z missing but one floor VP exists, build Z from near-vertical lines
    if (vz_idx is None) and ((fx_idx is not None) ^ (fy_idx is not None)):
        if wall_axis is not None:
            sel = np.where(np.array([angsep(d, wall_axis) for d in dirs]) <= math.radians(12.0))[0]
        elif len(floor_axes) >= 1:
            def far_from_floor(d): return min([angsep(d,a) for a in floor_axes]) >= math.radians(30.0)
            sel = np.where(np.array([far_from_floor(d) for d in dirs]))[0]
        else:
            sel = np.arange(len(dirs))
        cand, cw = [], []
        LS = np.asarray(lines, float)
        for i in sel:
            for j in sel:
                if j <= i: continue
                v1 = np.array([-LS[i,1], LS[i,0]]); v1 /= (np.linalg.norm(v1)+1e-12)
                v2 = np.array([-LS[j,1], LS[j,0]]); v2 /= (np.linalg.norm(v2)+1e-12)
                a = math.acos(np.clip(abs(v1@v2), 0.0, 1.0))
                if a < math.radians(8.0): continue
                p = np.cross(LS[i], LS[j])
                if not np.isfinite(p).all() or abs(p[2]) < 1e-12: continue
                p = p[:2]/p[2]; wt = float(lens[i]*lens[j]*math.sin(a))
                cand.append(p); cw.append(wt)
        if cand:
            cand = np.asarray(cand, float); cw = np.asarray(cw, float)
            vz_pt = (cand * cw[:,None]).sum(axis=0) / (cw.sum()+1e-12)
            vps.append(np.array([vz_pt[0], vz_pt[1], 1.0], float)); vz_idx = len(vps)-1
    # ================== END FALLBACKS ==================

    # --------- RELABEL AFTER FALLBACKS (key fix)
    if len(vps) > 0:
        M_final = np.stack([np.abs(Lnorm @ vp) for vp in vps], axis=1)
        lab = np.argmin(M_final, axis=1)
        # quick debug viz
        lab_vis = np.zeros(bgr.shape[:2], np.uint8)
        for k in range(len(vps)):
            col = int(60 + 160 * (k / max(1, len(vps)-1)))
            for i in np.where(lab == k)[0]:
                x1, y1, x2, y2 = segs[i].astype(int)
                cv.line(lab_vis, (x1, y1), (x2, y2), int(col), 1)
        imwrite_ok(os.path.join("outputs","labels_debug.png"), lab_vis, "labels_debug")

    vpx = vps[fx_idx] if fx_idx is not None else None
    vpy = vps[fy_idx] if fy_idx is not None else None
    vpz = vps[vz_idx] if vz_idx is not None else None

    # draw-set
    if draw_wall_n is None:
        draw_wall_n = draw_n

    def top_ids_for_label(lbl, n):
        ids = np.where(lab == lbl)[0]
        if len(ids) == 0: return []
        order = ids[np.argsort(-lens[ids])]
        return list(order[:min(n, len(order))])

    draw_ids = []
    if fx_idx is not None:
        draw_ids += top_ids_for_label(fx_idx, draw_n)
    if fy_idx is not None:
        draw_ids += top_ids_for_label(fy_idx, draw_n)
    if vz_idx is not None:
        draw_ids += top_ids_for_label(vz_idx, draw_wall_n)

    distmap = build_boundary_distance(floor_mask, walls_mask, bgr.shape, dilate_px=boundary_dilate_px)
    if max_boundary_px is None: max_boundary_px = max_boundary_frac * min(h, w)

    lines_for_overlay = []
    for i in draw_ids:
        a,b,c = lines[i]; pts=[]
        for x in (0, w-1):
            y = (-c - a*x)/(b + 1e-12)
            if 0 <= y < h: pts.append((int(x), int(round(y))))
        for y in (0, h-1):
            x = (-c - b*y)/(a + 1e-12)
            if 0 <= x < w: pts.append((int(round(x)), int(y)))
        if len(pts) < 2: continue
        p1,p2 = pts[0], pts[1]
        xs = np.clip(np.linspace(p1[0], p2[0], 220).astype(np.int32), 0, distmap.shape[1]-1)
        ys = np.clip(np.linspace(p1[1], p2[1], 220).astype(np.int32), 0, distmap.shape[0]-1)
        if float(distmap[ys, xs].max()) > max_boundary_px: continue
        role = 'wall' if (vpz is not None and lab[i]==vz_idx) else 'floor'
        lines_for_overlay.append({'p1': p1, 'p2': p2, 'role': role})

    if not lines_for_overlay:  # permissive fallback draw
        for i in draw_ids:
            a,b,c = lines[i]; pts=[]
            for x in (0, w-1):
                y = (-c - a*x)/(b + 1e-12)
                if 0 <= y < h: pts.append((int(x), int(round(y))))
            for y in (0, h-1):
                x = (-c - b*y)/(a + 1e-12)
                if 0 <= x < w: pts.append((int(round(x)), int(y)))
            if len(pts) < 2: continue
            p1,p2 = pts[0], pts[1]
            role = 'wall' if (vpz is not None and lab[i]==vz_idx) else 'floor'
            lines_for_overlay.append({'p1': p1, 'p2': p2, 'role': role})

    return {'x': vpx, 'y': vpy, 'z': vpz}, lines_for_overlay

def _pad_to_include_points(img, pts, margin=40):
    h, w = img.shape[:2]; xs, ys = [], []
    for p in pts:
        if p is None: continue
        q = p/(p[2]+1e-12)
        if not np.isfinite(q).all(): continue
        xs.append(q[0]); ys.append(q[1])
    if not xs: return img, (0,0)
    minx,maxx,miny,maxy = min(xs),max(xs),min(ys),max(ys)
    left  = max(0, int(math.floor(-min(0, minx))) + margin)
    right = max(0, int(math.floor(max(0, maxx-(w-1)))) + margin)
    top   = max(0, int(math.floor(-min(0, miny))) + margin)
    bottom= max(0, int(math.floor(max(0, maxy-(h-1)))) + margin)
    if left==right==top==bottom==0: return img,(0,0)
    return cv.copyMakeBorder(img, top,bottom,left,right, cv.BORDER_CONSTANT, value=(0,0,0)), (left,top)

def draw_overlay(img, empty_floor, lines_for_overlay, vps):
    vis = img.copy(); tint = vis.copy()
    tint[empty_floor > 0] = (tint[empty_floor > 0] * 0.7 + np.array((140,255,140))[None,None,:]*0.3).astype(np.uint8)
    vis = tint
    vis, shift = _pad_to_include_points(vis, [vps.get('x'), vps.get('y'), vps.get('z')], margin=40)
    sx, sy = shift
    FLOOR_COL, WALL_COL = (60,210,60), (0,210,255)
    for L in lines_for_overlay:
        col = FLOOR_COL if L['role']=='floor' else WALL_COL
        p1 = (L['p1'][0]+sx, L['p1'][1]+sy); p2 = (L['p2'][0]+sx, L['p2'][1]+sy)
        cv.line(vis, p1, p2, col, 2, cv.LINE_AA)
    def draw_vp(vp, col, tag):
        if vp is None: return
        p = vp/(vp[2]+1e-12)
        if np.isfinite(p).all():
            q = (int(round(p[0]+sx)), int(round(p[1]+sy)))
            cv.circle(vis, q, 7, (0,0,0), -1, cv.LINE_AA)
            cv.circle(vis, q, 7, col, 2, cv.LINE_AA)
            cv.putText(vis, tag, (q[0]+10, q[1]-8), cv.FONT_HERSHEY_SIMPLEX, 0.6, col, 2, cv.LINE_AA)
    draw_vp(vps.get('x'), (0,255,0), "X")
    draw_vp(vps.get('y'), (255,0,0), "Y")
    draw_vp(vps.get('z'), (0,255,255), "Z")
    return vis

# ============================ main ============================

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--image", required=True)
    ap.add_argument("--out_dir", default="outputs")
    ap.add_argument("--draw_n", type=int, default=10)
    ap.add_argument("--draw_wall_n", type=int, default=None)
    ap.add_argument("--max_keep", type=int, default=600)
    ap.add_argument("--min_len_frac", type=float, default=0.05)
    ap.add_argument("--score_bias", type=float, default=0.35)
    ap.add_argument("--axis_valid_frac", type=float, default=0.60)
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    img = cv.imread(args.image, cv.IMREAD_COLOR)
    if img is None:
        raise SystemExit(f"Cannot read image: {args.image}")

    # Load masks produced by planes.py
    floor_clean_p = os.path.join(args.out_dir, "floor_clean.png")
    wall_clean_p  = os.path.join(args.out_dir, "wall_clean.png")
    empty_floor_p = os.path.join(args.out_dir, "empty_floor.png")

    floor_m = cv.imread(floor_clean_p, cv.IMREAD_GRAYSCALE)
    wall_m  = cv.imread(wall_clean_p,  cv.IMREAD_GRAYSCALE)
    empty_floor = cv.imread(empty_floor_p, cv.IMREAD_GRAYSCALE)

    # --- make sure masks match image size (no logic change)
    h, w = img.shape[:2]
    def as_gray_and_resize(m):
        if m is None: return None
        if m.ndim == 3: m = cv.cvtColor(m, cv.COLOR_BGR2GRAY)
        if m.shape[:2] != (h, w):
            m = cv.resize(m, (w, h), interpolation=cv.INTER_NEAREST)
        return m

    floor_m = as_gray_and_resize(floor_m)
    wall_m  = as_gray_and_resize(wall_m)

    # still build empty room for computation
    empty_floor = make_empty_room_floor(floor_m, wall_m, feet_mask=None)
    imwrite_ok(os.path.join(args.out_dir, "empty_floor.png"), empty_floor, "empty_floor")

    # VPs + lines (use empty_floor for geometry)
    vps, lines_for_overlay = compute_vps_and_lines(
        img, floor_mask=empty_floor, walls_mask=wall_m,
        draw_n=args.draw_n, draw_wall_n=args.draw_wall_n,
        max_keep=args.max_keep, min_len_frac=args.min_len_frac,
        score_bias=args.score_bias, max_boundary_px=None,
        max_boundary_frac=0.45, boundary_dilate_px=2,
        axis_valid_frac=args.axis_valid_frac
    )

    # Draw overlay with the original cleaned floor mask for visual parity
    vp_overlay = draw_overlay(img, floor_m, lines_for_overlay, vps)
    imwrite_ok(os.path.join(args.out_dir, "vp_overlay.png"), vp_overlay, "vp_overlay")

    if empty_floor is None:
        if floor_m is None:
            raise SystemExit("[ERROR] Need floor_clean.png or empty_floor.png in outputs/")
        empty_floor = make_empty_room_floor(floor_m, wall_m, feet_mask=None)
        imwrite_ok(os.path.join(args.out_dir, "empty_floor.png"), empty_floor, "empty_floor")

    # Compute VPs and draw overlay
    vps, lines_for_overlay = compute_vps_and_lines(
        img, floor_mask=empty_floor, walls_mask=wall_m,
        draw_n=args.draw_n, draw_wall_n=args.draw_wall_n,
        max_keep=args.max_keep, min_len_frac=args.min_len_frac,
        score_bias=args.score_bias, max_boundary_px=None,
        max_boundary_frac=0.45, boundary_dilate_px=2,
        axis_valid_frac=args.axis_valid_frac
    )
    print(f"[VP px] X={_vp_px(vps['x'])}  Y={_vp_px(vps['y'])}  Z={_vp_px(vps['z'])}")

    vp_overlay = draw_overlay(img, empty_floor, lines_for_overlay, vps)
    imwrite_ok(os.path.join(args.out_dir, "vp_overlay.png"), vp_overlay, "vp_overlay")

    # Save VPs for bev.py
    def as_list(v):
        return None if v is None else [float(v[0]), float(v[1]), float(v[2])]
    with open(os.path.join(args.out_dir, "vps.json"), "w") as f:
        json.dump({"x": as_list(vps['x']), "y": as_list(vps['y']), "z": as_list(vps['z'])}, f, indent=2)

    print("[DONE] VPs saved to", os.path.abspath(os.path.join(args.out_dir, "vps.json")))

if __name__ == "__main__":
    main()
