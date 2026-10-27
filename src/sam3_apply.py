"""SAM 3 rules on saved per-image maps (scripts/sam3_maps.py), CPU only. Per image, its own input only:
trim: t1_test_74's elasmo trim (elasmo where max SAM 3 map < tau -> nearest kept label; skipped if > guard of it goes).
grow c:t:d: bg/coral pixels within d px (d 0: anywhere; d < 0: anywhere, even with no class-c blob) of the ensemble's class-c blob where SAM 3's class-c map > t become c (hybrid:
SAM for animals, the ensemble for coral/bg). CV (scratchpad sam3_rules3.py): trim + grow 2:0.7:8 q_img 0.6480 -> 0.6484.

    .venv/bin/python scripts/sam3_apply.py --src work/t1_test_61_seg_masks \
        --out work/t1_sam3hyb_masks --grow 2:0.7:8
"""
import argparse
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent.parent


def nearest_fill(m, kill):
    _, lab = cv2.distanceTransformWithLabels(kill.astype(np.uint8), cv2.DIST_L2, 5, labelType=cv2.DIST_LABEL_PIXEL)
    keep = np.flatnonzero(~kill.ravel())
    lut = np.zeros(lab.max() + 1, m.dtype); lut[lab.ravel()[keep]] = m.ravel()[keep]
    out = m.copy(); out[kill] = lut[lab[kill]]
    return out


def trim(m, anything, tau, guard):
    kill = (m == 3) & (anything < tau)
    if not kill.any() or kill.sum() > guard * (m == 3).sum():
        return m
    return nearest_fill(m, kill)


def grow(m, sm, c, t, d):
    own = (m == c).astype(np.uint8)
    if not own.any() and d >= 0:
        return m
    near = cv2.dilate(own, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * d + 1, 2 * d + 1))) > 0 if d > 0 else True
    out = m.copy(); out[near & (m <= 1) & (sm > t)] = c  # d 0: anywhere, if the image has any of our class c; d < 0: always
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default="work/t1_test_61_seg_masks")
    ap.add_argument("--maps", default="work/sam3_test_maps")
    ap.add_argument("--out", required=True)
    ap.add_argument("--tau", type=float, default=0.2, help="trim threshold; <= 0 disables trim")
    ap.add_argument("--guard", type=float, default=0.5)
    ap.add_argument("--grow", nargs="*", default=[], help="c:t:d, applied in order after trim")
    a = ap.parse_args()
    src, maps, out = ROOT / a.src, ROOT / a.maps, ROOT / a.out
    out.mkdir(parents=True, exist_ok=True)
    grows = [(int(c), float(t), int(d)) for c, t, d in (g.split(":") for g in a.grow)]
    files = sorted(src.glob("*.png"))
    changed = px = tot = 0
    for p in files:
        m0 = cv2.imread(str(p), cv2.IMREAD_UNCHANGED)
        z = np.load(maps / f"{p.stem}.npz")
        sm = {c: z[f"s{c}"].astype(np.float32) / 255 for c in (2, 3, 4)}
        assert sm[2].shape == m0.shape, (p.name, sm[2].shape, m0.shape)
        m = trim(m0, np.maximum.reduce(list(sm.values())), a.tau, a.guard) if a.tau > 0 else m0
        for c, t, d in grows:
            m = grow(m, sm[c], c, t, d)
        diff = int((m != m0).sum())
        changed += diff > 0; px += diff; tot += m.size
        cv2.imwrite(str(out / p.name), m)
    print(f"{len(files)} masks, {changed} changed, {100 * px / tot:.3f}% px differ from {a.src} -> {out}", flush=True)


if __name__ == "__main__":
    main()
