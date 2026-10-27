"""Shared bits for EURS restoration: data, metrics, models."""
import math
from functools import partial
import json
import os
import random
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset

cv2.setNumThreads(1)
ROOT = Path(__file__).resolve().parents[1]
CACHE = Path(os.environ.get("EURS_CACHE", ROOT / "cache"))  # cache_pil = organizer resize filter
TYPES = range(1, 11)
LEVELS = range(5, 101, 5)

# Scene-grouped folds; rare-class scenes (ray1, sharkFish, seahorse1, starfish2, turtles1) spread out.
FOLDS = [
    ["ray1", "seahorse1", "fish1", "fish11"],
    ["sharkFish", "starfish2", "fish12", "fish13"],
    ["turtles1", "fish14", "fish15", "fish17"],
    ["fish18", "fish2", "fish20", "fish21"],
    ["fish4", "fish5", "fish7", "fish9"],
]
ALL_SCENES = sorted(s for f in FOLDS for s in f)


def imread(path):
    """uint8 HxWx3 RGB."""
    return cv2.imread(str(path), cv2.IMREAD_COLOR_RGB)


def split_items(split="dev"):
    """(output stem, cached frame) of an unlabelled split cache/<split>/<scene>/deg/<x>.png. dev stems are
    <scene>_<x>; splits from prepare_cache.py --split carry src.json (key <scene>/<x> -> original file)."""
    src = CACHE / split / "src.json"
    names = json.loads(src.read_text()) if src.exists() else {}
    for sd in sorted(p for p in (CACHE / split).iterdir() if p.is_dir()):
        for p in sorted((sd / "deg").glob("*.png")):
            k = f"{sd.name}/{p.stem}"
            yield (Path(names[k]).stem if names else f"{sd.name}_{p.stem}"), p


def split_sources(split="dev"):
    """Original downloaded files of a split (the organizer resize script reads their sizes)."""
    src = CACHE / split / "src.json"
    if src.exists():
        return [Path(v) for v in json.loads(src.read_text()).values()]
    return sorted((ROOT / "RSUIGM for EURS/eurs_dev_degraded/dev/degraded").glob("*/*.png"))


def frame_meta(stem):
    """(type, level) from a name ending in _<type>_<level>, else None (the restorer then infers it)."""
    try:
        t, l = map(int, stem.split("_")[-2:])
    except ValueError:
        return None
    return (t, l) if t in TYPES and l in LEVELS else None


def to_tensor(img):
    return torch.from_numpy(np.ascontiguousarray(img)).permute(2, 0, 1).float() / 255


def deg_path(split, scene, t, l):
    return CACHE / split / scene / "deg" / f"{t}_{l}.png"


def lab_ab(im):
    """Mean Lab a*, b* of a uint8 RGB image."""
    return cv2.cvtColor(im.astype(np.float32) / 255, cv2.COLOR_RGB2LAB)[..., 1:].reshape(-1, 2).mean(0)


class CropPairs(Dataset):
    """Random (degradation, crop) pairs from the given scenes. Also returns a thumbnail of the
    whole degraded frame so models can see global colour statistics while training on crops."""

    def __init__(self, scenes, crop=384, length=4000, thumb=128, ext=0.0, hard=1.0, ext_match=False, t2=0, grids=False,
                 kd=0.0):
        """t2 = S > 0: Track 2 scale. The frame is squashed to 384x768 on the Jetson (sx = 768/W, sy = 384/H), so the
        input is a GT-aligned region of (S/sy, S/sx) area-resized to SxS, and the target y is that region at full
        scale (variable size: use collate_t2).
        grids: also return the crop's first pass from out-of-fold GridNet grids (load_grids); external photos get
        the cell's per-channel table affine instead (what the grid starts from).
        kd: share of samples from dev frames paired with their own Track 1 restorer output (scripts/kd_targets.py);
        Track 2 only, see DATA_SOURCES.md."""
        assert not (grids and t2) and not (grids and kd)
        self.scenes, self.crop, self.length, self.thumb, self.ext, self.t2 = scenes, crop, length, thumb, ext, t2
        self.kd, self.kd_scenes = kd, sorted(d.name for d in (CACHE / "dev").iterdir() if (d / "kd").is_dir()) if kd else []
        assert not kd or self.kd_scenes, "no cache/dev/*/kd: run scripts/kd_targets.py"
        self.tw = [hard if t >= 7 else 1.0 for t in TYPES]  # coastal 3C/5C/7C/9C are the weakest types held out
        self._gt, self._grids = {}, {}
        self.grids = fit_table(scenes).numpy() if grids else None
        if ext:
            self.fwd = fit_poly_table(scenes, forward=True).numpy()  # only training scenes
            self.ext_files = sorted((CACHE / "ext").glob("*.png"))
            # LSUI raw photos are green (a* -18, b* -1); EURS GTs are blue (a* +0.5, b* -9.5) (scripts/color_stats.py)
            self.gt_ab = [lab_ab(self.gt(s)) for s in scenes] if ext_match else None

    def __len__(self):
        return self.length

    def gt(self, s):
        if s not in self._gt:
            self._gt[s] = imread(CACHE / "train" / s / "gt.png")
        return self._gt[s]

    def emulate(self, t, l):
        """External raw photo as clean target, degraded with the (type, level) forward cubic colour map fitted on
        training scenes. Global pointwise map only; depth-driven haze did not fit EURS better (fit_haze)."""
        y = imread(random.choice(self.ext_files))
        if self.gt_ab:  # move the photo's mean a*/b* (colour cast) onto a random training GT's; L untouched
            lab = cv2.cvtColor(y.astype(np.float32) / 255, cv2.COLOR_RGB2LAB)
            lab[..., 1:] += np.array(random.choice(self.gt_ab), np.float32) - lab[..., 1:].reshape(-1, 2).mean(0)
            y = (np.clip(cv2.cvtColor(lab, cv2.COLOR_LAB2RGB), 0, 1) * 255).round().astype(np.uint8)
        h, w = y.shape[:2]
        if min(h, w) < self.crop:
            s = self.crop / min(h, w)
            y = cv2.resize(y, (max(self.crop, round(w * s)), max(self.crop, round(h * s))), interpolation=cv2.INTER_CUBIC)
        f = y.reshape(-1, 3).astype(np.float32) / 255
        x = np.stack(poly_terms(f[:, 0], f[:, 1], f[:, 2]), 1) @ self.fwd[cell(t, l)]
        x = np.clip(x * 255, 0, 255).round().astype(np.uint8).reshape(y.shape)
        return x, y

    def __getitem__(self, _):
        t, l = random.choices(TYPES, self.tw)[0], random.choice(LEVELS)
        s, r = None, random.random()
        if r < self.ext:
            x, y = self.emulate(t, l)
        elif r < self.ext + self.kd:
            s = random.choice(self.kd_scenes)
            x, y = imread(deg_path("dev", s, t, l)), imread(CACHE / "dev" / s / "kd" / f"{t}_{l}.png")
        else:
            s = random.choice(self.scenes)
            x, y = imread(deg_path("train", s, t, l)), self.gt(s)
        th = cv2.resize(x, (self.thumb, self.thumb), interpolation=cv2.INTER_AREA)
        st = frame_stats(to_tensor(x)[None])[0]
        h, w = y.shape[:2]
        if self.t2:
            # emulated photos borrow a random training frame's squash; organizer resize filter unknown (area)
            fh, fw = (h, w) if s is not None else self.gt(random.choice(self.scenes)).shape[:2]
            S = self.t2
            ch, cw = min(h, round(S * fh / 384)), min(w, round(S * fw / 768))
            i, j = random.randint(0, h - ch), random.randint(0, w - cw)
            x, y = x[i:i + ch, j:j + cw], y[i:i + ch, j:j + cw]
            x = cv2.resize(x, (S, S), interpolation=cv2.INTER_AREA)
        else:
            c = self.crop
            i, j = random.randint(0, h - c), random.randint(0, w - c)
            x, y = x[i:i + c, j:j + c], y[i:i + c, j:j + c]
        out = [x, y, th]
        if self.grids is not None:
            if s is None:
                ab = self.grids[cell(t, l)]
                out.append(x.astype(np.float32) / 255 * ab[:, 0] + ab[:, 1])
            else:
                if s not in self._grids:
                    self._grids[s] = load_grids("train", s)
                out.append(grid_crop(x, self._grids[s][cell(t, l)], (h, w), i, j))
        if random.random() < 0.5:
            out = [v[:, ::-1] for v in out]
        x, y, th, *g = out
        g = [to_tensor(v * 255) for v in g]  # float first pass, already in [0, 1]-ish
        return to_tensor(x), to_tensor(y), to_tensor(th), st, cell(t, l), *g


def collate_t2(batch):
    """CropPairs(t2=S) batches: targets differ in size, so y stays a list."""
    x, y, th, st, c = zip(*batch)
    return torch.stack(x), list(y), torch.stack(th), torch.stack(st), torch.tensor(c)


def cell(t, l):
    """(type 1..10, level 5..100) -> class index 0..199."""
    return (t - 1) * 20 + l // 5 - 1


def frame_stats(x):
    """Per-channel mean, std, max of the whole frame (B,3,H,W) -> (B,9).
    Biased std as a mean of squares: torch.std's N/(N-1) exports N=H*W as a constant, which overflows FP16
    (294912 > 65504) and made TensorRT FP16 output all-NaN. Differs from unbiased std by ~1e-6."""
    mu = x.mean((2, 3), keepdim=True)
    return torch.cat([mu.flatten(1), (x - mu).pow(2).mean((2, 3)).sqrt(), x.amax((2, 3))], 1)


def fit_table(scenes):
    """Least-squares per-channel affine gt ~ a*deg + b for each of the 200 (type, level) cells,
    pooled over `scenes` at 1/4 eval scale. Returns (200, 3, 2) float tensor; cached on disk."""
    key = CACHE / f"table_{'_'.join(sorted(scenes)) if len(scenes) < 20 else 'all'}.pt"
    if key.exists():
        return torch.load(key)
    small = lambda p: cv2.resize(im := imread(p), (im.shape[1] // 4, im.shape[0] // 4),
                                 interpolation=cv2.INTER_AREA).reshape(-1, 3) / 255.0
    gts = {s: small(CACHE / "train" / s / "gt.png") for s in scenes}
    tab = torch.zeros(200, 3, 2)
    for t in TYPES:
        for l in LEVELS:
            xs = np.concatenate([small(deg_path("train", s, t, l)) for s in scenes])
            ys = np.concatenate([gts[s] for s in scenes])
            for c in range(3):
                x, y = xs[:, c], ys[:, c]
                a = min(max(((x - x.mean()) * (y - y.mean())).mean() / (x.var() + 1e-8), 0), 64)
                tab[cell(t, l), c] = torch.tensor([a, y.mean() - a * x.mean()])
    torch.save(tab, key)
    return tab


# cubic cross-channel colour map: 20 monomials r^i g^j b^k, i+j+k <= 3
POLY_EXP = sorted({(i, j, k) for i in range(4) for j in range(4) for k in range(4) if i + j + k <= 3})


def poly_terms(r, g, b):
    """20 monomials of the three channels (torch or numpy, any matching shapes). Mul-only (ONNX/TRT friendly)."""
    one = r * 0 + 1
    pw = [(one, c, c * c, c * c * c) for c in (r, g, b)]
    return [pw[0][i] * pw[1][j] * pw[2][k] for i, j, k in POLY_EXP]


def poly_identity():
    w = torch.zeros(20, 3)
    for c, e in enumerate([(1, 0, 0), (0, 1, 0), (0, 0, 1)]):
        w[POLY_EXP.index(e), c] = 1
    return w


def fit_poly_table(scenes, forward=False):
    """Per (type, level) cell ridge fit of a cubic colour map, pooled over `scenes` at 256x160:
    inverse gt ~ poly(deg) (physics branch) or forward deg ~ poly(gt) (ext emulator). (200, 20, 3); cached.
    Known-cell ceiling on fold-0 val: 27.8 dB vs 26.0 for the per-channel affine (scripts/inverse_bounds.py)."""
    key = CACHE / f"poly_{'fwd' if forward else 'inv'}_{'_'.join(sorted(scenes)) if len(scenes) < 20 else 'all'}.pt"
    if key.exists():
        return torch.load(key)
    sm = lambda p: cv2.resize(imread(p), (256, 160), interpolation=cv2.INTER_AREA).reshape(-1, 3) / 255.0
    feats = lambda x: np.stack(poly_terms(x[:, 0], x[:, 1], x[:, 2]), 1)
    gts = np.concatenate([sm(CACHE / "train" / s / "gt.png") for s in scenes])
    tab = torch.zeros(200, 20, 3)
    for t in TYPES:
        for l in LEVELS:
            deg = np.concatenate([sm(deg_path("train", s, t, l)) for s in scenes])
            x, y = (gts, deg) if forward else (deg, gts)
            P = feats(x)
            tab[cell(t, l)] = torch.from_numpy(np.linalg.solve(P.T @ P + 1e-4 * np.eye(20), P.T @ y)).float()
    torch.save(tab, key)
    return tab


def load_model(name, ckpt):
    """Build MODELS[name] with the table buffer stored in the checkpoint (affine or poly), then load weights."""
    sd = torch.load(ckpt, map_location="cpu")
    model = MODELS[name](sd.get("table", sd.get("phys.table")))
    model.load_state_dict(sd)
    return model


def val_items(scenes, levels=(5, 25, 50, 75, 100)):
    return [(s, t, l) for s in scenes for t in TYPES for l in levels]


def thumbnail(x, size=128):
    return F.interpolate(x, size=(size, size), mode="area")


def apply_grid(x, coef):
    """x (B,3,H,W), coef (B,12,g,g): out_c = sum_k M[c,k] x_k + M[c,3], coefficients bilinearly upsampled to H,W."""
    m = F.interpolate(coef, size=x.shape[-2:], mode="bilinear", align_corners=False).view(x.shape[0], 3, 4, *x.shape[-2:])
    return (m[:, :, :3] * x[:, None]).sum(2) + m[:, :, 3]


def load_grids(split, scene):
    """(200, 12, g, g) float16 out-of-fold GridNet coefficients in cell order (restoration/gridnet.py --export)."""
    return np.load(CACHE / "grids" / split / f"{scene}.npy")


def _interp_rows(n, g, start, length):
    """(length, g) weights reproducing F.interpolate(bilinear, align_corners=False) from g to n, rows start.."""
    src = np.maximum((np.arange(start, start + length) + 0.5) * g / n - 0.5, 0)
    i0 = np.minimum(np.floor(src).astype(int), g - 1)
    i1, f = np.minimum(i0 + 1, g - 1), src - i0
    w = np.zeros((length, g), np.float32)
    np.add.at(w, (np.arange(length), i0), 1 - f)
    np.add.at(w, (np.arange(length), i1), f)
    return w


def grid_crop(x, coef, hw, i, j):
    """apply_grid on a crop: x (c,c,3) uint8 at rows i.., cols j.. of an (h, w) frame; coef (12, g, g)."""
    c, g = x.shape[0], coef.shape[-1]
    m = _interp_rows(hw[0], g, i, c) @ coef.astype(np.float32) @ _interp_rows(hw[1], g, j, x.shape[1]).T  # (12, c, w)
    m = m.transpose(1, 2, 0).reshape(c, x.shape[1], 3, 4)
    return (m[..., :3] * (x.astype(np.float32) / 255)[:, :, None]).sum(-1) + m[..., 3]


# ---------------------------------------------------------------- metrics

def _gauss(win=11, sigma=1.5, device="cpu"):
    g = torch.exp(-((torch.arange(win, device=device) - win // 2) ** 2) / (2 * sigma ** 2))
    g = g / g.sum()
    return (g[:, None] @ g[None, :])[None, None]


def ssim(x, y):
    """Mean SSIM over RGB channels, standard 11x11 gaussian window, inputs in [0,1]."""
    c = x.shape[1]
    w = _gauss(device=x.device).expand(c, 1, 11, 11)
    mu_x, mu_y = F.conv2d(x, w, groups=c), F.conv2d(y, w, groups=c)
    sxx = F.conv2d(x * x, w, groups=c) - mu_x ** 2
    syy = F.conv2d(y * y, w, groups=c) - mu_y ** 2
    sxy = F.conv2d(x * y, w, groups=c) - mu_x * mu_y
    c1, c2 = 0.01 ** 2, 0.03 ** 2
    m = ((2 * mu_x * mu_y + c1) * (2 * sxy + c2)) / ((mu_x ** 2 + mu_y ** 2 + c1) * (sxx + syy + c2))
    return m.mean(dim=(1, 2, 3))


def psnr(x, y):
    return -10 * torch.log10(((x - y) ** 2).mean(dim=(1, 2, 3)).clamp_min(1e-10))


_LAP = torch.tensor([[0., 1, 0], [1, -4, 1], [0, 1, 0]])[None, None]


def sharpness(x):
    g = (0.299 * x[:, 0] + 0.587 * x[:, 1] + 0.114 * x[:, 2])[:, None]
    return F.conv2d(g, _LAP.to(x.device)).var(dim=(1, 2, 3))


def restoration_score(p, s, sh):
    """Our proxy for the 0.3-weighted restoration group (sharpness ratio capped at 1 — a guess)."""
    return 0.45 * min(max((p - 15) / 20, 0), 1) + 0.45 * s + 0.10 * min(sh, 1)


# ---------------------------------------------------------------- building blocks

class LayerNorm2d(nn.Module):
    """Channel LayerNorm. Exports as ONNX LayerNormalization (opset 17), which TensorRT runs as one
    normalization layer with FP32 accumulation; the old mean/pow/sqrt graph risked FP16 overflow."""

    def __init__(self, c, eps=1e-6):
        super().__init__()
        self.weight, self.bias, self.eps = nn.Parameter(torch.ones(c)), nn.Parameter(torch.zeros(c)), eps

    def forward(self, x):
        x = F.layer_norm(x.permute(0, 2, 3, 1), (x.shape[1],), self.weight, self.bias, self.eps)
        return x.permute(0, 3, 1, 2)


class NAFBlock(nn.Module):
    def __init__(self, c, film=0):
        super().__init__()
        self.n1, self.n2 = LayerNorm2d(c), LayerNorm2d(c)
        self.c1 = nn.Conv2d(c, 2 * c, 1)
        self.dw = nn.Conv2d(2 * c, 2 * c, 3, padding=1, groups=2 * c)
        self.sca = nn.Conv2d(c, c, 1)
        self.c3 = nn.Conv2d(c, c, 1)
        self.c4 = nn.Conv2d(c, 2 * c, 1)
        self.c5 = nn.Conv2d(c, c, 1)
        self.beta = nn.Parameter(torch.zeros(1, c, 1, 1))
        self.gamma = nn.Parameter(torch.zeros(1, c, 1, 1))
        self.film = nn.Linear(film, 2 * c) if film else None
        self.c = c

    def forward(self, x, z=None):
        c = self.c  # python-int slices, not chunk(): chunk exports runtime-index Slices that break ONNX shape inference
        y = self.n1(x)
        if self.film is not None:
            f = self.film(z)[..., None, None]
            y = y * (1 + f[:, :c]) + f[:, c:]
        y = self.dw(self.c1(y))
        y = y[:, :c] * y[:, c:]
        y = y * self.sca(y.mean((2, 3), keepdim=True))
        x = x + self.c3(y) * self.beta
        y = self.c4(self.n2(x))
        return x + self.c5(y[:, :c] * y[:, c:]) * self.gamma


class GlobalHead(nn.Module):
    """Thumbnail (B,3,128,128) -> global descriptor z. Sees the whole frame even when training on crops."""

    def __init__(self, dim=64, width=32):
        super().__init__()
        ch = [3, width // 2, width, 2 * width, 2 * width]
        self.convs = nn.Sequential(*[m for i in range(4) for m in
                                     (nn.Conv2d(ch[i], ch[i + 1], 3, 2, 1), nn.GELU())])
        self.mlp = nn.Sequential(nn.Linear(2 * ch[-1] + 9, 2 * dim), nn.GELU(), nn.Linear(2 * dim, dim))

    def forward(self, th, st):
        f = self.convs(th)
        return self.mlp(torch.cat([f.mean((2, 3)), f.amax((2, 3)), st], 1))


class LKA(nn.Module):
    """Large-kernel attention (VAN / NemoNet): dw5x5 -> dw7x7 dil3 -> 1x1, used as a multiplicative gate."""

    def __init__(self, c):
        super().__init__()
        self.dw = nn.Conv2d(c, c, 5, padding=2, groups=c)
        self.dwd = nn.Conv2d(c, c, 7, padding=9, groups=c, dilation=3)
        self.pw = nn.Conv2d(c, c, 1)

    def forward(self, x):
        return x * self.pw(self.dwd(self.dw(x)))


# ---------------------------------------------------------------- models

class PhysNet(nn.Module):
    """Track-2-sized restorer.
    1. global head (thumbnail + frame stats) -> posterior over the 200 (type, level) cells
    2. posterior-weighted colour map from a per-cell table fitted on training scenes, applied at full res:
       (200, 3, 2) per-channel affine (old checkpoints) or (200, 20, 3) cubic cross-channel map (~60 MAC/px)
    3. 1/8-res gain/offset map (LKA) for depth-varying veil, upsampled and applied elementwise
    4. small 1/2-res residual refiner."""

    def __init__(self, table=None, zdim=64, gw=32, vc=32, rc=16, down=8):
        super().__init__()
        self.down = down
        self.register_buffer("table", table.clone() if table is not None else poly_identity().repeat(200, 1, 1))
        self.g = GlobalHead(zdim, gw)
        self.cls = nn.Linear(zdim, 200)
        self.v_in = nn.Conv2d(6, vc, 3, padding=1)
        self.v_film = nn.Linear(zdim, 2 * vc)
        self.v_body = nn.Sequential(nn.GELU(), LKA(vc), nn.Conv2d(vc, vc, 3, padding=1), nn.GELU(),
                                    LKA(vc), nn.Conv2d(vc, 6, 1))
        self.r = nn.Sequential(nn.Conv2d(24, rc, 3, padding=1), nn.GELU(),
                               nn.Conv2d(rc, rc, 3, padding=1), nn.GELU(),
                               nn.Conv2d(rc, rc, 3, padding=1), nn.GELU(),
                               nn.Conv2d(rc, 12, 3, padding=1))
        for m in (self.v_body[-1], self.r[-1]):
            nn.init.zeros_(m.weight), nn.init.zeros_(m.bias)

    def physics(self, x, th, st, cell=None):
        """cell: optional true (type, level) class per image (B,), from the filename ("meta"): uses that table entry
        instead of the classifier's soft mix."""
        z = self.g(th, st)
        logits = self.cls(z)
        # static-shape ops only (fixed slices, constant-scale Resize, no runtime reshape): keeps ONNX shape
        # inference and the organizer FLOP count exact. H, W must be multiples of `down` (crops 384, pad 16, 384x768).
        with torch.autocast(x.device.type, enabled=False):  # dark cells amplify x50: keep the colour map fp32
            p = logits.float().softmax(1) if cell is None else F.one_hot(cell, logits.shape[1]).float()
            x, w = x.float(), p @ self.table.flatten(1)
            if self.table.shape[-1] == 2:  # per-channel affine
                ab = w.view(-1, 3, 2)
                jt = x * ab[..., 0, None, None] + ab[..., 1, None, None]
            else:  # cubic map as 20 elementwise multiply-adds (~60 MAC/px)
                w = w.view(-1, 20, 3)[..., None, None]
                jt = sum(f[:, None] * w[:, k] for k, f in enumerate(poly_terms(x[:, 0], x[:, 1], x[:, 2])))
        v = self.v_in(F.avg_pool2d(torch.cat([x, jt.clamp(0, 1)], 1), self.down))
        f = self.v_film(z)[..., None, None]
        vc = self.v_in.out_channels
        v = self.v_body(v * (1 + f[:, :vc]) + f[:, vc:])
        v = F.interpolate(v, scale_factor=self.down, mode="bilinear", align_corners=False)
        return jt * torch.exp(v[:, :3].clamp(-4, 4)) + v[:, 3:], z, logits

    def forward(self, x, th=None, st=None, cell=None):
        th = thumbnail(x) if th is None else th
        st = frame_stats(x) if st is None else st
        j0, _, logits = self.physics(x, th, st, cell)
        r = F.pixel_shuffle(self.r(F.pixel_unshuffle(torch.cat([x, j0.clamp(0, 1)], 1), 2)), 2)
        return (j0 + r).clamp(0, 1), j0, logits


class BigNet(nn.Module):
    """Track-1 restorer: PhysNet front end + a NAFNet U-Net residual refiner conditioned on the global descriptor."""

    def __init__(self, table=None, width=32, enc=(1, 1, 2, 4), mid=6, dec=(1, 1, 1, 1), gw=64, half=False, grow=2.0):
        """half: U-Net runs at 1/2 res via pixel (un)shuffle (~4x fewer FLOPs); grow: channel factor per level
        (< 2 keeps deep levels cheap in params; Track-2-sized variants). grow=2 is the original layout."""
        super().__init__()
        self.phys = PhysNet(table, gw=gw)
        self.half = half
        s = 4 if half else 1
        z = 64
        self.intro = nn.Conv2d(6 * s, width, 3, padding=1)
        self.encs, self.downs, self.ups, self.decs = (nn.ModuleList() for _ in range(4))
        cs = [width]
        for n in enc:
            self.encs.append(nn.ModuleList([NAFBlock(cs[-1], z) for _ in range(n)]))
            cs.append(round(cs[-1] * grow))
            self.downs.append(nn.Conv2d(cs[-2], cs[-1], 2, 2))
        self.mid = nn.ModuleList([NAFBlock(cs[-1], z) for _ in range(mid)])
        for n, (c_in, c) in zip(dec, zip(cs[:0:-1], cs[-2::-1])):
            self.ups.append(nn.Sequential(nn.Conv2d(c_in, 4 * c, 1), nn.PixelShuffle(2)))
            self.decs.append(nn.ModuleList([NAFBlock(c, z) for _ in range(n)]))
        self.ending = nn.Conv2d(width, 3 * s, 3, padding=1)
        nn.init.zeros_(self.ending.weight), nn.init.zeros_(self.ending.bias)
        self.pad = 2 ** len(enc) * (2 if half else 1)

    def forward(self, x, th=None, st=None, cell=None):
        th = thumbnail(x) if th is None else th
        st = frame_stats(x) if st is None else st
        j0, z, logits = self.phys.physics(x, th, st, cell)
        H, W = x.shape[-2:]
        ph, pw = (-H) % self.pad, (-W) % self.pad
        inp = torch.cat([x, j0.clamp(0, 1)], 1)
        if ph or pw:  # skipped at export (384x768): a runtime-sized Pad breaks ONNX shape inference / FLOP count
            inp = F.pad(inp, (0, pw, 0, ph), mode="reflect")
        h = self.intro(F.pixel_unshuffle(inp, 2) if self.half else inp)
        skips = []
        for blocks, down in zip(self.encs, self.downs):
            for b in blocks:
                h = b(h, z)
            skips.append(h)
            h = down(h)
        for b in self.mid:
            h = b(h, z)
        for up, blocks, s in zip(self.ups, self.decs, reversed(skips)):
            h = up(h) + s
            for b in blocks:
                h = b(h, z)
        r = self.ending(h)
        r = F.pixel_shuffle(r, 2) if self.half else r
        out = j0 + (r[..., :H, :W] if ph or pw else r)
        return out.clamp(0, 1), j0, logits


class RestNet(nn.Module):
    """Track-1 restorer: PhysNet front end + official Restormer (MIT, external/restormer) refining [x, j0],
    initialised from its public real-denoising weights. Output = j0 + Restormer residual."""

    def __init__(self, table=None, pretrained=ROOT / "external/restormer/real_denoising.pth"):
        super().__init__()
        import sys
        sys.path.insert(0, str(ROOT / "external/restormer"))
        from restormer_arch import Restormer
        self.phys = PhysNet(table)
        # 6 channels in AND out: Restormer adds its input to its output, so out[:, 3:] = residual + j0
        self.net = Restormer(inp_channels=6, out_channels=6, LayerNorm_type="BiasFree")
        if pretrained and Path(pretrained).exists():
            sd = torch.load(pretrained, map_location="cpu")["params"]
            w, o = sd["patch_embed.proj.weight"], sd["output.weight"]
            sd["patch_embed.proj.weight"] = torch.cat([torch.zeros_like(w), w], 1)  # x taps start at 0, j0 = pretrained
            sd["output.weight"] = torch.cat([torch.zeros_like(o), o], 0)
            self.net.load_state_dict(sd)
        # gradient checkpointing per transformer block: 23 GB -> fits 16 GB at 256^2 x4 (costs ~30% compute)
        from torch.utils.checkpoint import checkpoint
        for b in self.net.modules():
            if type(b).__name__ == "TransformerBlock":
                b.forward = partial(lambda f, x: checkpoint(f, x, use_reentrant=False)
                                    if torch.is_grad_enabled() else f(x), b.forward)

    def forward(self, x, th=None, st=None, cell=None, j0=None):
        """j0: optional first pass from outside (whole-frame GridNet colour grid) replacing the PhysNet one."""
        th = thumbnail(x) if th is None else th
        st = frame_stats(x) if st is None else st
        if j0 is None:
            j0, _, logits = self.phys.physics(x, th, st, cell)
        else:
            logits = self.phys.cls(self.phys.g(th, st))
        out = self.net(torch.cat([x, j0.clamp(0, 1)], 1))[:, 3:]
        return out.clamp(0, 1), j0, logits


class PhaseNet(nn.Module):
    """RestNet with official PhaseFormer (Khan et al. WACV 2025, MIT, external/phaseformer; 1.77M params) as the
    refiner instead of Restormer. PhaseFormer predicts the image in [-1, 1] from its first (input-resolution) head;
    its 3-channel embedding is widened to [x, j0] with x taps zeroed, so at init it is the public UIEB model on j0."""

    def __init__(self, table=None, pretrained=ROOT / "external/phaseformer/checkpoints/UIEB/best.pth"):
        super().__init__()
        import sys
        sys.path.insert(0, str(ROOT / "external/phaseformer"))
        import model_with_eca
        from model_with_eca import Restormer as PhaseFormer
        if not hasattr(model_with_eca, "_inv_mag_fp32"):  # cuFFT has no bf16: phase-only FFT in fp32, rest stays bf16
            f = model_with_eca._inv_mag_fp32 = model_with_eca.inv_mag
            model_with_eca.inv_mag = lambda x: f(x.float()).to(x.dtype)
        self.phys = PhysNet(table)
        net = PhaseFormer()
        if pretrained and Path(pretrained).exists():
            net.load_state_dict(torch.load(pretrained, map_location="cpu", weights_only=False).state_dict())
        w = net.embed_conv_rgb.weight.data
        net.embed_conv_rgb = nn.Conv2d(6, w.shape[0], 3, padding=1, bias=False)
        net.embed_conv_rgb.weight.data = torch.cat([torch.zeros_like(w), w], 1)
        self.net = net
        from torch.utils.checkpoint import checkpoint  # complex FFT buffers: 16.4 GB at 256^2 x8 without it
        for b in self.net.modules():
            if type(b).__name__ == "TransformerBlock":
                b.forward = partial(lambda f, x: checkpoint(f, x, use_reentrant=False)
                                    if torch.is_grad_enabled() else f(x), b.forward)

    def forward(self, x, th=None, st=None):
        th = thumbnail(x) if th is None else th
        st = frame_stats(x) if st is None else st
        j0, _, logits = self.phys.physics(x, th, st)
        out = self.net(torch.cat([x, j0.clamp(0, 1)], 1) * 2 - 1)[0]
        return ((out + 1) / 2).clamp(0, 1), j0, logits


class XResNet(RestNet):
    """RestNet with official X-Restormer (Chen et al. ECCV 2024, MIT, external/xrestormer; 26.0M) as the refiner,
    initialised from its public SOTS dehazing weights. Its overlapping window attention runs on 1/8-res maps with
    8x8 windows, so inputs are reflect-padded to a multiple of 64."""

    def __init__(self, table=None, pretrained=ROOT / "external/xrestormer/checkpoints/dehaze_300k.pth"):
        nn.Module.__init__(self)
        import sys
        import types
        for m in ("basicsr", "basicsr.utils", "basicsr.utils.registry", "torchstat"):  # train-framework imports only
            sys.modules.setdefault(m, types.ModuleType(m))
        sys.modules["basicsr.utils.registry"].ARCH_REGISTRY = types.SimpleNamespace(register=lambda: lambda c: c)
        sys.modules["torchstat"].stat = None
        sys.path.insert(0, str(ROOT / "external/xrestormer/xrestormer/archs"))
        from xrestormer_arch import XRestormer
        self.phys = PhysNet(table)
        self.net = XRestormer(inp_channels=6, out_channels=6, num_blocks=[2, 4, 4, 4], spatial_heads=[1, 2, 4, 8])
        if pretrained and Path(pretrained).exists():
            sd = torch.load(pretrained, map_location="cpu")["params"]
            w, o = sd["patch_embed.proj.weight"], sd["output.weight"]
            sd["patch_embed.proj.weight"] = torch.cat([torch.zeros_like(w), w], 1)
            sd["output.weight"] = torch.cat([torch.zeros_like(o), o], 0)
            self.net.load_state_dict(sd)
        from torch.utils.checkpoint import checkpoint
        for b in self.net.modules():
            if type(b).__name__ == "TransformerBlock":
                b.forward = partial(lambda f, x: checkpoint(f, x, use_reentrant=False)
                                    if torch.is_grad_enabled() else f(x), b.forward)

    def forward(self, x, th=None, st=None):
        th = thumbnail(x) if th is None else th
        st = frame_stats(x) if st is None else st
        H, W = x.shape[-2:]
        y, j0, logits = super().forward(F.pad(x, (0, -W % 64, 0, -H % 64), mode="reflect"), th, st)
        return y[..., :H, :W], j0[..., :H, :W], logits


class SwinNet(RestNet):
    """RestNet with official SwinIR-M (Liang et al. ICCVW 2021, Apache-2.0, external/swinir; 11.8M) as the refiner,
    initialised from its public colour-denoising (sigma 15) weights. SwinIR pads to its 8x8 windows itself."""

    def __init__(self, table=None, pretrained=ROOT / "external/swinir/005_colorDN_DFWB_s128w8_SwinIR-M_noise15.pth"):
        nn.Module.__init__(self)
        import sys
        import types
        try:
            import timm  # noqa: F401
        except ImportError:  # SwinIR needs only three timm helpers; drop_path_rate=0 never builds DropPath
            for m in ("timm", "timm.models", "timm.models.layers"):
                sys.modules[m] = types.ModuleType(m)
            lay = sys.modules["timm.models.layers"]
            lay.DropPath, lay.to_2tuple, lay.trunc_normal_ = None, lambda v: (v, v), nn.init.trunc_normal_
        sys.path.insert(0, str(ROOT / "external/swinir"))
        from network_swinir import SwinIR
        self.phys = PhysNet(table)
        self.net = SwinIR(img_size=128, window_size=8, in_chans=6, embed_dim=180, depths=[6] * 6, num_heads=[6] * 6,
                          mlp_ratio=2, drop_path_rate=0., upsampler="", resi_connection="1conv")
        # x channels pass through unshifted; j0 gets the pretrained RGB mean, so at init out[:, 3:] = SwinIR(j0)
        self.net.mean = torch.tensor([0, 0, 0, 0.4488, 0.4371, 0.4040]).view(1, 6, 1, 1)
        if pretrained and Path(pretrained).exists():
            sd = torch.load(pretrained, map_location="cpu")
            sd = sd.get("params", sd)
            w, o, ob = sd["conv_first.weight"], sd["conv_last.weight"], sd["conv_last.bias"]
            sd["conv_first.weight"] = torch.cat([torch.zeros_like(w), w], 1)
            sd["conv_last.weight"] = torch.cat([torch.zeros_like(o), o], 0)
            sd["conv_last.bias"] = torch.cat([torch.zeros_like(ob), ob])
            self.net.load_state_dict(sd)
        # SwinIR's residual stream reaches ~1e4: a 3x3 conv over 180 channels on it overflows fp16 (NaN from step 1 on
        # the TITAN RTX), so the convs reading it run in fp32 (cheap: attention and MLPs dominate the compute)
        def fp32(f, x):
            with torch.autocast(x.device.type, enabled=False):
                return f(x.float())

        for conv in [layer.conv for layer in self.net.layers] + [self.net.conv_after_body, self.net.conv_last]:
            conv.forward = partial(fp32, conv.forward)
        # SwinIR rebuilds the shifted-window mask on the CPU in every block, every call, for any size other than
        # img_size (an all-zero one for unshifted blocks): cache it on the GPU per (shift, size)
        masks = {}

        def cached_mask(f, shift, size):
            if (shift, size) not in masks:
                masks[shift, size] = f(size).cuda()
            return masks[shift, size]

        from torch.utils.checkpoint import checkpoint
        for b in self.net.modules():
            if type(b).__name__ == "SwinTransformerBlock":
                b.calculate_mask = partial(cached_mask, b.calculate_mask, b.shift_size)
                b.forward = partial(lambda f, x, s: checkpoint(f, x, s, use_reentrant=False)
                                    if torch.is_grad_enabled() else f(x, s), b.forward)


class Identity(nn.Module):
    def __init__(self, *_):
        super().__init__()

    def forward(self, x, th=None, st=None):
        return x, x, None


MODELS = {"phys": PhysNet, "big": BigNet, "identity": Identity, "restormer": RestNet, "phaseformer": PhaseNet,
          "xrestormer": XResNet, "swinir": SwinNet,
          # Track-2-sized refiners (~6 GFLOPs at 384x768, see scripts/flops.py)
          "t2a": partial(BigNet, width=32, enc=(1, 2, 2), mid=2, dec=(1, 1, 1), gw=32, half=True, grow=1.25),  # 0.42M
          "t2b": partial(BigNet, width=16, enc=(1, 1, 1), mid=2, dec=(1, 1, 1), gw=32, grow=1.5),  # 0.27M
          # t2a slimmed to fit next to the "xl" segmentation student (whole model <= 1.0M)
          "t2c": partial(BigNet, width=24, enc=(1, 2, 2), mid=2, dec=(1, 1, 1), gw=32, half=True, grow=1.25)}


def count_params(m):
    return sum(p.numel() for p in m.parameters())
