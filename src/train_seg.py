"""DINOv3 ViT-L segmenter for EURS (Track 1 model, later the Track 2 distillation teacher).

    python seg/train_seg.py --fold 0 --name dino_f0                 # hold out SEG_FOLDS[0]
    python seg/train_seg.py --fold -1 --name dino_all --predict     # all 20 scenes, then dev masks -> work/dino_all_masks
    bash scripts/seg_cv.sh cv_ftp --bb_lr 2e-5 --paste 0.5           # 5-fold CV -> seg/cv_report.py (the LB proxy)

Backbone: timm vit_large_patch16_dinov3.lvd1689m (DINOv3 License), layers 6/12/18/24 -> 1/16 features, fused with a
small 1/4 conv detail branch. --bb_lr 0 keeps the backbone frozen; otherwise layer-wise decayed fine-tuning.
"""
import argparse
import json
import math
import os
import random
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint
from torch.utils.data import DataLoader, Dataset

ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("HF_HOME", str(ROOT / "external/hf"))  # before timm import
import timm  # noqa: E402

sys.path.insert(0, str(ROOT / "restoration"))
from eurs import (ALL_SCENES, CACHE, FOLDS, LEVELS, TYPES, cell, fit_poly_table, imread, lab_ab, poly_terms,  # noqa: E402
                  split_items)
from qseg import N_CLASSES, NAMES, confusion, from_confusion  # noqa: E402

cv2.setNumThreads(1)
RARE = ["ray1", "sharkFish", "seahorse1", "starfish2", "turtles1"]  # the only scenes with elasmo (3) / other (4)
_COMMON = [s for s in ALL_SCENES if s not in RARE]
SEG_FOLDS = [[r] + _COMMON[k::5] for k, r in enumerate(RARE)]  # each fold holds out exactly one rare scene
# 1/sqrt(pixel share) of the 20 training masks (bg .553, coral .337, fish .078, elasmo .0094, other .022), bg = 1
CLASS_W = torch.tensor([1.0, 1.28, 2.66, 7.67, 5.01])
AMP = torch.bfloat16  # --amp fp16 on GPUs without fast bf16 (Turing: TITAN RTX 5.6x faster), with a GradScaler
# hand-in: run.py sets False, so backbones are built empty and every weight comes from our checkpoints (no base downloads)
PRETRAINED = True


def read_mask(scene):
    return cv2.imread(str(CACHE / "train" / scene / "mask.png"), cv2.IMREAD_UNCHANGED)


class SegCrops(Dataset):
    """Random crops of (degraded or clean) training frames with their scene mask. Rare scenes are drawn 3x as often
    and half their crops are centred on a rare-class pixel.
    ext: share of crops from unlabelled photos (pool "lsui": LSUI raw, cache/ext; "all": + UIIS10K/AquaOV255,
    cache/seg_unl from seg/build_unlabelled.py), colour-shifted onto a training GT and degraded with the forward colour
    map; mask all 255 except pasted objects, so they only teach through distillation (seg/student.py).
    pseudo: dir of seg/pseudo_label.py masks; the pool shrinks to the photos that have one, and it is their mask.
    dev, dev_masks: share of crops from the 1200 dev degraded frames with a Track 1 ensemble's per-image dev masks as
    labels. Track 2 student since 16 Oct; Track 1 segmenters since 26 Oct (Track 1 now scores new test images, the dev
    split is training data under the 17 Oct organizer ruling), with per-scene consensus masks (seg/dev_consensus.py)."""

    def __init__(self, scenes, crop=512, clean_p=0.1, length=10 ** 7, paste=0.0, ext=0.0, pool="all", pseudo="",
                 t2=False, dev=0.0, dev_masks=""):
        """t2: Track 2 geometry: each frame squashed like the Jetson's 384x768 input (x 768/W, y 384/H, x0.8-1.25)."""
        self.scenes, self.crop, self.clean_p, self.length, self.paste_p = scenes, crop, clean_p, length, paste
        self.t2 = t2
        self.dev, self.dev_masks = dev, ROOT / dev_masks
        self.dev_files = sorted((CACHE / "dev").glob("*/deg/*.png")) if dev else []
        assert not dev or len(self.dev_files) == 1200 and all(
            (self.dev_masks / f"{p.parts[-3]}_{p.name}").exists() for p in self.dev_files), "dev frames / masks missing"
        self.ext, self.ext_files, self.pseudo = ext, [], ROOT / pseudo if pseudo else None
        if ext:
            self.ext_files = sorted((CACHE / "ext").glob("*.png"))
            if pool in ("all", "allc"):
                self.ext_files += sorted((CACHE / "seg_unl").glob("*.jpg"))
            if pool == "allc":  # + Coralscapes V1/V2 reef frames, unlabelled (teacher soft labels only)
                self.ext_files += sorted((CACHE / "coralscapes").glob("*.jpg")) + sorted((CACHE / "coralscapes2").glob("*.jpg"))
            if self.pseudo:  # labelled external sets (seg/build_coralscapes.py) stay out of the distillation pool
                self.ext_files += sorted((CACHE / "coralscapes").glob("*.jpg")) + sorted((CACHE / "coralscapes2").glob("*.jpg"))
                self.ext_files = [p for p in self.ext_files if (self.pseudo / f"{p.stem}.png").exists()]
                assert self.ext_files, f"no pseudo masks in {self.pseudo}"
        self.masks = {s: read_mask(s) for s in scenes}
        self.rare_px = {s: np.argwhere(self.masks[s] >= 3) for s in scenes}
        self.w = np.array([3.0 if len(self.rare_px[s]) else 1.0 for s in scenes])
        self.w /= self.w.sum()
        if paste or ext:
            self.fwd = fit_poly_table(scenes, forward=True).numpy()  # training scenes only
            self.gt_ab = [lab_ab(imread(CACHE / "train" / s / "gt.png")) for s in scenes]
        if paste:
            bank = json.loads((CACHE / "seg_bank" / "bank.json").read_text())["objects"]
            self.bank = {c: [CACHE / "seg_bank" / r["file"] for r in bank if r["cls"] == c] for c in (3, 4)}

    def __len__(self):
        return self.length

    def paste(self, im, m, t, l, asp=1.0):
        """Copy-paste one bank object (elasmo 60% / other 40%) into the crop. The object's colour cast is moved onto a
        training GT's (Lab a*/b*), then degraded with the same (type, level) forward colour map as the frame.
        Global pointwise water model, no depth-dependent haze on the object."""
        c = 3 if random.random() < 0.6 else 4
        o = cv2.cvtColor(cv2.imread(str(random.choice(self.bank[c])), cv2.IMREAD_UNCHANGED), cv2.COLOR_BGRA2RGBA)
        k = random.uniform(0.2, 0.8) * self.crop / max(o.shape[0] * asp, o.shape[1])
        o = cv2.resize(o, None, fx=k, fy=k * asp, interpolation=cv2.INTER_AREA if k < 1 else cv2.INTER_LINEAR)
        if random.random() < 0.5:
            o = o[:, ::-1]
        rgb, a = o[..., :3], o[..., 3]
        lab = cv2.cvtColor(rgb.astype(np.float32) / 255, cv2.COLOR_RGB2LAB)
        sel = a > 127
        lab[..., 1:] += np.array(random.choice(self.gt_ab), np.float32) - lab[sel][:, 1:].mean(0)
        f = np.clip(cv2.cvtColor(lab, cv2.COLOR_LAB2RGB), 0, 1)
        if t:
            f = np.stack(poly_terms(f[..., 0], f[..., 1], f[..., 2]), -1) @ self.fwd[cell(t, l)]
        h, w = a.shape
        y0, x0 = random.randint(0, self.crop - h), random.randint(0, self.crop - w)
        al = cv2.GaussianBlur(a, (3, 3), 0)[..., None].astype(np.float32) / 255
        win = im[y0:y0 + h, x0:x0 + w].astype(np.float32)
        im[y0:y0 + h, x0:x0 + w] = (win * (1 - al) + np.clip(f * 255, 0, 255) * al).round().astype(np.uint8)
        m[y0:y0 + h, x0:x0 + w][sel] = c

    def unlabelled(self, t, l):
        """Unlabelled photo with a training GT's colour cast, degraded to cell (t, l); mask all ignore or pseudo."""
        p = random.choice(self.ext_files)
        y = imread(p)
        lab = cv2.cvtColor(y.astype(np.float32) / 255, cv2.COLOR_RGB2LAB)
        lab[..., 1:] += np.array(random.choice(self.gt_ab), np.float32) - lab[..., 1:].reshape(-1, 2).mean(0)
        f = np.clip(cv2.cvtColor(lab, cv2.COLOR_LAB2RGB), 0, 1)
        if t:
            f = np.stack(poly_terms(f[..., 0], f[..., 1], f[..., 2]), -1) @ self.fwd[cell(t, l)]
        if self.pseudo:
            m = cv2.resize(cv2.imread(str(self.pseudo / f"{p.stem}.png"), cv2.IMREAD_UNCHANGED), (y.shape[1], y.shape[0]),
                           interpolation=cv2.INTER_NEAREST)
        else:
            m = np.full(y.shape[:2], 255, np.uint8)
        return np.clip(f * 255, 0, 255).round().astype(np.uint8), m

    def __getitem__(self, i):
        s = self.scenes[np.random.choice(len(self.scenes), p=self.w)]
        t, l = (0, 0) if random.random() < self.clean_p else (random.choice(TYPES), random.choice(LEVELS))
        if self.dev and random.random() < self.dev:
            p = random.choice(self.dev_files)
            t, l = map(int, p.stem.split("_"))  # the frame's own cell, so pasted objects get the same water
            s, im = p.parts[-3], imread(p)
            m = cv2.imread(str(self.dev_masks / f"{s}_{p.name}"), cv2.IMREAD_UNCHANGED)
            rare = np.argwhere(m >= 3)
        elif self.ext and random.random() < self.ext:
            s, (im, m), rare = None, self.unlabelled(t, l), ()
        else:
            im = imread(CACHE / "train" / s / ("gt.png" if not t else f"deg/{t}_{l}.png"))
            m, rare = self.masks[s], self.rare_px[s]
        sc = random.uniform(0.6, 1.25)
        if s is None:  # unlabelled photos are small: fill the crop instead of mirror-padding
            sc = max(sc, self.crop / min(im.shape[:2]))
        fx = fy = sc
        if self.t2:  # unlabelled photos borrow a random training frame's squash, then grow to fill the crop
            fh, fw = (im.shape[:2] if s else self.masks[random.choice(self.scenes)].shape)
            j = random.uniform(0.8, 1.25)
            fx, fy = j * 768 / fw, j * 384 / fh
            if s is None:
                g = max(1.0, self.crop / min(im.shape[0] * fy, im.shape[1] * fx))
                fx, fy = fx * g, fy * g
        im = cv2.resize(im, None, fx=fx, fy=fy, interpolation=cv2.INTER_AREA if max(fx, fy) < 1 else cv2.INTER_LINEAR)
        m = cv2.resize(m, (im.shape[1], im.shape[0]), interpolation=cv2.INTER_NEAREST)
        c = self.crop
        ph, pw = max(0, c - im.shape[0]), max(0, c - im.shape[1])
        if ph or pw:
            im = cv2.copyMakeBorder(im, 0, ph, 0, pw, cv2.BORDER_REFLECT)
            m = cv2.copyMakeBorder(m, 0, ph, 0, pw, cv2.BORDER_CONSTANT, value=255)
        H, W = m.shape
        if len(rare) and random.random() < 0.5:
            cy, cx = (rare[random.randrange(len(rare))] * (fy, fx)).astype(int)
            y0 = int(np.clip(cy - c // 2 + random.randint(-c // 4, c // 4), 0, H - c))
            x0 = int(np.clip(cx - c // 2 + random.randint(-c // 4, c // 4), 0, W - c))
        else:
            y0, x0 = random.randint(0, H - c), random.randint(0, W - c)
        im, m = im[y0:y0 + c, x0:x0 + c].copy(), m[y0:y0 + c, x0:x0 + c].copy()
        if self.paste_p and random.random() < self.paste_p:
            self.paste(im, m, t, l, asp=fy / fx)
        if random.random() < 0.5:
            im, m = im[:, ::-1], m[:, ::-1]
        x = torch.from_numpy(np.ascontiguousarray(im)).permute(2, 0, 1).float() / 255
        return x, torch.from_numpy(np.ascontiguousarray(m)).long()


def seed_worker(i):
    np.random.seed(torch.initial_seed() % 2 ** 32)


def cbr(i, o, k=3, s=1):
    return nn.Sequential(nn.Conv2d(i, o, k, s, k // 2, bias=False), nn.BatchNorm2d(o), nn.GELU())


ARCH = "vit_large_patch16_dinov3.lvd1689m"
RADIO = ("nvidia/C-RADIOv3-L", "9d0413465e8a91e67bbf2c1ad342815478d1b906")  # NVIDIA Open Model License, pinned


class RadioBB(nn.Module):
    """C-RADIOv3-L (ViT-L/16 distilled from DFN CLIP, SigLIP2, DINOv2-g, SAM) posing as a timm ViT for DinoSeg
    (--arch radio). Same maths as RADIO's forward_intermediates (conditioner, CPE patches, blocks, summary tokens
    dropped), plus gradient checkpointing. Takes [0, 1] input: RADIO normalises it itself."""

    def __init__(self):
        super().__init__()
        from transformers import AutoModel
        r = AutoModel.from_pretrained(RADIO[0], revision=RADIO[1], trust_remote_code=True).radio_model
        self.blocks = r.model.blocks  # registered before r: parameter names start "blocks." for the lr groups
        self.r = r
        self.patch_embed = SimpleNamespace(patch_size=(r.patch_size, r.patch_size))
        self.pretrained_cfg = {"mean": (0.0, 0.0, 0.0), "std": (1.0, 1.0, 1.0)}
        self.ckpt = False

    def set_grad_checkpointing(self, on=True):
        self.ckpt = on

    def forward_intermediates(self, x, indices, intermediates_only=True):
        B, _, H, W = x.shape
        p, r = self.r.patch_size, self.r
        t = r.model.patch_generator(r.input_conditioner(x))
        out = []
        for i, blk in enumerate(self.blocks):
            t = torch.utils.checkpoint.checkpoint(blk, t, use_reentrant=False) if self.ckpt and t.requires_grad else blk(t)
            if i in indices:
                out.append(t[:, r.num_summary_tokens:].reshape(B, H // p, W // p, -1).permute(0, 3, 1, 2))
        return out


def deconvs(c, k):
    """k stride-2 transposed convs (ViTDet-style upsampling of a single-scale ViT map)."""
    return nn.Sequential(*[m for _ in range(k) for m in (nn.ConvTranspose2d(c, c, 2, 2), nn.BatchNorm2d(c), nn.GELU())])


class DinoSeg(nn.Module):
    """Any timm backbone with forward_intermediates (plain ViT/EVA: 4 evenly spaced blocks; ConvNeXt: its 4 stages).
    head "simple": all features resized to 1/16, concatenated, fused with a 1/4 conv detail branch (t1_13).
    head "fpn": UPerNet-style pyramid at strides 4/8/16/32 (ViT maps upsampled by transposed convs), top-down sums,
    all levels + detail branch fused at 1/4."""

    def __init__(self, arch=ARCH, layers=None, dim=256, head="simple"):
        super().__init__()
        vit = not arch.startswith("convnext")
        kw = {"dynamic_img_size": True} if vit and "dinov3" not in arch else {}  # DINOv3 (RoPE) is size-free already
        self.bb = RadioBB() if arch == "radio" else timm.create_model(arch, pretrained=PRETRAINED, **kw)
        n = len(self.bb.blocks) if vit else len(self.bb.stages)
        self.layers = list(layers or ([n // 4 * k - 1 for k in (1, 2, 3, 4)] if vit else range(n)))
        self.mult = self.bb.patch_embed.patch_size[0] if vit else 32
        cfg = self.bb.pretrained_cfg
        self.register_buffer("mean", torch.tensor(cfg["mean"]).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor(cfg["std"]).view(1, 3, 1, 1))
        with torch.no_grad():
            ch = [f.shape[1] for f in self.bb.forward_intermediates(
                torch.zeros(1, 3, 8 * self.mult, 8 * self.mult), indices=self.layers, intermediates_only=True)]
        self.head = head
        self.detail = nn.Sequential(cbr(3, 32, 3, 2), cbr(32, 64, 3, 2))  # 1/4 edges for small fish
        if head == "simple":
            self.proj = nn.Sequential(nn.BatchNorm2d(sum(ch)), cbr(sum(ch), dim, 1), cbr(dim, dim))
            self.fuse = nn.Sequential(cbr(dim + 64, 128), cbr(128, 128))
        else:
            self.lat = nn.ModuleList(cbr(c, dim, 1) for c in ch)
            self.ups = nn.ModuleList(deconvs(dim, k if vit else 0) for k in (2, 1, 0, 0))
            self.smooth = nn.ModuleList(cbr(dim, dim) for _ in ch)
            self.fuse = nn.Sequential(cbr(dim * len(ch) + 64, 256), cbr(256, 128))
        self.cls = nn.Conv2d(128, N_CLASSES, 1)

    def forward(self, x):
        H, W = x.shape[-2:]
        m = self.mult
        xp = F.pad(x, (0, (-W) % m, 0, (-H) % m), mode="reflect")
        Hp, Wp = xp.shape[-2:]
        xn = (xp - self.mean) / self.std
        f = self.bb.forward_intermediates(xn, indices=self.layers, intermediates_only=True)
        d = self.detail(xn)
        up = dict(mode="bilinear", align_corners=False)
        if self.head == "simple":
            sz = min(f, key=lambda t: abs(Wp / t.shape[-1] - 16)).shape[-2:]  # the ~1/16 map
            f = self.proj(torch.cat([fi if fi.shape[-2:] == sz else F.interpolate(fi, size=sz, **up) for fi in f], 1))
            y = self.cls(self.fuse(torch.cat([F.interpolate(f, size=d.shape[-2:], **up), d], 1)))
        else:
            ts = [(math.ceil(Hp / s), math.ceil(Wp / s)) for s in (4, 8, 16, 32)]
            p = [F.interpolate(u(l(fi)), size=t, **up) for fi, l, u, t in zip(f, self.lat, self.ups, ts)]
            for i in (2, 1, 0):
                p[i] = p[i] + F.interpolate(p[i + 1], size=ts[i], **up)
            p = [F.interpolate(s(pi), size=d.shape[-2:], **up) for s, pi in zip(self.smooth, p)]
            y = self.cls(self.fuse(torch.cat(p + [d], 1)))
        return F.interpolate(y, size=xp.shape[-2:], **up)[..., :H, :W]


class _Pyramid(nn.Module):
    """timm ViT -> 4 maps at strides 4/8/16/32 with the given channels (ViTDet simple pyramid), posing as the
    Mask2Former backbone (its pixel decoder reads .feature_maps / .channels)."""

    def __init__(self, arch, chans):
        super().__init__()
        self.bb = timm.create_model(arch, pretrained=PRETRAINED)
        n = len(self.bb.blocks)
        self.layers, self.channels = [n // 4 * k - 1 for k in (1, 2, 3, 4)], chans
        self.lat = nn.ModuleList(cbr(self.bb.embed_dim, c, 1) for c in chans)
        self.ups = nn.ModuleList(deconvs(c, k) for c, k in zip(chans, (2, 1, 0, 0)))

    def forward(self, x):
        f = self.bb.forward_intermediates(x, indices=self.layers, intermediates_only=True)
        ts = [(math.ceil(x.shape[-2] / s), math.ceil(x.shape[-1] / s)) for s in (4, 8, 16, 32)]
        return SimpleNamespace(feature_maps=tuple(
            F.interpolate(u(l(fi)), size=t, mode="bilinear", align_corners=False)
            for fi, l, u, t in zip(f, self.lat, self.ups, ts)))


class M2FSeg(nn.Module):
    """Mask2Former (mask classification: 100 queries, Hungarian-matched mask + class losses), initialised from the
    ADE20K-semantic Swin-L model (MIT). arch "swin": that model as is (new 5-class classifier); any timm ViT: its
    pyramid replaces the Swin backbone, the pretrained pixel/transformer decoders are kept as init.
    forward() returns log of the per-class mask scores (sum_q p(c|q) sigmoid(mask_q)), so predict()/Ensemble work."""
    SRC = "facebook/mask2former-swin-large-ade-semantic"

    def __init__(self, arch="swin"):
        super().__init__()
        from transformers import Mask2FormerConfig, Mask2FormerForUniversalSegmentation
        lab = dict(enumerate(NAMES))
        kw = dict(num_labels=N_CLASSES, id2label=lab, label2id={v: k for k, v in lab.items()})
        self.m = (Mask2FormerForUniversalSegmentation.from_pretrained(self.SRC, ignore_mismatched_sizes=True, **kw)
                  if PRETRAINED else  # hand-in: SRC = a folder with the model's config.json only
                  Mask2FormerForUniversalSegmentation(Mask2FormerConfig.from_pretrained(self.SRC, **kw)))
        pl = self.m.model.pixel_level_module
        if arch != "swin":
            pl.encoder = _Pyramid(arch, list(pl.encoder.channels))
        self.backbone = pl.encoder.bb if arch != "swin" else pl.encoder
        self.register_buffer("mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))

    def _pad(self, x, value):
        H, W = x.shape[-2:]
        return F.pad(x, (0, (-W) % 32, 0, (-H) % 32), mode="reflect" if value is None else "constant",
                     **({} if value is None else {"value": value}))

    def forward(self, x):
        H, W = x.shape[-2:]
        xp = self._pad(x, None)
        out = self.m(pixel_values=(xp - self.mean) / self.std)
        cls = out.class_queries_logits.float().softmax(-1)[..., :-1]
        seg = torch.einsum("bqc,bqhw->bchw", cls, out.masks_queries_logits.float().sigmoid())
        seg = F.interpolate(seg, size=xp.shape[-2:], mode="bilinear", align_corners=False)[..., :H, :W]
        return seg.clamp_min(1e-6).log()

    def loss(self, x, m):
        """bf16 forward; matching + losses in fp32 outside autocast (scipy's matcher cannot take bf16).
        Ignore pixels (255, borders only) count as negatives for every mask."""
        xp, mp = self._pad(x, None), self._pad(m[:, None].float(), 255.0)[:, 0].long()
        with torch.autocast("cuda", AMP):
            out = self.m(pixel_values=(xp - self.mean) / self.std, output_auxiliary_logits=True)
        ml, cl = [], []
        for mi in mp:
            cs = [c for c in range(N_CLASSES) if (mi == c).any()]
            ml.append(torch.stack([(mi == c).float() for c in cs]) if cs else mi.new_zeros((0, *mi.shape), dtype=torch.float))
            cl.append(torch.tensor(cs, dtype=torch.long, device=mi.device))
        aux = [{k: v.float() for k, v in d.items()} for d in out.auxiliary_logits]
        ld = self.m.get_loss_dict(out.masks_queries_logits.float(), out.class_queries_logits.float(), ml, cl, aux)
        return self.m.get_loss(ld)


class EpflBgCoral(nn.Module):
    """EPFL-ECEO/coralscapesv2-dinov3-vitl-lora-dpt (Apache-2.0: HF DINOv3 ViT-L/16 with merged LoRA + DPT head, 39
    Coralscapes V1 classes, trained on CoralscapesV2) as a coral-vs-background expert: forward() is its coral share of
    coral + background (V1 LUT of seg/build_coralscapes.py); Ensemble mixes that into the members' bg/coral split
    with weight w, their thing classes untouched. --ens spec "epfl:<w>[:<input scale>]"; weights in external/seg/epfl_cs2_dpt."""
    CORAL, BG = [3, 4, 6, 10, 12, 16, 17, 19, 20, 21, 22, 23, 25, 27, 28, 29, 30, 31, 32, 33, 34, 36, 37], [5, 13]
    WEIGHTS = ROOT / "external/seg/epfl_cs2_dpt/model.safetensors"  # hand-in: run.py points this into weights/

    def __init__(self, w, scale=1.0):
        from safetensors.torch import load_file
        from transformers import DINOv3ViTConfig, DINOv3ViTModel, DPTConfig
        from transformers.models.dpt.modeling_dpt import DPTNeck, DPTSemanticSegmentationHead
        super().__init__()
        self.w, self.scale = w, scale
        self.encoder = DINOv3ViTModel(DINOv3ViTConfig(hidden_size=1024, intermediate_size=4096, num_attention_heads=16,
                                                      num_hidden_layers=24, num_register_tokens=4))
        cfg = DPTConfig(hidden_size=1024, num_labels=40, patch_size=16, is_hybrid=False, readout_type="project",
                        neck_hidden_sizes=[96, 192, 384, 768], reassemble_factors=[4, 2, 1, 0.5],
                        fusion_hidden_size=256, head_in_index=-1, use_batch_norm_in_fusion_residual=True)
        self.neck, self.seg_head = DPTNeck(cfg), DPTSemanticSegmentationHead(cfg)
        self.load_state_dict(load_file(self.WEIGHTS))
        self.register_buffer("mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))

    def forward(self, x):
        H, W = x.shape[-2:]
        h, w = round(H * self.scale / 16) * 16, round(W * self.scale / 16) * 16
        x = F.interpolate((x - self.mean) / self.std, size=(h, w), mode="bilinear", align_corners=False)
        hs = self.encoder(pixel_values=x, output_hidden_states=True).hidden_states
        hs = hs[len(hs) - 24:]  # drop the embedding output if present; DPT layers 0/8/15/23, registers removed
        feats = [torch.cat([hs[i][:, :1], hs[i][:, 5:]], 1) for i in (0, 8, 15, 23)]
        y = self.seg_head(self.neck(feats, patch_height=h // 16, patch_width=w // 16))
        p = F.interpolate(y.float(), size=(H, W), mode="bilinear", align_corners=False).softmax(1)
        c, b = p[:, self.CORAL].sum(1, keepdim=True), p[:, self.BG].sum(1, keepdim=True)
        return c / (c + b).clamp_min(1e-6)


class VkSeg(nn.Module):
    """Track 1 ViT-B DPT segmenter: EPFL-ECEO
    coralscapesv2-dinov3-vitb-lora-dpt with a new 5-class head, fine-tuned on 16 EURS scenes + CoralscapesV2 + USIS16K.
    Raw frame, reflect-padded to /16, per-image standardisation; returns 5-class logits. Ensemble mixes its softmax in
    at weight w after the EPFL re-split. --ens spec "vk:<w>:<checkpoint>"."""

    def __init__(self, w, ckpt):
        from transformers import DINOv3ViTConfig, DINOv3ViTModel, DPTConfig
        from transformers.models.dpt.modeling_dpt import DPTNeck, DPTSemanticSegmentationHead
        super().__init__()
        self.w = w
        self.encoder = DINOv3ViTModel(DINOv3ViTConfig(hidden_size=768, intermediate_size=3072, num_attention_heads=12,
                                                      num_hidden_layers=12, num_register_tokens=4))
        cfg = DPTConfig(hidden_size=768, num_labels=5, patch_size=16, is_hybrid=False, readout_type="project",
                        neck_hidden_sizes=[96, 192, 384, 768], reassemble_factors=[4, 2, 1, 0.5],
                        fusion_hidden_size=256, head_in_index=-1, use_batch_norm_in_fusion_residual=True)
        self.neck, self.seg_head = DPTNeck(cfg), DPTSemanticSegmentationHead(cfg)
        self.load_state_dict(torch.load(ROOT / ckpt, map_location="cpu", weights_only=True))
        self.register_buffer("mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1), persistent=False)
        self.register_buffer("std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1), persistent=False)

    def forward(self, x):
        H, W = x.shape[-2:]
        x = F.pad(x, (0, -W % 16, 0, -H % 16), mode="reflect")
        m, s = x.mean((2, 3), keepdim=True), x.std((2, 3), keepdim=True).clamp_min(1 / 255)
        x = (((x - m) / s).clamp(-4, 4) * 0.225 + 0.45 - self.mean) / self.std
        hs = self.encoder(pixel_values=x, output_hidden_states=True).hidden_states
        hs = hs[len(hs) - 12:]  # DPT layers 0/4/7/11 (hub default for 12 blocks), registers removed
        feats = [torch.cat([hs[i][:, :1], hs[i][:, 5:]], 1) for i in (0, 4, 7, 11)]
        y = self.seg_head(self.neck(feats, patch_height=x.shape[-2] // 16, patch_width=x.shape[-1] // 16))
        return F.interpolate(y.float(), size=x.shape[-2:], mode="bilinear", align_corners=False)[..., :H, :W]


def load_seg(ckpt, dev):
    """Segmenter from a checkpoint, rebuilt with the arch/head in the args.json next to it (older runs: defaults)."""
    if ckpt.startswith("epfl:"):
        return EpflBgCoral(*map(float, ckpt[5:].split(":"))).to(dev).eval()
    if ckpt.startswith("vk:"):
        w, path = ckpt[3:].split(":", 1)
        return VkSeg(float(w), path).to(dev).eval()
    a = json.loads((ROOT / ckpt).parent.joinpath("args.json").read_text())
    arch, head = a.get("arch", ARCH), a.get("head", "simple")
    m = (M2FSeg(arch) if head == "m2f" else DinoSeg(arch, head=head)).to(dev).eval()
    m.load_state_dict(torch.load(ROOT / ckpt, map_location=dev))
    return m


def seg_loss(logits, m, w):
    if not (m != 255).any():  # all-unlabelled batch (SegCrops ext): CE would be NaN
        return logits.sum() * 0
    ce = F.cross_entropy(logits, m, weight=w, ignore_index=255)
    valid = (m != 255).unsqueeze(1)
    p = logits.float().softmax(1) * valid
    oh = F.one_hot(m.clamp(max=N_CLASSES - 1), N_CLASSES).permute(0, 3, 1, 2) * valid
    inter, den = (p * oh).sum((0, 2, 3)), (p + oh).sum((0, 2, 3))
    present = oh.sum((0, 2, 3)) > 0  # Dice only over classes in the batch
    dice = 1 - (2 * inter[present] + 1) / (den[present] + 1)
    return ce + dice.mean()


@torch.no_grad()
def predict(model, x, scales=(1.0,), flip=False, size=None):
    """Softmax averaged over scales (and hflip), at the input resolution.
    size: run the model on x resized to this (H, W), e.g. the Track 2 static input (384, 768)."""
    H, W = x.shape[-2:]
    if size:
        x = F.interpolate(x, size=size, mode="bilinear", align_corners=False)
    acc = 0
    for s in scales:
        xs = x if s == 1 else F.interpolate(x, scale_factor=s, mode="bilinear", align_corners=False)
        for fl in ((False, True) if flip else (False,)):
            xi = xs.flip(-1) if fl else xs
            with torch.autocast("cuda", AMP):
                y = model(xi).float()
            y = y.flip(-1) if fl else y
            acc = acc + F.interpolate(y, size=(H, W), mode="bilinear", align_corners=False).softmax(1)
    return acc


def to_x(im, dev):
    return torch.from_numpy(np.ascontiguousarray(im)).permute(2, 0, 1).float().div(255)[None].to(dev)


def decide(prob, min_frac=0.0):
    """argmax; with min_frac, a rare class (elasmo, other) covering less than that share of the image is dropped
    (its pixels take the next-best class): cuts false positives, which cost IoU across the whole set."""
    pred = prob.argmax(1)
    if min_frac:
        drop = [c for c in (3, 4) if 0 < (pred == c).float().mean() < min_frac]
        if drop:
            prob = prob.clone()
            prob[:, drop] = -1
            pred = prob.argmax(1)
    return pred[0].cpu().numpy().astype(np.uint8)


@torch.no_grad()
def evaluate(model, scenes, dev, levels=(5, 25, 50, 75, 100), min_fracs=(0.0,), **tta):
    """Per-image Qseg (mean), dataset-level per-class IoU, Qseg by level (0 = clean GT input).
    Main numbers use min_fracs[0]; q_ds for every min_frac is reported under 'q_ds_mf'."""
    model.eval()
    cm_all, qs, by_level, per_scene = np.zeros((N_CLASSES, N_CLASSES), np.int64), [], {}, {}
    cm_mf = {mf: np.zeros_like(cm_all) for mf in min_fracs}
    for s in scenes:
        gt = read_mask(s)
        items = [(0, 0, CACHE / "train" / s / "gt.png")] + [
            (t, l, CACHE / "train" / s / "deg" / f"{t}_{l}.png") for t in TYPES for l in levels]
        cm_s, q_s = np.zeros_like(cm_all), []
        for t, l, p in items:
            prob = predict(model, to_x(imread(p), dev), **tta)
            for mf in min_fracs[1:]:
                cm_mf[mf] += confusion(decide(prob, mf), gt)
            pred = decide(prob, min_fracs[0])
            cm = confusion(pred, gt)
            cm_mf[min_fracs[0]] += cm
            cm_s += cm
            q = from_confusion(cm)[0]
            q_s.append(q)
            by_level.setdefault(l, []).append(q)
        cm_all += cm_s
        qs += q_s
        iou_s = from_confusion(cm_s)[4]  # NaN = class absent from GT and prediction
        per_scene[s] = f"q {np.mean(q_s):.3f} iou " + " ".join(f"{v:.2f}" for v in iou_s)
    q_ds, miou, f1, cov, iou = from_confusion(cm_all)
    model.train()
    return {"q_img": float(np.mean(qs)), "zeroed": float(np.mean([q == 0 for q in qs])), "q_ds": float(q_ds),
            "miou": float(miou), **{f"iou_{n}": float(v) for n, v in zip(NAMES, iou)},
            **{f"q_l{l}": float(np.mean(v)) for l, v in sorted(by_level.items())}, "scenes": per_scene,
            "q_ds_mf": {str(mf): round(float(from_confusion(c)[0]), 4) for mf, c in cm_mf.items()},
            "cm": cm_all.tolist()}  # summed over the 5 folds by seg/cv_report.py


class Ensemble(nn.Module):
    """Mean softmax of several segmenters, returned as log-probabilities (predict() re-softmaxes to the mean)."""

    def __init__(self, models):
        super().__init__()
        self.models = nn.ModuleList(models)

    def forward(self, x):
        ref = [m for m in self.models if isinstance(m, EpflBgCoral)]
        vk = [m for m in self.models if isinstance(m, VkSeg)]
        own = [m for m in self.models if m not in ref and m not in vk]
        p = torch.stack([m(x).float().softmax(1) for m in own]).mean(0) if own else 0
        for m in ref:  # re-split the bg + coral mass: coral share (1 - w) * members' + w * expert's
            s = p[:, :2].sum(1, keepdim=True)
            r = (1 - m.w) * p[:, 1:2] / s.clamp_min(1e-8) + m.w * m(x).float()
            p = torch.cat([s * (1 - r), s * r, p[:, 2:]], 1)
        for m in vk:  # soft mix: (1 - w) * ensemble + w * ViT-B DPT
            p = (1 - m.w) * p + m.w * m(x).float().softmax(1)
        return p.clamp_min(1e-8).log()


@torch.no_grad()
def predict_dev(model, out, dev, min_frac=0.0, split="dev", **tta):
    """Masks for every frame of cache/<split> (dev, or a split from scripts/prepare_cache.py --split)."""
    out.mkdir(parents=True, exist_ok=True)
    model.eval()
    for stem, p in split_items(split):
        pred = decide(predict(model, to_x(imread(p), dev), **tta), min_frac)
        cv2.imwrite(str(out / f"{stem}.png"), pred)
    print("wrote", len(list(out.glob("*.png"))), split, "masks to", out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fold", type=int, default=0, help="-1 = train on all 20 scenes")
    ap.add_argument("--split", default="seg", choices=["seg", "rest"], help="rest = restoration FOLDS")
    ap.add_argument("--name", required=True)
    ap.add_argument("--iters", type=int, default=8000)
    ap.add_argument("--bs", type=int, default=4)
    ap.add_argument("--crop", type=int, default=512)
    ap.add_argument("--lr", type=float, default=1e-3, help="head")
    ap.add_argument("--bb_lr", type=float, default=0.0, help="backbone top-block lr; 0 = frozen")
    ap.add_argument("--decay", type=float, default=0.9, help="layer-wise lr decay per block")
    ap.add_argument("--eval_every", type=int, default=2000)
    ap.add_argument("--paste", type=float, default=0.0, help="prob. of copy-pasting a rare bank object into a crop")
    ap.add_argument("--predict", action="store_true", help="after training, write dev masks with TTA")
    ap.add_argument("--eval_only", default="", help="checkpoint to evaluate (with and without TTA) instead of training")
    ap.add_argument("--ens", default="", help="comma-separated checkpoints: write dev masks from their mean softmax")
    ap.add_argument("--predict_split", default="dev",
                    help="--ens with --fold -1: cache/<split> to predict (prepare_cache.py --split), masks to "
                         "work/<name>_<split>_masks")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--arch", default=ARCH, help="timm backbone; with --head m2f also 'swin' (ADE20K Mask2Former)")
    ap.add_argument("--head", default="simple", choices=["simple", "fpn", "m2f"],
                    help="m2f: Mask2Former (use --lr 1e-4 --bb_lr 1e-5, its standard; flat backbone lr)")
    ap.add_argument("--freeze", type=int, default=0, help="keep the first N blocks (ConvNeXt: stages) + stem frozen")
    ap.add_argument("--tta_eval", action="store_true", help="fold runs: also evaluate last.pt with multi-scale+flip TTA")
    ap.add_argument("--ext", type=float, default=0.0, help="share of crops from pseudo-labelled photos (needs --pseudo)")
    ap.add_argument("--pseudo", default="", help="seg/pseudo_label.py mask dir; '{k}' -> fold")
    ap.add_argument("--dev", type=float, default=0.0, help="share of crops from dev frames labelled by --dev_masks")
    ap.add_argument("--dev_masks", default="", help="per-frame dev masks <scene>_<type>_<level>.png (seg/dev_consensus.py)")
    ap.add_argument("--amp", default="bf16", choices=["bf16", "fp16"])
    a = ap.parse_args()
    global AMP
    AMP = torch.float16 if a.amp == "fp16" else torch.bfloat16
    scaler = torch.amp.GradScaler(enabled=a.amp == "fp16")

    torch.manual_seed(a.seed)
    dev = torch.device("cuda")
    out = ROOT / "runs" / a.name
    out.mkdir(parents=True, exist_ok=True)
    (out / "args.json").write_text(json.dumps(vars(a), indent=1))
    val = (SEG_FOLDS if a.split == "seg" else FOLDS)[a.fold] if a.fold >= 0 else [RARE[0], _COMMON[0]]  # all-scene runs report seen scenes
    train = [s for s in ALL_SCENES if a.fold < 0 or s not in val]
    tta = dict(scales=(0.75, 1.0, 1.25), flip=True)

    if a.ens:  # with --fold k: CV of the ensemble, "{k}" in the paths -> k (scripts/seg_cv.sh <name> --ens ...)
        ens = Ensemble([load_seg(c.format(k=a.fold), dev) for c in a.ens.split(",")])
        if a.fold < 0:
            predict_dev(ens, ROOT / "work" / f"{a.name}{'' if a.predict_split == 'dev' else '_' + a.predict_split}_masks",
                        dev, split=a.predict_split)
        else:
            r = evaluate(ens, val, dev)
            (out / "metrics.json").write_text(json.dumps(r))
            print("EVAL", json.dumps({k: round(v, 4) if isinstance(v, float) else v for k, v in r.items()}), flush=True)
        return
    model = (M2FSeg(a.arch) if a.head == "m2f" else DinoSeg(a.arch, head=a.head)).to(dev)

    if a.eval_only:
        model.load_state_dict(torch.load(a.eval_only, map_location=dev))
        mfs = (0.0, 0.001, 0.003, 0.01, 0.02, 0.05)
        print("EVAL", json.dumps(evaluate(model, val, dev, min_fracs=mfs)), flush=True)
        print("EVAL_TTA", json.dumps(evaluate(model, val, dev, min_fracs=mfs, **tta)), flush=True)
        return

    bb_params = []
    if a.head == "m2f":
        if hasattr(model.backbone, "set_grad_checkpointing"):
            model.backbone.set_grad_checkpointing(True)
        bb_params = [{"params": list(model.backbone.parameters()), "lr": a.bb_lr}]
        ids = {id(p) for p in model.backbone.parameters()}
        head = [p for p in model.parameters() if id(p) not in ids]
    elif a.bb_lr > 0:
        model.bb.set_grad_checkpointing(True)
        key = "blocks" if hasattr(model.bb, "blocks") else "stages"
        blocks = getattr(model.bb, key)
        n = len(blocks)
        for i, blk in enumerate(blocks):  # decay exponent scaled so the bottom gets decay**24 whatever the depth
            if i < a.freeze:
                blk.requires_grad_(False)
                continue
            bb_params.append({"params": list(blk.parameters()), "lr": a.bb_lr * a.decay ** ((n - 1 - i) * 24 / n)})
        rest = [p for nm, p in model.bb.named_parameters() if not nm.startswith(key + ".")]
        if a.freeze:
            for p in rest:
                p.requires_grad_(False)
        else:
            bb_params.append({"params": rest, "lr": a.bb_lr * a.decay ** 24})
    else:
        model.bb.requires_grad_(False)
    if a.head != "m2f":
        head = [p for nm, p in model.named_parameters() if not nm.startswith("bb.")]
    groups = bb_params + [{"params": head, "lr": a.lr}]
    for g in groups:
        g["base_lr"] = g["lr"]
    opt = torch.optim.AdamW(groups, weight_decay=0.01)
    dl = DataLoader(SegCrops(train, a.crop, paste=a.paste, ext=a.ext, pseudo=a.pseudo.format(k=a.fold), dev=a.dev,
                             dev_masks=a.dev_masks), batch_size=a.bs, num_workers=6, persistent_workers=True,
                    worker_init_fn=seed_worker)
    w = CLASS_W.to(dev)
    print("train", train, "\nval", val, flush=True)
    log, best, t0 = open(out / "log.jsonl", "a"), -1, time.time()
    ck, start = out / "ckpt.pt", 0  # resume point: a WSL "CUDA unknown error" kills runs at random steps
    if ck.exists():
        s = torch.load(ck, map_location=dev)
        model.load_state_dict(s["model"]), opt.load_state_dict(s["opt"]), scaler.load_state_dict(s["scaler"])
        start, best = s["it"], s["best"]
        print(f"resumed from {ck} at it {start}", flush=True)
    model.train()
    if a.bb_lr == 0:
        model.bb.eval()
    for it, (x, m) in enumerate(dl, start + 1):
        if it > a.iters:
            break
        f = min(1, it / 500) * 0.5 * (1 + math.cos(math.pi * it / a.iters))  # warmup + cosine
        for g in opt.param_groups:
            g["lr"] = g["base_lr"] * f
        x, m = x.to(dev, non_blocking=True), m.to(dev, non_blocking=True)
        if a.head == "m2f":
            loss = model.loss(x, m)
        else:
            with torch.autocast("cuda", AMP):
                logits = model(x)
            loss = seg_loss(logits.float(), m, w)
        opt.zero_grad(set_to_none=True)
        scaler.scale(loss).backward()
        scaler.unscale_(opt)
        torch.nn.utils.clip_grad_norm_([p for g in groups for p in g["params"]], 1.0)
        scaler.step(opt)
        scaler.update()
        if it % 100 == 0:
            print(f"it {it} loss {loss.item():.4f} {time.time() - t0:.0f}s", flush=True)
        if it % a.eval_every == 0 or it == a.iters:
            r = {"it": it, **evaluate(model, val, dev)}
            if a.bb_lr == 0:
                model.bb.eval()
            log.write(json.dumps(r) + "\n"), log.flush()
            print("EVAL", json.dumps({k: round(v, 4) if isinstance(v, float) else v for k, v in r.items()}), flush=True)
            torch.save(model.state_dict(), out / "last.pt")
            if r["q_ds"] > best and a.eval_every < a.iters:
                best = r["q_ds"]
                torch.save(model.state_dict(), out / "best.pt")
        if it % 1000 == 0 and it < a.iters:
            torch.save({"model": model.state_dict(), "opt": opt.state_dict(), "scaler": scaler.state_dict(),
                        "it": it, "best": best}, out / "ckpt.tmp")
            (out / "ckpt.tmp").replace(ck)
    ck.unlink(missing_ok=True)
    if a.fold >= 0:
        (out / "metrics.json").write_text(json.dumps(r))  # last.pt, no TTA: the CV number (seg/cv_report.py)
    if a.fold >= 0 and a.tta_eval:
        model.load_state_dict(torch.load(out / "last.pt", map_location=dev))
        r = evaluate(model, val, dev, **tta)
        (out / "metrics_tta.json").write_text(json.dumps(r, indent=1))
        print("EVAL_TTA", json.dumps({k: round(v, 4) if isinstance(v, float) else v for k, v in r.items()}), flush=True)
    if a.predict:
        predict_dev(model, ROOT / "work" / f"{a.name}_masks", dev)  # no TTA: hurt held out (seg_ft_f0, seg_frz_f0)


if __name__ == "__main__":
    main()
