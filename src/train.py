"""Train a restorer on EURS with a scene-grouped holdout.

    python restoration/train.py --model phys --fold 0 --iters 20000 --name phys_f0
    python restoration/train.py --model big --fold -1 ...   # fold -1 = train on all 20 scenes
"""
import argparse
import json
import time

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from eurs import (ALL_SCENES, CACHE, FOLDS, MODELS, ROOT, CropPairs, apply_grid, cell, collate_t2, count_params,
                  deg_path, fit_poly_table, fit_table, frame_stats, imread, load_grids, psnr, restoration_score,
                  sharpness, ssim, thumbnail, to_tensor, val_items)

T2_SIZE = (384, 768)


def run_full(model, x, pad=16, cell=None, j0=None):
    """Full-frame inference: global inputs from the true frame, reflect-pad to a multiple of `pad`.
    cell: optional true (type, level) class tensor (B,) from the filename (see PhysNet.physics).
    j0: optional external first pass (GridNet grid applied to x, RestNet only)."""
    th, st = thumbnail(x), frame_stats(x)
    H, W = x.shape[-2:]
    p = lambda v: F.pad(v, (0, (-W) % pad, 0, (-H) % pad), mode="reflect")
    kw = ({} if cell is None else {"cell": cell}) | ({} if j0 is None else {"j0": p(j0)})
    y, j0, logits = model(p(x), th, st, **kw)
    return y[..., :H, :W], j0[..., :H, :W], logits


class FlipTTA(torch.nn.Module):
    """Test-time augmentation: average the restorer over a horizontal flip. Vertical flips cost 0.6 dB held out
    (light/haze vary with depth, training only flips horizontally)."""

    def __init__(self, m):
        super().__init__()
        self.m = m

    def forward(self, x, th, st, cell=None, j0=None):
        acc, first = 0, None
        for d in ((), (-1,)):
            f = (lambda t: t.flip(d)) if d else (lambda t: t)
            kw = ({} if cell is None else {"cell": cell}) | ({} if j0 is None else {"j0": f(j0)})
            y, j0_, logits = self.m(f(x), f(th), st, **kw)
            j0_ = f(j0_)
            acc = acc + f(y)
            first = first or (j0_, logits)
        return acc / 2, *first


@torch.no_grad()
def evaluate(model, scenes, dev, levels=(5, 25, 50, 75, 100), meta=False, amp=torch.bfloat16, t2=False, grids=False):
    """meta: feed the true (type, level) cell, as read from the filename at test time.
    t2: Track 2 path: frame area-resized to 384x768, output bilinearly upsampled back to the GT (scripts/t2_gap.py).
    grids: first pass from the out-of-fold GridNet grids (load_grids)."""
    model.eval()
    rows, hits = [], []
    gts = {s: to_tensor(imread(CACHE / "train" / s / "gt.png"))[None].to(dev) for s in scenes}
    grid = {s: load_grids("train", s) for s in scenes} if grids else None
    for s, t, l in val_items(scenes, levels):
        x = to_tensor(imread(deg_path("train", s, t, l)))[None].to(dev)
        g = gts[s]
        if t2:
            x = F.interpolate(x, size=T2_SIZE, mode="area")
        j0 = apply_grid(x, torch.from_numpy(grid[s][cell(t, l)]).float()[None].to(dev)) if grids else None
        with torch.autocast("cuda", amp):
            y, _, logits = run_full(model, x, cell=torch.tensor([cell(t, l)], device=dev) if meta else None, j0=j0)
        if t2:
            y = F.interpolate(y.float().clamp(0, 1), size=g.shape[-2:], mode="bilinear", align_corners=False)
        y = (y.float().clamp(0, 1) * 255).round() / 255  # score what we would save
        rows.append((s, t, l, psnr(y, g).item(), ssim(y, g).item(), (sharpness(y) / sharpness(g)).item()))
        if logits is not None:
            k = logits.argmax(1).item()
            hits.append((k // 20 + 1 == t, abs((k % 20 + 1) * 5 - l)))
    model.train()
    p, s_, sh = (float(np.mean([r[i] for r in rows])) for i in (3, 4, 5))
    m = {"psnr": p, "ssim": s_, "sharp": sh, "score": restoration_score(p, s_, sh),
         "score_mean": float(np.mean([restoration_score(r[3], r[4], r[5]) for r in rows]))}
    if hits:
        m.update(type_acc=float(np.mean([h[0] for h in hits])), level_mae=float(np.mean([h[1] for h in hits])))
    return m, rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="phys")
    ap.add_argument("--fold", type=int, default=0)
    ap.add_argument("--iters", type=int, default=20000)
    ap.add_argument("--bs", type=int, default=8)
    ap.add_argument("--crop", type=int, default=384)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--eval_every", type=int, default=2500)
    ap.add_argument("--w_ssim", type=float, default=0.2)
    ap.add_argument("--w_j0", type=float, default=0.3)
    ap.add_argument("--w_cls", type=float, default=0.05)
    ap.add_argument("--ext", type=float, default=0.0, help="fraction of samples from emulated external photos")
    ap.add_argument("--ext_match", action="store_true", help="shift external photos' a*/b* cast onto training GTs'")
    ap.add_argument("--meta", action="store_true", help="use the true (type, level) cell from the filename (train + eval)")
    ap.add_argument("--hard",type=float, default=1.0, help="sampling weight of water types 7-10 (others 1)")
    ap.add_argument("--table", default="poly", choices=["poly", "affine"])
    ap.add_argument("--loss", default="l1", choices=["l1", "psnr"],
                    help="psnr: per-image 10*log10(MSE), scaled 0.01 so its gradient is ~L1-sized (final-stage fine-tune)")
    ap.add_argument("--patience", type=float, default=0,
                    help="early stop after this many EURS image epochs (200 x train scenes) without a better val score")
    ap.add_argument("--init", default="")
    ap.add_argument("--extra", default="", help="extra training scenes under cache/train, e.g. dev_<scene> from dev_pseudo_gt.py")
    ap.add_argument("--resume", action="store_true", help="continue runs/<name>/resume.pt after a crash")
    ap.add_argument("--compile", action="store_true", help="torch.compile the refiner (restormer: 1.46x faster)")
    ap.add_argument("--amp", default="bf16", choices=["bf16", "fp16"], help="fp16 + loss scaling for pre-Ampere GPUs")
    ap.add_argument("--t2scale", type=int, default=0,
                    help="S > 0: train and evaluate on the Track 2 path (SxS squashed input, loss after upsampling)")
    ap.add_argument("--grids", action="store_true",
                    help="RestNet first pass = out-of-fold GridNet grid (cache/grids, restoration/gridnet.py --export)")
    ap.add_argument("--kd", type=float, default=0.0,
                    help="share of samples from dev frames with Track 1 restorer targets (scripts/kd_targets.py), Track 2")
    ap.add_argument("--name", required=True)
    a = ap.parse_args()
    assert not a.grids or (a.meta and not a.t2scale), "--grids needs --meta (grids are per cell) and full scale"
    amp = torch.float16 if a.amp == "fp16" else torch.bfloat16
    scaler = torch.amp.GradScaler(enabled=a.amp == "fp16")

    out = ROOT / "runs" / a.name
    out.mkdir(parents=True, exist_ok=True)
    if not a.resume:
        (out / "args.json").write_text(json.dumps(vars(a), indent=1))
    val = FOLDS[a.fold] if a.fold >= 0 else FOLDS[0][:2]  # all-scene runs still report (seen) scenes
    train = [s for s in ALL_SCENES if a.fold < 0 or s not in val] + [s for s in a.extra.split(",") if s]

    dev = "cuda"
    torch.backends.cudnn.benchmark = True
    model = MODELS[a.model](fit_poly_table(train) if a.table == "poly" else fit_table(train)).to(dev)
    if a.init:
        model.load_state_dict(torch.load(ROOT / a.init, map_location=dev), strict=False)
    print(a.name, "params", count_params(model), "train", len(train), "val", val, flush=True)
    if a.iters == 0:  # evaluate --init only
        print("EVAL", json.dumps(evaluate(model, val, dev, meta=a.meta, amp=amp, t2=a.t2scale > 0, grids=a.grids)[0]), flush=True)
        return
    if a.compile:
        model.net.compile()  # in place: state_dict keys unchanged
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=1e-4, betas=(0.9, 0.99))
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, a.lr, total_steps=a.iters, pct_start=0.03)
    start, best = 0, -1
    patience = round(a.patience * 200 * len(train) / (a.bs * (1 - a.ext)))  # image epochs -> iterations
    if a.resume:
        r = torch.load(out / "resume.pt", map_location=dev)
        model.load_state_dict(r["model"]), opt.load_state_dict(r["opt"]), sched.load_state_dict(r["sched"])
        start, best = r["it"], r["best"]
        print("resumed at", start, flush=True)
    best_it = r.get("best_it", start) if a.resume else 0
    dl = DataLoader(CropPairs(train, a.crop, (a.iters - start) * a.bs, ext=a.ext, hard=a.hard, ext_match=a.ext_match,
                              t2=a.t2scale, grids=a.grids, kd=a.kd), batch_size=a.bs, num_workers=6, persistent_workers=True, drop_last=True,
                    prefetch_factor=4, collate_fn=collate_t2 if a.t2scale else None)

    def recon(pred, j0, y):
        rec = (0.1 * torch.log10((pred - y).pow(2).mean((1, 2, 3)) + 1e-8).mean() if a.loss == "psnr"
               else F.l1_loss(pred, y))
        return rec + a.w_ssim * (1 - ssim(pred, y)).mean() + a.w_j0 * F.l1_loss(j0.clamp(0, 1), y)

    log = open(out / "log.jsonl", "a")
    t0, run = time.time(), 0.0
    for it, batch in enumerate(dl, start + 1):
        x, y, th, st, c, *g = (b.to(dev, non_blocking=True) if torch.is_tensor(b) else [v.to(dev) for v in b]
                           for b in batch)
        with torch.autocast("cuda", amp):
            pred, j0, logits = model(x, th, st, cell=c, **({"j0": g[0]} if g else {})) if a.meta else model(x, th, st)
        pred, j0 = pred.float(), j0.float()
        if a.t2scale:  # upsample each output to its full-scale target region, as the evaluator does
            up = lambda v, k: F.interpolate(v[k:k + 1], size=y[k].shape[-2:], mode="bilinear", align_corners=False)
            r = sum(recon(up(pred, k), up(j0, k), y[k][None]) for k in range(len(y))) / len(y)
        else:
            r = recon(pred, j0, y)
        loss = r + a.w_cls * F.cross_entropy(logits.float(), c)
        opt.zero_grad(set_to_none=True)
        scaler.scale(loss).backward()
        scaler.unscale_(opt)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
        scaler.step(opt)
        scaler.update()
        sched.step()
        run = 0.98 * run + 0.02 * loss.item()
        if it % 500 == 0:
            print(f"it {it} loss {run:.4f} {time.time() - t0:.0f}s", flush=True)
        if it % a.eval_every == 0 or it == a.iters:
            m, rows = evaluate(model, val, dev, meta=a.meta, amp=amp, t2=a.t2scale > 0, grids=a.grids)
            m.update(it=it, time=time.time() - t0)
            log.write(json.dumps(m) + "\n"), log.flush()
            print("EVAL", json.dumps({k: round(v, 4) for k, v in m.items()}), flush=True)
            torch.save(model.state_dict(), out / "last.pt")
            if m["score"] > best:
                best, best_it = m["score"], it
                torch.save(model.state_dict(), out / "best.pt")
                with open(out / "val_rows.json", "w") as f:
                    json.dump(rows, f)
            if patience and it - best_it >= patience:
                print(f"early stop at it {it}: no val gain since it {best_it}", flush=True)
                break
        if it % 1000 == 0 or it % a.eval_every == 0:  # frequent: WSL CUDA crashes come every few thousand iterations
            torch.save({"model": model.state_dict(), "opt": opt.state_dict(), "sched": sched.state_dict(),
                        "it": it, "best": best, "best_it": best_it}, out / "resume.pt")
    print("best", best)


if __name__ == "__main__":
    main()
