"""Track 1 restorer B, run on the 1024 cache:
water types 1-4 = the per-(type, level) 3x4 affine colour table (cells_all.json, fitted on all 20 scenes); types 5-10 =
table j0 (cells_fold.json) + Restormer residual on [x, j0], hflip-averaged. Writes <out>/<stem>.png.

    .venv/bin/python scripts/vk_restore.py --split test --out work/vk_rest_test
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

ROOT = Path(__file__).resolve().parent.parent
sys.path[:0] = [str(ROOT / "restoration"), str(ROOT / "external/restormer")]
from eurs import frame_meta, imread, split_items  # noqa: E402
from restormer_arch import Restormer  # noqa: E402


def load_cells(p):
    return {k: np.array(v, np.float32) for k, v in json.loads(Path(p).read_text()).items()}


def table(x01, cells, t, l):
    A = cells[f"{t}_{l}"]
    return np.clip(x01 @ A[:3] + A[3], 0, 1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="test")
    ap.add_argument("--out", required=True)
    ap.add_argument("--ckpt", default="work/vk/restormer_phys_stage5_it1500.pt")
    ap.add_argument("--cells_rest", default="work/vk/cells_fold.json", help="table the Restormer was trained with")
    ap.add_argument("--cells_table", default="work/vk/cells_all.json", help="table used alone for water types 1-4")
    a = ap.parse_args()
    out = ROOT / a.out
    out.mkdir(parents=True, exist_ok=True)
    CR, CT = load_cells(ROOT / a.cells_rest), load_cells(ROOT / a.cells_table)
    net = Restormer(inp_channels=6, out_channels=3, dim=48, num_blocks=[4, 6, 6, 8], num_refinement_blocks=4,
                    heads=[1, 2, 4, 8], ffn_expansion_factor=2.66, bias=False, LayerNorm_type="BiasFree",
                    dual_pixel_task=True)
    net.load_state_dict(torch.load(ROOT / a.ckpt, map_location="cpu", weights_only=True))
    net = net.cuda().eval()
    items = list(split_items(a.split))
    with torch.no_grad():
        for i, (stem, p) in enumerate(items):
            if (out / f"{stem}.png").exists():
                continue
            t, l = frame_meta(stem)
            x01 = imread(p).astype(np.float32) / 255
            if t <= 4:
                r = table(x01, CT, t, l)
            else:
                x6 = torch.from_numpy(np.concatenate([x01, table(x01, CR, t, l)], 2)).permute(2, 0, 1)[None].cuda()
                h, w = x6.shape[2:]
                xp = F.pad(x6, (0, (8 - w % 8) % 8, 0, (8 - h % 8) % 8), mode="reflect")
                y = 0.5 * ((xp[:, 3:] + net(xp)) + (xp.flip(-1)[:, 3:] + net(xp.flip(-1))).flip(-1))
                r = y[..., :h, :w].clamp(0, 1)[0].permute(1, 2, 0).cpu().numpy()
            Image.fromarray((r * 255 + 0.5).astype(np.uint8)).save(out / f"{stem}.png")
            if i % 100 == 0:
                print(i, len(items), flush=True)
    print("wrote", len(list(out.glob("*.png"))), "to", out, flush=True)


if __name__ == "__main__":
    main()
