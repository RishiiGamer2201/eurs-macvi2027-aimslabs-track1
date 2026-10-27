"""Track 1 inference, team aimslabs (testing-phase entry t1_test_102). The organizers call main(model_dir, input_dir,
output_dir, device). Each output comes from its own input image and the trailing _<type>_<level> of its file name
(organizer rulings 6 and 27 Oct; the scene part of the name is never read). Same steps as our leaderboard pipeline:

 1. cache: the input resized to longest side 1024 with cv2 INTER_AREA (our scripts/prepare_cache.py); all models run on it
 2. restorer A: PhysNet j0 from the true (type, level) cell + Restormer residual, hflip average, bf16
 3. restorer B: per-cell colour table; + Restormer residual on [x, j0] for water types 5-10, hflip, fp32
 4. restored = round(0.5 A + 0.5 B), written as JPEG quality 97, 4:4:4 (as in our leaderboard zip)
 5. masks: mean softmax of 5 fine-tuned ViT-L segmenters, bg/coral re-split by the EPFL CoralscapesV2 model (w 0.55),
    fine-tuned ViT-B DPT mixed in (w 0.5), probabilities quantised to uint8, fish x1.5, argmax
 6. SAM 3 text-prompt maps of the image: elasmobranch trim (tau 0.2, guard 0.5), then fish grow (map > 0.5)

Stages run one model group at a time over all images (fits one 24 GB GPU and 32 GB RAM); intermediates go to
output_dir/_tmp, which is deleted at the end. Deterministic: model.eval(), no test-time randomness.
"""
import os
import shutil
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE / "src"))
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

import cv2  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
from PIL import Image  # noqa: E402

Image.MAX_IMAGE_PIXELS = None
MAX_SIDE = 1024
IMG_EXTS = (".png", ".jpg", ".jpeg")
SEG_RUNS = ["seg_all_ftpdev", "seg_all_m2fdinodev", "seg_all_fpndev", "seg_all_dinov2dev", "seg_all_cs30dev"]
FISH_SCALE = np.array([1, 1, 1.5, 1, 1], np.float32).reshape(5, 1, 1)
SAM3_PROMPTS = {2: ["fish"], 3: ["shark", "stingray"], 4: ["sea turtle", "starfish", "seahorse"]}


def target_size(w, h):
    """The organizer resize_for_submission.py formula (also our cache's)."""
    s = MAX_SIDE / max(w, h)
    return (w, h) if s >= 1 else (max(1, round(w * s)), max(1, round(h * s)))


def log(t0, msg):
    print(f"[{time.time() - t0:7.0f} s] {msg}", flush=True)


def main(model_dir, input_dir, output_dir, device):
    W, inp, out, dev = Path(model_dir), Path(input_dir), Path(output_dir), torch.device(device)
    os.environ["HF_HOME"] = str(out / "_tmp/hf")  # writable (weights/ is read-only); nothing is downloaded; before src imports
    import train_seg
    from eurs import cell, frame_meta, imread, load_model, to_tensor
    from train import FlipTTA, run_full
    from vk_restore import Restormer, load_cells, table
    from sam3_apply import grow, trim

    torch.set_grad_enabled(False)
    t0 = time.time()
    tmp = out / "_tmp"
    for d in ("restored", "seg_masks"):
        (out / d).mkdir(parents=True, exist_ok=True)
    files = sorted((p for p in inp.rglob("*") if p.suffix.lower() in IMG_EXTS), key=lambda p: p.name)
    meta = {p.stem: frame_meta(p.stem) for p in files}
    bad = [s for s, m in meta.items() if m is None]
    assert not bad, f"file names must end in _<type>_<level> (as released), e.g. {bad[:3]}"

    # 1. cache, exactly as scripts/prepare_cache.py (cv2 INTER_AREA, PNG), read back as RGB with eurs.imread
    (tmp / "cache").mkdir(parents=True, exist_ok=True)
    for p in files:
        img = cv2.imread(str(p), cv2.IMREAD_UNCHANGED)
        h, w = img.shape[:2]
        cv2.imwrite(str(tmp / "cache" / f"{p.stem}.png"), cv2.resize(img, target_size(w, h), interpolation=cv2.INTER_AREA),
                    [cv2.IMWRITE_PNG_COMPRESSION, 1])
    items = [(p.stem, tmp / "cache" / f"{p.stem}.png") for p in files]
    log(t0, f"cache: {len(items)} images")

    # 2. restorer A (restoration/make_submission.py --model restormer --tta --meta)
    (tmp / "A").mkdir(exist_ok=True)
    model = FlipTTA(load_model("restormer", W / "restorer_a/meta_all_dev.pt")).to(dev).eval()
    for stem, p in items:
        x = to_tensor(imread(p))[None].to(dev)
        c = torch.tensor([cell(*meta[stem])]).to(dev)
        with torch.autocast(dev.type, torch.bfloat16):
            y, _, _ = run_full(model, x, cell=c)
        y = (y[0].float().clamp(0, 1).permute(1, 2, 0).cpu().numpy() * 255).round().astype(np.uint8)
        Image.fromarray(y).save(tmp / "A" / f"{stem}.png")
    del model
    log(t0, "restorer A done")

    # 3. restorer B (src/vk_restore.py: per-cell colour tables + Restormer)
    (tmp / "B").mkdir(exist_ok=True)
    CR, CT = load_cells(W / "restorer_b/cells_fold.json"), load_cells(W / "restorer_b/cells_all.json")
    net = Restormer(inp_channels=6, out_channels=3, dim=48, num_blocks=[4, 6, 6, 8], num_refinement_blocks=4,
                    heads=[1, 2, 4, 8], ffn_expansion_factor=2.66, bias=False, LayerNorm_type="BiasFree",
                    dual_pixel_task=True)
    net.load_state_dict(torch.load(W / "restorer_b/restormer_phys_stage5_it1500.pt", map_location="cpu", weights_only=True))
    net = net.to(dev).eval()
    F = torch.nn.functional
    for stem, p in items:
        t, l = meta[stem]
        x01 = imread(p).astype(np.float32) / 255
        if t <= 4:
            r = table(x01, CT, t, l)
        else:
            x6 = torch.from_numpy(np.concatenate([x01, table(x01, CR, t, l)], 2)).permute(2, 0, 1)[None].to(dev)
            h, w = x6.shape[2:]
            xp = F.pad(x6, (0, (8 - w % 8) % 8, 0, (8 - h % 8) % 8), mode="reflect")
            y = 0.5 * ((xp[:, 3:] + net(xp)) + (xp.flip(-1)[:, 3:] + net(xp.flip(-1))).flip(-1))
            r = y[..., :h, :w].clamp(0, 1)[0].permute(1, 2, 0).cpu().numpy()
        Image.fromarray((r * 255 + 0.5).astype(np.uint8)).save(tmp / "B" / f"{stem}.png")
    del net
    log(t0, "restorer B done")

    # 4. 50/50 average -> organizer size (already the cache size) -> JPEG q97 4:4:4 (restoration/make_submission.py)
    for (stem, p), src in zip(items, files):
        a, b = (cv2.imread(str(tmp / k / f"{stem}.png")).astype(np.float32) for k in ("A", "B"))
        avg = np.round(0.5 * a + 0.5 * b).astype(np.uint8)
        with Image.open(src) as im:
            assert target_size(*im.size) == (avg.shape[1], avg.shape[0]), (stem, im.size, avg.shape)
        Image.fromarray(cv2.cvtColor(avg, cv2.COLOR_BGR2RGB)).save(out / "restored" / f"{stem}.jpg", quality=97,
                                                                   subsampling=0)
    log(t0, "restored images written")

    # 5. segmentation ensemble (scripts/predict_seg_tta.py --save_probs, then scripts/probs_to_masks.py --scale 2:1.5)
    train_seg.PRETRAINED = False
    train_seg.M2FSeg.SRC = str(W / "mask2former_config")
    train_seg.EpflBgCoral.WEIGHTS = W / "epfl_coralscapesv2_dinov3_vitl_dpt/model.safetensors"
    specs = [str(W / "seg" / r / "last.pt") for r in SEG_RUNS] + ["epfl:0.55", f"vk:0.5:{W / 'seg_vk/coralseg_it4000.pt'}"]
    ens = train_seg.Ensemble([train_seg.load_seg(s, dev) for s in specs]).eval()
    (tmp / "seg").mkdir(exist_ok=True)
    for stem, p in items:
        prob = train_seg.predict(ens, train_seg.to_x(imread(p), dev))
        prob = prob / prob.sum(1, keepdim=True)
        q = (prob[0] * 255).round().byte().cpu().numpy().astype(np.float32)
        cv2.imwrite(str(tmp / "seg" / f"{stem}.png"), (q * FISH_SCALE).argmax(0).astype(np.uint8))
    del ens
    torch.cuda.empty_cache() if dev.type == "cuda" else None
    log(t0, "segmentation ensemble done")

    # 6. SAM 3 maps (scripts/sam3_maps.py) + rules (scripts/sam3_apply.py --grow 2:0.5:0, default trim)
    from transformers import Sam3Model, Sam3Processor
    proc = Sam3Processor.from_pretrained(W / "sam3")
    sam = Sam3Model.from_pretrained(W / "sam3", torch_dtype=torch.bfloat16).to(dev).eval()
    texts = [t for c in SAM3_PROMPTS for t in SAM3_PROMPTS[c]]
    for stem, p in items:
        im = imread(p)
        inp3 = proc(images=[Image.fromarray(im)] * len(texts), text=texts, return_tensors="pt").to(dev)
        inp3["pixel_values"] = inp3["pixel_values"].to(torch.bfloat16)
        res = proc.post_process_instance_segmentation(sam(**inp3), threshold=0.3, mask_threshold=0.5,
                                                      target_sizes=[im.shape[:2]] * len(texts))
        sm, k = {}, 0
        for c, ts in SAM3_PROMPTS.items():
            m = torch.zeros(im.shape[:2], device=dev)
            for _ in ts:
                for mk, s in zip(res[k]["masks"], res[k]["scores"]):
                    m = torch.maximum(m, mk.float() * float(s))
                k += 1
            sm[c] = (m * 255).round().byte().cpu().numpy().astype(np.float32) / 255  # stored as uint8 in our pipeline
        m0 = cv2.imread(str(tmp / "seg" / f"{stem}.png"), cv2.IMREAD_UNCHANGED)
        m = grow(trim(m0, np.maximum.reduce(list(sm.values())), 0.2, 0.5), sm[2], 2, 0.5, 0)
        cv2.imwrite(str(out / "seg_masks" / f"{stem}.png"), m)
    log(t0, "masks written")
    shutil.rmtree(tmp)


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", default=str(HERE / "weights"))
    ap.add_argument("--input", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--device", default="cuda")
    a = ap.parse_args()
    main(a.weights, a.input, a.output, a.device)
