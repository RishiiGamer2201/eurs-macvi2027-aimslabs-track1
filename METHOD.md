# Method description

Team: aimslabs   Track(s): 1 (this repository; Track 2 is in eurs-macvi2027-aimslabs-track2)
Reported testing-phase submission(s): Track 1 #19381 (aimslabs_t1_final, user #1995, composite 0.739; re-upload of our entry t1_test_102)

## Approach

Every degraded image comes from one of 200 (water type, level) cells of the RSUIGM generator. Within a cell the colour
cast is close to a fixed per-channel affine map. We read the cell from the trailing `_<type>_<level>` of the file name
(organizer rulings, 6 and 27 Oct). All models run on the input resized to longest side 1024 (cv2 INTER_AREA), which is
also the scoring size. Restoration and segmentation are separate. Both see only the degraded image and its file name.

**Restoration.** The output is the pixel mean of two Restormer models (Zamir et al., CVPR 2022), each taking
`[x, j0]` (6 channels) and predicting a residual on `j0`. Each restorer's output is rounded to uint8 first
(A: `round(255*y)`, B: `uint8(255*y + 0.5)`), then their mean is rounded to uint8.
- (A) 26.25 M parameters. `j0` comes from the physics front end, using a per-cell affine colour table fitted on
  the 20 training scenes, then a learned 1/8-resolution gain/offset veil map. Horizontal-flip averaging, bf16.
- (B) 26.12 M parameters. `j0` comes from a 3×4 ridge-fitted colour table per cell. For water types 1–4 the table output is
  used as is; for types 5–10 the network refines it. Horizontal-flip averaging, fp32.

Restored images are written as JPEG quality 97 with 4:4:4 chroma, exactly as in our leaderboard zip.

**Segmentation.**
1. Mean softmax of five ViT-L segmenters fine-tuned on EURS:
   - DINOv3-L/16 with a simple head
   - DINOv3-L with a Mask2Former decoder on a ViTDet pyramid
   - DINOv3-L with an FPN/UPerNet head
   - DINOv2-L/14-reg4 with a simple head
   - DINOv3-L with a simple head, trained with Coralscapes crops (30 % of its non-dev crops, 22.5 % of all)
2. The frozen EPFL CoralscapesV2 DINOv3-L LoRA-DPT re-splits the background + coral probability (weight 0.55). Thing
   classes stay with the ensemble.
3. A ViT-B DPT (initialised from the EPFL ViT-B model, fine-tuned on EURS) is mixed in with weight 0.5. Its input is
   reflect-padded to a multiple of 16, standardised per channel with that padded image's own mean and standard
   deviation (clamped below at 1/255), clipped to [-4, 4], mapped by 0.225 z + 0.45, then ImageNet-normalised.
4. Probabilities are quantised to uint8, fish is scaled by 1.5, and the argmax is taken.
5. SAM 3 (Meta), prompted with animal names on each image alone (instance threshold 0.3, mask threshold 0.5,
   score-weighted masks combined by per-class maximum), applies two rules:
   - Elasmobranch pixels where SAM 3 sees no animal (map < 0.2) take the nearest other label. This is skipped if it
     would remove more than half of them.
   - In images where the ensemble already predicts fish, background or coral pixels where SAM 3's fish map is > 0.5
     become fish.

## Training data

- **EURS:** all 20 training scenes for restorer (A) and the five ViT-L segmenters. The ViT-B DPT used 16 of them
  (fish4, fish12, fish21, seahorse1 held out); restorer (B) used the same 16 in stages 1-2 and all 20 in stages 3-5
  (its network-input table `cells_fold.json` is fitted on the 16, the table-only `cells_all.json` on all 20).
  Architectures and recipes were compared by 5-fold scene-grouped cross-validation.
- **EURS dev:** the 6 development scenes (inputs only, no GT) were used, as allowed when disclosed (organizer ruling,
  17 Oct):
  - in the last stage of restorer (A), with pseudo-clean targets: all frames of a scene share one clean image, so a
    dev scene's target is the mean of the earlier restorer (A)'s outputs on its frames with water types 1-3 and
    levels up to 50
  - as 25 % of the crops of the five ViT-L segmenters, labelled by the argmax of our earlier ensemble's softmax
    averaged over the frames of the same dev scene

  This cross-frame aggregation only builds dev training targets; it is never used at test time. Restorer (B) and
  the ViT-B DPT did not use dev images.
- **Test images:** no testing-phase pixels were used to fit weights, targets, dataset statistics or image-specific
  settings; per-image statistics are computed only by the fixed inference pipeline, and none are aggregated across
  test images. We used the aggregate testing-phase leaderboard scores of our own submissions to select the ensemble
  members, EPFL weight 0.55, fish scale 1.5 and SAM 3 grow threshold 0.5. This is model selection using
  testing-phase evaluation feedback; all selected settings are fixed for every image.
- **MaCVi 2025 MVRC:** no data used.

External data and pretrained models (all public; licence texts are in the weights archive, `LICENSES/`):

| What | Link | Licence | Use |
|---|---|---|---|
| LSUI | https://github.com/LintaoPeng/U-shape_Transformer_for_Underwater_Image_Enhancement | no explicit licence (paper cited) | raw photos re-degraded with our per-cell colour maps, restorer (A) training |
| Restormer real-denoising weights | https://github.com/swz30/Restormer | MIT | init of both restorers |
| DINOv3 ViT-L/16 LVD-1689M | https://huggingface.co/timm/vit_large_patch16_dinov3.lvd1689m | DINOv3 License | backbone of 4 segmenters |
| DINOv2 ViT-L/14 reg4 LVD-142M | https://huggingface.co/timm/vit_large_patch14_reg4_dinov2.lvd142m | Apache-2.0 | backbone of 1 segmenter |
| Mask2Former Swin-L ADE20K semantic | https://huggingface.co/facebook/mask2former-swin-large-ade-semantic | MIT | decoder init of 1 segmenter |
| AquaOV255 | https://huggingface.co/datasets/kkk2026/UOVSBench | dataset card: Apache-2.0; article: CC BY 4.0 | rare-class cut-outs copy-pasted into the ViT-L segmenters' EURS training crops |
| Coralscapes | https://huggingface.co/datasets/EPFL-ECEO/coralscapes | Apache-2.0 | masks remapped to EURS classes, 22.5 % of one segmenter's crops |
| EPFL CoralscapesV2 DINOv3-L LoRA-DPT | https://huggingface.co/EPFL-ECEO/coralscapesv2-dinov3-vitl-lora-dpt | Apache-2.0 (+ DINOv3 License) | frozen, inference only |
| EPFL CoralscapesV2 DINOv3-B LoRA-DPT | https://huggingface.co/EPFL-ECEO/coralscapesv2-dinov3-vitb-lora-dpt | Apache-2.0 (+ DINOv3 License) | init of the ViT-B DPT segmenter |
| CoralscapesV2, USIS16K | https://huggingface.co/datasets/josauder/coralscapesV2, https://github.com/LinHong-HIT/USIS16K | Apache-2.0; academic non-commercial | ViT-B DPT segmenter training |
| SAM 3 | https://huggingface.co/facebook/sam3 | **SAM License** (text: `LICENSES/SAM_LICENSE.txt` in the weights archive) | frozen, inference only. The archive carries an unchanged copy under the SAM License |

Thumbnail-correlation checks against the 26 known EURS scenes found no near-duplicates in the LSUI, AquaOV255,
Coralscapes and CoralscapesV2 pools. AquaOV255 contains 16,000 of USIS16K's 16,151 photos; no separate record was
kept for the rest of the USIS16K subset used by the ViT-B DPT.

## Anything non-standard at inference

- **Restoration:**
  - resize to longest side 1024 (INTER_AREA)
  - the (type, level) cell is read from the file name
  - hflip averaging in both restorers
  - each restorer rounded to uint8, then their 50/50 average rounded to uint8
  - JPEG q97 4:4:4 output
- **Segmentation:**
  - single scale, no flip; fixed per-image normalisation inside the ViT-B DPT (above)
  - ensemble of 7 models
  - uint8 quantisation of the mean probabilities, then fish ×1.5 before argmax
  - SAM 3 trim and grow, as above

Each output depends only on its own input image and the trailing `_<type>_<level>` of its name.

## Reproduction notes

- Build the image with `docker build -t aimslabs-t1 .` (base `pytorch/pytorch:2.5.1-cuda12.4-cudnn9-runtime`, plus the
  pinned `requirements.txt`; organizer ruling, 27 Oct). It runs offline (`HF_HUB_OFFLINE=1`).
- `repro_check.py` warns that `requirements.txt` repeats base-image packages. That warning assumes the old torch 2.4.0
  image; our versions are meant for the 2.5.1 image named in the `Dockerfile`, and pip keeps its torch 2.5.1 (checked in
  a Python 3.11 + torch 2.5.1 environment, where `run.py` also runs end to end).
- Weights: unpack the archive from `weights/weights_link.yaml` into `weights/` (mounted read-only at `/pkg/weights`).
  The archive is about 11.6 GB.
- Run:

  ```
  python run.py --input <dir> --output <dir>
  ```

  or call `main(model_dir, input_dir, output_dir, device)`.
- The stages run one model group at a time over all images. Intermediates go to `output_dir/_tmp`, which is deleted
  at the end.
- **Measured on an RTX 5070 Ti 16 GB** (1,200 test images), from the run log:
  - 78 min in total, 3.9 s per image
  - per image: decode + resize 0.31 s (inputs up to tens of megapixels), restorer A 1.14 s, restorer B 0.97 s, JPEG
    writing 0.04 s, segmentation ensemble 0.57 s, SAM 3 + rules 0.89 s (model loading included)
  - GPU memory under 16 GB; peak process RSS about 6.5 GiB
  - we expect about 4–5 s per image on an RTX 3090 (24 GB)
- **Verified:** on our GPU (torch 2.11 / CUDA 12.8), run.py reproduces the outputs of our leaderboard zip t1_test_102
  on all 1,200 test images: all 1,200 restored JPEGs are byte-identical and all 1,200 decoded mask arrays identical.
- On torch 2.5.1 / CUDA 12.4, bf16 kernels may differ in the last bit, so a few JPEG values or mask pixels can change.
  We expect agreement within normal hardware-dependent numerical variation.
