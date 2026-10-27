"""EURS segmentation score, as Nikhil (EURS co-lead) spelled it out on Discord, 16 Oct 2026 (both tracks):
  - pixel TP/FP/FN summed over the whole test set (one confusion matrix), not per image
  - mIoU and F1 = mean per-class IoU / F1 over the classes present in the GT of that set
  - coverage_penalty = min(1, worst present-class IoU / 0.02)
  - Qseg = F1 x mIoU x coverage_penalty
The per-image score (q_img in seg/train_seg.py) is only a stress test; the dataset-level one (`q_ds`) is the LB number.
"""
import numpy as np

N_CLASSES = 5
NAMES = ["bg", "coral", "fish", "elasmo", "other"]
IOU_FLOOR = 0.02


def confusion(pred, gt, n=N_CLASSES):
    """n x n pixel counts, rows = GT, cols = prediction."""
    return np.bincount(gt.ravel().astype(np.int64) * n + pred.ravel(), minlength=n * n).reshape(n, n)


def from_confusion(cm):
    """(q, miou, f1, coverage penalty, per-class IoU) from one confusion matrix. The IoU vector is NaN only for classes
    absent from both GT and prediction (a predicted-only class shows its 0, though it is not scored)."""
    tp = np.diag(cm).astype(float)
    g, p = cm.sum(1), cm.sum(0)
    used = (g + p) > 0
    iou = np.full(len(tp), np.nan)
    iou[used] = tp[used] / (g + p - tp)[used]
    present = g > 0
    f1 = (2 * tp[present] / (g + p)[present]).mean()
    miou = iou[present].mean()
    penalty = min(1.0, iou[present].min() / IOU_FLOOR)
    return f1 * miou * penalty, miou, f1, penalty, iou


def qseg(pred, gt):
    return from_confusion(confusion(pred, gt))[0]


if __name__ == "__main__":
    gt = np.zeros((10, 10), np.uint8)
    gt[:5] = 1
    gt[0, 0] = 3  # 1 elasmo pixel
    assert qseg(gt, gt) == 1.0
    miss = gt.copy()
    miss[0, 0] = 1
    assert qseg(miss, gt) == 0.0  # ignoring a present class zeroes the score
    half = gt.copy()
    half[5:7] = 1  # 20 bg pixels called coral
    q, miou, f1, cov, iou = from_confusion(confusion(half, gt))
    assert cov == 1 and 0 < q < 1 and np.isnan(iou[2]) and abs(iou[0] - 30 / 50) < 1e-9
    stray = gt.copy()
    stray[9, 9] = 4  # predicted "other" absent from GT: IoU 0 is shown but not scored
    q, miou, f1, cov, iou = from_confusion(confusion(stray, gt))
    assert iou[4] == 0 and abs(miou - np.nanmean(iou[:4])) < 1e-9
    big = np.zeros((10, 2000), np.uint8)
    big[:, 0] = 3  # 10 elasmo pixels
    weak = big.copy()
    weak[:, 1:100] = 3  # 990 bg pixels called elasmo: elasmo IoU 0.01 -> penalty 0.5
    assert abs(from_confusion(confusion(weak, big))[3] - 0.5) < 1e-9
    print("qseg self-check OK")
