import numpy as np

CLASS_NAMES = [
    "Trees", "Lush Bushes", "Dry Grass", "Dry Bushes",
    "Ground Clutter", "Flowers", "Logs", "Rocks",
    "Landscape", "Sky"
]

class IoUMetric:
    def __init__(self, num_classes=10):
        self.num_classes = num_classes
        self.reset()

    def reset(self):
        self.cm = np.zeros((self.num_classes, self.num_classes), dtype=np.int64)

    def update(self, pred_tensor, target_tensor):
        pred   = pred_tensor.argmax(dim=1).cpu().numpy().flatten()
        target = target_tensor.cpu().numpy().flatten()
        valid  = (target >= 0) & (target < self.num_classes)
        combo  = self.num_classes * target[valid].astype(np.int64) + pred[valid]
        self.cm += np.bincount(combo, minlength=self.num_classes**2).reshape(
            self.num_classes, self.num_classes
        )

    def compute(self):
        tp   = np.diag(self.cm)
        fp   = self.cm.sum(axis=0) - tp
        fn   = self.cm.sum(axis=1) - tp
        iou  = tp / (tp + fp + fn + 1e-10)
        miou = iou.mean()

        print("\n── Per-Class IoU ──────────────────────────")
        for name, val in zip(CLASS_NAMES, iou):
            bar = "█" * int(val * 25)
            print(f"  {name:<16} {val:.4f}  {bar}")
        print(f"\n  mIoU : {miou:.4f}  ({miou*100:.2f}%)")
        print("────────────────────────────────────────────")
        return iou, miou


class mAP50Metric:
    """mAP@IoU=0.5 for semantic segmentation.
    For each image, for each GT class: compute IoU with prediction.
    If IoU >= 0.5, count as TP. AP per class = TP / total.
    mAP50 = mean(AP per class)."""
    def __init__(self, num_classes=10, iou_threshold=0.5):
        self.num_classes = num_classes
        self.iou_threshold = iou_threshold
        self.reset()

    def reset(self):
        self.class_tp = np.zeros(self.num_classes, dtype=np.int64)
        self.class_total = np.zeros(self.num_classes, dtype=np.int64)

    def update(self, pred_tensor, target_tensor):
        pred = pred_tensor.argmax(dim=1).cpu().numpy()
        target = target_tensor.cpu().numpy()
        batch_size = pred.shape[0]
        for b in range(batch_size):
            t = target[b]
            p = pred[b]
            for c in range(self.num_classes):
                gt_pixels = np.count_nonzero(t == c)
                if gt_pixels == 0:
                    continue
                self.class_total[c] += 1
                pred_pixels = np.count_nonzero(p == c)
                intersection = np.count_nonzero((t == c) & (p == c))
                union = gt_pixels + pred_pixels - intersection
                if union > 0 and (intersection / union) >= self.iou_threshold:
                    self.class_tp[c] += 1

    def compute(self):
        ap_per_class = np.zeros(self.num_classes)
        valid = 0
        print("\n── Per-Class AP@50 ────────────────────────")
        for c in range(self.num_classes):
            if self.class_total[c] > 0:
                ap_per_class[c] = self.class_tp[c] / self.class_total[c]
                valid += 1
            bar = "█" * int(ap_per_class[c] * 25)
            print(f"  {CLASS_NAMES[c]:<16} {ap_per_class[c]:.4f}  {bar}")
        map50 = ap_per_class.sum() / max(valid, 1)
        print(f"\n  mAP@50 : {map50:.4f}  ({map50*100:.2f}%)")
        print("────────────────────────────────────────────")
        return ap_per_class, map50
