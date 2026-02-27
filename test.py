"""
Duality AI — Offroad Segmentation
Test Script — unseen images pe predict karo
Run: python test.py
"""

import os
import torch
import torch.nn as nn
import numpy as np
from PIL import Image
import segmentation_models_pytorch as smp
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.dataset    import DesertDataset, TestDataset
from src.transforms import get_test_transforms, get_val_transforms
from src.metrics    import IoUMetric, mAP50Metric

# ── CONFIG ──────────────────────────────────────────────────────────────────
TEST_RGB    = "train/Color_Images"
TEST_SEG    = "train/Segmentation"
MODEL_PATH  = "runs/best_model.pth"
OUTPUT_DIR  = "runs/predictions"
IMG_SIZE    = 256
NUM_CLASSES = 10
DEVICE      = "cuda" if torch.cuda.is_available() else "cpu"
USE_AMP     = (DEVICE == "cuda")

# ── Color map for visualization ──────────────────────────────────────────────
# Each class gets a unique color
COLOR_MAP = np.array([
    [34,  139, 34 ],  # 0 Trees         — Forest Green
    [0,   200, 100],  # 1 Lush Bushes   — Lime Green
    [210, 180, 140],  # 2 Dry Grass     — Tan
    [139, 90,  43 ],  # 3 Dry Bushes    — Brown
    [169, 169, 169],  # 4 Ground Clutter— Grey
    [255, 215, 0  ],  # 5 Flowers       — Gold
    [101, 67,  33 ],  # 6 Logs          — Dark Brown
    [128, 128, 128],  # 7 Rocks         — Stone Grey
    [194, 178, 128],  # 8 Landscape     — Sand
    [135, 206, 235],  # 9 Sky           — Sky Blue
], dtype=np.uint8)

os.makedirs(OUTPUT_DIR, exist_ok=True)

# ── MODEL ───────────────────────────────────────────────────────────────────
model = smp.Unet(
    encoder_name    = "mobilenet_v2",
    encoder_weights = None,
    in_channels     = 3,
    classes         = NUM_CLASSES,
    decoder_use_batchnorm = True,
)
model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
model = model.to(DEVICE)
model.eval()
print(f"Model loaded from: {MODEL_PATH}")

# ── EVALUATION (batched for speed) ────────────────────────────────────────
eval_dataset = DesertDataset(TEST_RGB, TEST_SEG, get_val_transforms(IMG_SIZE))
eval_loader  = DataLoader(eval_dataset, batch_size=4, shuffle=False, num_workers=0)

iou_metric = IoUMetric(NUM_CLASSES)
map_metric = mAP50Metric(NUM_CLASSES)

print(f"Evaluating on {len(eval_dataset)} images...")
with torch.no_grad():
    for images, masks in tqdm(eval_loader, desc="Evaluating"):
        images = images.to(DEVICE, non_blocking=True)
        masks  = masks.to(DEVICE, non_blocking=True)
        if USE_AMP:
            with torch.amp.autocast('cuda'):
                outputs = model(images)
        else:
            outputs = model(images)
        if outputs.shape[-2:] != masks.shape[-2:]:
            outputs = nn.functional.interpolate(
                outputs, size=masks.shape[-2:],
                mode="bilinear", align_corners=False
            )
        iou_metric.update(outputs, masks)
        map_metric.update(outputs, masks)

_, miou = iou_metric.compute()
_, map50 = map_metric.compute()

# ── SAVE PREDICTIONS (subset for visualization) ───────────────────────
test_dataset = TestDataset(TEST_RGB, get_test_transforms(IMG_SIZE))
MAX_SAVE = min(50, len(test_dataset))
print(f"\nSaving {MAX_SAVE} prediction images...")

with torch.no_grad():
    for i in tqdm(range(MAX_SAVE), desc="Saving predictions"):
        img_tensor, fname = test_dataset[i]
        img_tensor = img_tensor.unsqueeze(0).to(DEVICE)

        if USE_AMP:
            with torch.amp.autocast('cuda'):
                output = model(img_tensor)
        else:
            output = model(img_tensor)
        if output.shape[-2:] != img_tensor.shape[-2:]:
            output = nn.functional.interpolate(
                output, size=img_tensor.shape[-2:],
                mode="bilinear", align_corners=False
            )

        pred_mask = output.argmax(dim=1).squeeze().cpu().numpy()
        color_mask = COLOR_MAP[pred_mask]
        out_img = Image.fromarray(color_mask)

        save_name = os.path.splitext(fname)[0] + "_pred.png"
        out_img.save(os.path.join(OUTPUT_DIR, save_name))

print(f"\n✅ Evaluation complete!")
print(f"  mIoU:   {miou*100:.2f}%")
print(f"  mAP@50: {map50*100:.2f}%")
print(f"  Predictions saved → {OUTPUT_DIR}/")
