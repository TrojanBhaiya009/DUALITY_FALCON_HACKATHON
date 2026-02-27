"""
Duality AI — Offroad Segmentation
Main Training Script — 5 epochs, 30 min budget, target 65%+ mAP50
Run: python train.py
"""

import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, RandomSampler
from torch.optim import AdamW
from torch.optim.lr_scheduler import OneCycleLR
import segmentation_models_pytorch as smp
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from tqdm import tqdm
import time
import multiprocessing

from src.dataset    import DesertDataset
from src.transforms import get_train_transforms, get_val_transforms
from src.losses     import CombinedLoss
from src.metrics    import IoUMetric, mAP50Metric

# ── CONFIG ──────────────────────────────────────────────────────────────────
TRAIN_RGB   = "train/Color_Images"
TRAIN_SEG   = "train/Segmentation"
VAL_RGB     = "val/Color_Images"
VAL_SEG     = "val/Segmentation"

EPOCHS      = 5
BATCH_SIZE  = 8
IMG_SIZE    = 256
LR          = 4e-3
NUM_CLASSES = 10
NUM_WORKERS = 0
PATIENCE    = 5
VAL_EVERY   = 1
SAVE_DIR    = "runs"
DEVICE      = "cuda" if torch.cuda.is_available() else "cpu"
USE_AMP     = (DEVICE == "cuda")

# Sub-epoch: 480 images/epoch → 60 iters × ~2.5s = ~2.5 min train on CPU
SAMPLES_PER_EPOCH = 480

if DEVICE == "cuda":
    torch.backends.cudnn.benchmark = True

if DEVICE == "cpu":
    n_threads = multiprocessing.cpu_count()
    torch.set_num_threads(n_threads)
    print(f"CPU threads set to {n_threads}")

print(f"Using device: {DEVICE} | AMP: {USE_AMP}")
print(f"Epochs: {EPOCHS} | Batch: {BATCH_SIZE} | IMG: {IMG_SIZE}px")
os.makedirs(SAVE_DIR, exist_ok=True)

# ── DATA ────────────────────────────────────────────────────────────────────
train_dataset = DesertDataset(TRAIN_RGB, TRAIN_SEG, get_train_transforms(IMG_SIZE))
val_dataset   = DesertDataset(VAL_RGB,   VAL_SEG,   get_val_transforms(IMG_SIZE))

train_sampler = RandomSampler(train_dataset, replacement=False,
                              num_samples=min(SAMPLES_PER_EPOCH, len(train_dataset)))
train_loader  = DataLoader(train_dataset, batch_size=BATCH_SIZE,
                           sampler=train_sampler, num_workers=NUM_WORKERS,
                           pin_memory=False, drop_last=True)
val_loader    = DataLoader(val_dataset, batch_size=BATCH_SIZE,
                           shuffle=False, num_workers=NUM_WORKERS,
                           pin_memory=False)

print(f"Train: {len(train_dataset)} total ({SAMPLES_PER_EPOCH}/epoch) | Val: {len(val_dataset)}")

# ── MODEL ───────────────────────────────────────────────────────────────────
# MobileNetV2 — fastest encoder on CPU, good ImageNet features
model = smp.Unet(
    encoder_name    = "mobilenet_v2",
    encoder_weights = "imagenet",
    in_channels     = 3,
    classes         = NUM_CLASSES,
    decoder_use_batchnorm = True,
)
model = model.to(DEVICE)

total = sum(p.numel() for p in model.parameters())
print(f"Unet (MobileNetV2) — {total/1e6:.1f}M params")

# ── OPTIMIZER + SCHEDULER ───────────────────────────────────────────────────
optimizer = AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
scheduler = OneCycleLR(optimizer, max_lr=LR,
                       steps_per_epoch=len(train_loader),
                       epochs=EPOCHS, pct_start=0.3,
                       div_factor=10, final_div_factor=50)
criterion = CombinedLoss(NUM_CLASSES)
iou_metric  = IoUMetric(NUM_CLASSES)
map_metric  = mAP50Metric(NUM_CLASSES)
scaler      = torch.amp.GradScaler('cuda') if USE_AMP else None

# ── TRAINING LOOP ───────────────────────────────────────────────────────────
best_map50   = 0.0
best_miou    = 0.0
train_losses = []
val_mious    = []
val_maps     = []
no_improve   = 0
total_start  = time.time()

for epoch in range(1, EPOCHS + 1):
    epoch_start = time.time()

    # ── Train ──
    model.train()
    total_loss = 0.0

    loop = tqdm(train_loader, desc=f"Epoch {epoch}/{EPOCHS} [Train]")
    for images, masks in loop:
        images = images.to(DEVICE, non_blocking=True)
        masks  = masks.to(DEVICE, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)

        if USE_AMP:
            with torch.amp.autocast('cuda'):
                outputs = model(images)
                if outputs.shape[-2:] != masks.shape[-2:]:
                    outputs = nn.functional.interpolate(
                        outputs, size=masks.shape[-2:],
                        mode="bilinear", align_corners=False
                    )
                loss = criterion(outputs, masks)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
        else:
            outputs = model(images)
            if outputs.shape[-2:] != masks.shape[-2:]:
                outputs = nn.functional.interpolate(
                    outputs, size=masks.shape[-2:],
                    mode="bilinear", align_corners=False
                )
            loss = criterion(outputs, masks)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

        scheduler.step()
        total_loss += loss.item()
        loop.set_postfix(loss=f"{loss.item():.4f}", lr=f"{scheduler.get_last_lr()[0]:.2e}")

    avg_loss = total_loss / len(train_loader)
    train_losses.append(avg_loss)
    train_time = time.time() - epoch_start

    # ── Validate ──
    model.eval()
    iou_metric.reset()
    map_metric.reset()

    val_start = time.time()
    with torch.no_grad():
        for images, masks in tqdm(val_loader, desc=f"Epoch {epoch}/{EPOCHS} [Val]"):
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

    per_class_iou, miou = iou_metric.compute()
    _, map50 = map_metric.compute()
    val_mious.append(miou)
    val_maps.append(map50)

    val_time = time.time() - val_start
    elapsed = time.time() - total_start

    print(f"\nEpoch {epoch} | Loss: {avg_loss:.4f} | mIoU: {miou*100:.2f}% | mAP50: {map50*100:.2f}%")
    print(f"  Train: {train_time:.0f}s | Val: {val_time:.0f}s | Total elapsed: {elapsed/60:.1f} min\n")

    # Save best model based on mAP50
    if map50 > best_map50:
        best_map50 = map50
        best_miou = miou
        no_improve = 0
        torch.save(model.state_dict(), os.path.join(SAVE_DIR, "best_model.pth"))
        print(f"  ✅ Best model saved! mAP50={map50*100:.2f}% mIoU={miou*100:.2f}%\n")
    else:
        no_improve += 1
        if no_improve >= PATIENCE:
            print(f"\n⏹ Early stopping at epoch {epoch}")
            break

total_time = time.time() - total_start
print(f"\n{'='*50}")
print(f"Training complete in {total_time/60:.1f} min")
print(f"Best mAP50: {best_map50*100:.2f}%  |  Best mIoU: {best_miou*100:.2f}%")
print(f"{'='*50}")

# ── SAVE GRAPHS ─────────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(18, 5))

axes[0].plot(train_losses, color="tomato", linewidth=2, marker='o')
axes[0].set_title("Training Loss", fontsize=14)
axes[0].set_xlabel("Epoch"); axes[0].set_ylabel("Loss"); axes[0].grid(True)

axes[1].plot([v * 100 for v in val_mious], color="steelblue", linewidth=2, marker='o')
axes[1].set_title("Validation mIoU (%)", fontsize=14)
axes[1].set_xlabel("Epoch"); axes[1].set_ylabel("mIoU (%)"); axes[1].grid(True)

axes[2].plot([v * 100 for v in val_maps], color="forestgreen", linewidth=2, marker='o')
axes[2].set_title("Validation mAP@50 (%)", fontsize=14)
axes[2].axhline(y=65, color="red", linestyle="--", label="Target 65%")
axes[2].set_xlabel("Epoch"); axes[2].set_ylabel("mAP@50 (%)"); axes[2].legend(); axes[2].grid(True)

plt.tight_layout()
plt.savefig(os.path.join(SAVE_DIR, "training_graphs.png"), dpi=150)
print(f"Graphs saved → runs/training_graphs.png")
