"""
Duality AI — Segmentation Visualizer
RGB image + Predicted mask side by side dikhata hai
Run: python visualize.py
"""

import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from PIL import Image

# ── CONFIG ──────────────────────────────────────────────────────────────────
RGB_DIR  = "train/Color_Images"
PRED_DIR = "runs/predictions"
OUT_DIR  = "runs/visualizations"
NUM_SHOW = 5   # kitni images dikhani hain

os.makedirs(OUT_DIR, exist_ok=True)

CLASS_NAMES = [
    "Trees", "Lush Bushes", "Dry Grass", "Dry Bushes",
    "Ground Clutter", "Flowers", "Logs", "Rocks",
    "Landscape", "Sky"
]

COLOR_MAP = np.array([
    [34,  139, 34 ], [0,   200, 100], [210, 180, 140],
    [139, 90,  43 ], [169, 169, 169], [255, 215, 0  ],
    [101, 67,  33 ], [128, 128, 128], [194, 178, 128],
    [135, 206, 235],
], dtype=np.uint8)

# ── LEGEND ──────────────────────────────────────────────────────────────────
def make_legend():
    patches = []
    for i, name in enumerate(CLASS_NAMES):
        color = COLOR_MAP[i] / 255.0
        patches.append(mpatches.Patch(color=color, label=name))
    return patches

# ── VISUALIZE ───────────────────────────────────────────────────────────────
rgb_files  = sorted(os.listdir(RGB_DIR))[:NUM_SHOW]
pred_files = [os.path.splitext(f)[0] + "_pred.png" for f in rgb_files]

for rgb_name, pred_name in zip(rgb_files, pred_files):
    rgb_path  = os.path.join(RGB_DIR,  rgb_name)
    pred_path = os.path.join(PRED_DIR, pred_name)

    if not os.path.exists(pred_path):
        print(f"Prediction not found: {pred_name} — run test.py first")
        continue

    rgb  = np.array(Image.open(rgb_path).convert("RGB"))
    pred = np.array(Image.open(pred_path).convert("RGB"))

    fig, axes = plt.subplots(1, 2, figsize=(16, 7))

    axes[0].imshow(rgb)
    axes[0].set_title("Original Image", fontsize=14)
    axes[0].axis("off")

    axes[1].imshow(pred)
    axes[1].set_title("Predicted Segmentation", fontsize=14)
    axes[1].axis("off")

    # Legend
    fig.legend(handles=make_legend(), loc="lower center",
               ncol=5, fontsize=9, framealpha=0.9)

    plt.suptitle(rgb_name, fontsize=12, y=1.01)
    plt.tight_layout()

    save_path = os.path.join(OUT_DIR, os.path.splitext(rgb_name)[0] + "_viz.png")
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {save_path}")

print(f"\n✅ Visualizations saved → {OUT_DIR}/")
