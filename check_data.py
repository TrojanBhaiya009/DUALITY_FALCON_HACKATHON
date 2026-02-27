"""
Dataset verify karne ke liye
Run: python check_data.py
"""
import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from PIL import Image

TRAIN_RGB = "train/Color_Images"
TRAIN_SEG = "train/Segmentation"
VAL_RGB   = "val/Color_Images"
VAL_SEG   = "val/Segmentation"
TEST_DIR  = "test/Color_Images"

EXPECTED_IDS = {100, 200, 300, 500, 550, 600, 700, 800, 7100, 10000}
CLASS_NAMES  = {
    100:"Trees", 200:"Lush Bushes", 300:"Dry Grass",
    500:"Dry Bushes", 550:"Ground Clutter", 600:"Flowers",
    700:"Logs", 800:"Rocks", 7100:"Landscape", 10000:"Sky"
}

def count_dir(path):
    if not os.path.exists(path):
        return 0
    return len([f for f in os.listdir(path) if f.lower().endswith(('.png','.jpg','.jpeg'))])

print("=" * 50)
print("  DATASET CHECK")
print("=" * 50)
print(f"  Train RGB   : {count_dir(TRAIN_RGB)} images")
print(f"  Train SEG   : {count_dir(TRAIN_SEG)} masks")
print(f"  Val   RGB   : {count_dir(VAL_RGB)} images")
print(f"  Val   SEG   : {count_dir(VAL_SEG)} masks")
print(f"  Test  Images: {count_dir(TEST_DIR)} images")
print("=" * 50)

# Check mask pixel values
rgb_files = sorted(os.listdir(TRAIN_RGB))
all_ids   = set()

print("\nChecking first 5 masks...")
for fname in rgb_files[:5]:
    seg_path = os.path.join(TRAIN_SEG, fname)
    if not os.path.exists(seg_path):
        print(f"  ⚠️  Mask not found: {fname}")
        continue
    mask = np.array(Image.open(seg_path))
    ids  = set(np.unique(mask).tolist())
    all_ids |= ids
    print(f"  {fname} | shape:{mask.shape} | IDs:{sorted(ids)}")

print(f"\nAll unique IDs found : {sorted(all_ids)}")
unknown = all_ids - EXPECTED_IDS
if unknown:
    print(f"⚠️  UNKNOWN IDs: {unknown}")
else:
    print("✅ All IDs are correct!")

# Visual check
sample_rgb  = np.array(Image.open(os.path.join(TRAIN_RGB, rgb_files[0])).convert("RGB"))
sample_mask = np.array(Image.open(os.path.join(TRAIN_SEG, rgb_files[0])))

fig, axes = plt.subplots(1, 2, figsize=(14, 6))
axes[0].imshow(sample_rgb)
axes[0].set_title("RGB Image")
axes[0].axis("off")
axes[1].imshow(sample_mask, cmap="tab20")
axes[1].set_title("Segmentation Mask (raw IDs)")
axes[1].axis("off")
plt.tight_layout()
plt.savefig("data_check.png", dpi=150)
print("\n✅ Visual saved → data_check.png")
plt.close()
