import os
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset

# ── Raw pixel ID → class index ──────────────────────────
ID_TO_CLASS = {
    100:   0,   # Trees
    200:   1,   # Lush Bushes
    300:   2,   # Dry Grass
    500:   3,   # Dry Bushes
    550:   4,   # Ground Clutter
    600:   5,   # Flowers
    700:   6,   # Logs
    800:   7,   # Rocks
    7100:  8,   # Landscape
    10000: 9,   # Sky
}

CLASS_NAMES = [
    "Trees", "Lush Bushes", "Dry Grass", "Dry Bushes",
    "Ground Clutter", "Flowers", "Logs", "Rocks",
    "Landscape", "Sky"
]

# Build a fast vectorized lookup table (max raw ID is 10000)
_MAX_ID = max(ID_TO_CLASS.keys()) + 1
_LUT = np.zeros(_MAX_ID + 1, dtype=np.uint8)
for _raw, _cls in ID_TO_CLASS.items():
    _LUT[_raw] = _cls

def remap_mask(mask_array):
    # Clip to valid range then do O(1) per-pixel lookup
    return _LUT[np.clip(mask_array, 0, _MAX_ID)]


class DesertDataset(Dataset):
    def __init__(self, rgb_dir, seg_dir, transform=None):
        self.rgb_dir   = rgb_dir
        self.seg_dir   = seg_dir
        self.transform = transform
        self.images    = sorted([
            f for f in os.listdir(rgb_dir)
            if f.lower().endswith(('.png', '.jpg', '.jpeg'))
        ])

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        name = self.images[idx]
        img  = np.array(Image.open(os.path.join(self.rgb_dir, name)).convert("RGB"))
        mask = np.array(Image.open(os.path.join(self.seg_dir, name)))
        mask = remap_mask(mask)

        if self.transform:
            out  = self.transform(image=img, mask=mask)
            img  = out["image"]
            mask = out["mask"]

        return img, mask.clone().detach().long() if isinstance(mask, torch.Tensor) else torch.as_tensor(mask, dtype=torch.long)


class TestDataset(Dataset):
    """testImages folder ke liye — no masks"""
    def __init__(self, rgb_dir, transform=None):
        self.rgb_dir   = rgb_dir
        self.transform = transform
        self.images    = sorted([
            f for f in os.listdir(rgb_dir)
            if f.lower().endswith(('.png', '.jpg', '.jpeg'))
        ])

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        name = self.images[idx]
        img  = np.array(Image.open(os.path.join(self.rgb_dir, name)).convert("RGB"))
        if self.transform:
            img = self.transform(image=img)["image"]
        return img, name
