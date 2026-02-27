# Offroad Semantic Segmentation

Semantic segmentation pipeline for offroad terrain imagery. Classifies each pixel into one of 10 terrain categories.

**mAP50: 51.2%**

## Setup

```bash
pip install torch torchvision segmentation-models-pytorch albumentations opencv-python matplotlib pillow tqdm
```

## Usage

1. Verify dataset integrity:
```bash
python check_data.py
```

2. Train the model:
```bash
python train.py
```

3. Run inference on test images:
```bash
python test.py
```

4. Visualize predictions:
```bash
python visualize.py
```

## Folder Structure

```
duality_hackathon/
├── train/
│   ├── Color_Images/          training RGB images
│   └── Segmentation/          training masks
├── val/
│   ├── Color_Images/          validation RGB images
│   └── Segmentation/          validation masks
├── test/
│   └── Color_Images/          test images (held out)
├── src/
│   ├── dataset.py             data loading and mask remapping
│   ├── transforms.py          augmentation pipeline
│   ├── losses.py              combined CrossEntropy + Dice loss
│   └── metrics.py             IoU and mAP50 computation
├── runs/
│   ├── best_model.pth         saved checkpoint
│   ├── training_graphs.png    loss/metric curves
│   └── predictions/           test output masks
├── train.py
├── test.py
├── visualize.py
├── check_data.py
└── ENV_SETUP/setup_env.bat
```

## Classes

| ID    | Class          |
|-------|----------------|
| 100   | Trees          |
| 200   | Lush Bushes    |
| 300   | Dry Grass      |
| 500   | Dry Bushes     |
| 550   | Ground Clutter |
| 600   | Flowers        |
| 700   | Logs           |
| 800   | Rocks          |
| 7100  | Landscape      |
| 10000 | Sky            |

## Model

U-Net with MobileNetV2 encoder, pretrained on ImageNet. Trained for 5 epochs at 256x256 resolution using AdamW with OneCycleLR scheduling. Loss function is a combination of CrossEntropy and Dice loss. Evaluation uses per-class IoU and mAP50.
```
