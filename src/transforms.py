import albumentations as A
from albumentations.pytorch import ToTensorV2

MEAN = [0.485, 0.456, 0.406]
STD  = [0.229, 0.224, 0.225]

def get_train_transforms(img_size=256):
    return A.Compose([
        A.Resize(img_size, img_size),
        A.HorizontalFlip(p=0.5),
        A.ShiftScaleRotate(shift_limit=0.08, scale_limit=0.15, rotate_limit=10,
                           border_mode=0, p=0.4),
        A.OneOf([
            A.GaussNoise(std_range=(0.1, 0.3), p=1),
            A.GaussianBlur(blur_limit=(3, 5), p=1),
        ], p=0.2),
        A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.4),
        A.HueSaturationValue(hue_shift_limit=8, sat_shift_limit=15,
                             val_shift_limit=10, p=0.2),
        A.Normalize(mean=MEAN, std=STD),
        ToTensorV2()
    ], additional_targets={"mask": "mask"})

def get_val_transforms(img_size=256):
    return A.Compose([
        A.Resize(img_size, img_size),
        A.Normalize(mean=MEAN, std=STD),
        ToTensorV2()
    ], additional_targets={"mask": "mask"})

def get_test_transforms(img_size=256):
    return A.Compose([
        A.Resize(img_size, img_size),
        A.Normalize(mean=MEAN, std=STD),
        ToTensorV2()
    ])
