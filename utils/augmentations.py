"""Data augmentation pipeline using Albumentations."""

import albumentations as A
from albumentations.pytorch import ToTensorV2


def get_training_augmentation(image_size: tuple[int, int] = (512, 512)) -> A.Compose:
    """Get training augmentation pipeline.

    Args:
        image_size: Target (height, width)

    Returns:
        Albumentations Compose object
    """
    return A.Compose([
        A.Resize(image_size[0], image_size[1]),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.1),
        A.RandomRotate90(p=0.3),
        A.ShiftScaleRotate(
            shift_limit=0.1,
            scale_limit=0.1,
            rotate_limit=15,
            p=0.5,
            border_mode=0,
        ),
        A.OneOf([
            A.ElasticTransform(alpha=120, sigma=120 * 0.05, p=1),
            A.GridDistortion(p=1),
            A.OpticalDistortion(distort_limit=0.05, shift_limit=0.05, p=1),
        ], p=0.2),
        A.OneOf([
            A.GaussNoise(p=1),
            A.GaussianBlur(p=1),
            A.MotionBlur(p=1),
        ], p=0.2),
        A.OneOf([
            A.RandomBrightnessContrast(p=1),
            A.HueSaturationValue(p=1),
            A.CLAHE(p=1),
        ], p=0.3),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2(),
    ])


def get_validation_augmentation(image_size: tuple[int, int] = (512, 512)) -> A.Compose:
    """Get validation augmentation pipeline (no random augmentations).

    Args:
        image_size: Target (height, width)

    Returns:
        Albumentations Compose object
    """
    return A.Compose([
        A.Resize(image_size[0], image_size[1]),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2(),
    ])


def get_test_augmentation(image_size: tuple[int, int] = (512, 512)) -> A.Compose:
    """Get test-time augmentation pipeline.

    Args:
        image_size: Target (height, width)

    Returns:
        Albumentations Compose object
    """
    return A.Compose([
        A.Resize(image_size[0], image_size[1]),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2(),
    ])
