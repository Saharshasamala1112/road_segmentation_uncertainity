"""PyTorch Dataset and DataLoader for road segmentation."""

import os
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
from typing import Optional
import albumentations as A
from albumentations.pytorch import ToTensorV2


class RoadSegmentationDataset(Dataset):
    """Dataset for road segmentation with paired images and masks."""

    def __init__(
        self,
        image_dir: str,
        mask_dir: str,
        transform: Optional[A.Compose] = None,
        image_size: tuple[int, int] = (512, 512),
    ):
        self.image_dir = Path(image_dir)
        self.mask_dir = Path(mask_dir)
        self.transform = transform
        self.image_size = image_size

        # Get sorted list of image files
        valid_extensions = {".jpg", ".jpeg", ".png", ".bmp", ".tiff"}
        self.images = sorted([
            f for f in self.image_dir.iterdir()
            if f.suffix.lower() in valid_extensions
        ])

        if not self.images:
            raise FileNotFoundError(
                f"No images found in {self.image_dir}. "
                f"Supported formats: {valid_extensions}"
            )

    def __len__(self) -> int:
        return len(self.images)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        """Load and return a single image-mask pair.

        Returns:
            Dictionary with 'image' (C, H, W) and 'mask' (1, H, W) tensors
        """
        img_path = self.images[idx]
        mask_path = self.mask_dir / img_path.name

        # Load image
        image = cv2.imread(str(img_path))
        if image is None:
            raise ValueError(f"Failed to load image: {img_path}")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Load mask
        if mask_path.exists():
            mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
            if mask is None:
                raise ValueError(f"Failed to load mask: {mask_path}")
        else:
            # Create empty mask if not found
            mask = np.zeros(image.shape[:2], dtype=np.uint8)

        # Apply transforms
        if self.transform:
            augmented = self.transform(image=image, mask=mask)
            image = augmented["image"]
            mask = augmented["mask"]
        else:
            # Default: resize and convert to tensor
            image = cv2.resize(image, (self.image_size[1], self.image_size[0]))
            mask = cv2.resize(mask, (self.image_size[1], self.image_size[0]))
            image = torch.from_numpy(image.transpose(2, 0, 1)).float() / 255.0
            mask = torch.from_numpy(mask).unsqueeze(0).float() / 255.0

        # Ensure mask is binary
        mask = (mask > 0.5).float()

        return {"image": image, "mask": mask, "name": img_path.name}


def build_transforms(config: dict, is_train: bool = True) -> A.Compose:
    """Build augmentation pipeline from config.

    Args:
        config: Augmentation configuration dictionary
        is_train: Whether this is for training (includes augmentations)

    Returns:
        Albumentations Compose object
    """
    transforms = []

    if is_train:
        if config.get("horizontal_flip", 0) > 0:
            transforms.append(A.HorizontalFlip(p=config["horizontal_flip"]))
        if config.get("vertical_flip", 0) > 0:
            transforms.append(A.VerticalFlip(p=config["vertical_flip"]))
        if config.get("random_rotate", 0) > 0:
            transforms.append(A.RandomRotate90(p=config["random_rotate"]))
        if config.get("color_jitter", 0) > 0:
            transforms.append(A.ColorJitter(p=config["color_jitter"]))
        if config.get("gaussian_noise", 0) > 0:
            transforms.append(A.GaussNoise(p=config["gaussian_noise"]))
        if config.get("blur", 0) > 0:
            transforms.append(A.Blur(p=config["blur"]))

    # Always apply normalization and tensor conversion
    transforms.extend([
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2(),
    ])

    return A.Compose(transforms)


def build_dataloaders(config: dict) -> tuple[DataLoader, DataLoader]:
    """Build training and validation dataloaders.

    Args:
        config: Full configuration dictionary

    Returns:
        Tuple of (train_loader, val_loader)
    """
    aug_config = config.get("augmentation", {})
    data_config = config.get("data", {})
    train_config = config.get("training", {})

    image_size = tuple(data_config.get("image_size", [512, 512]))

    # Training dataset
    train_transform = build_transforms(aug_config, is_train=True)
    train_dataset = RoadSegmentationDataset(
        image_dir=data_config["train_images"],
        mask_dir=data_config["train_masks"],
        transform=train_transform,
        image_size=image_size,
    )

    # Validation dataset
    val_transform = build_transforms({}, is_train=False)
    val_dataset = RoadSegmentationDataset(
        image_dir=data_config["val_images"],
        mask_dir=data_config["val_masks"],
        transform=val_transform,
        image_size=image_size,
    )

    # Dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=train_config.get("batch_size", 8),
        shuffle=True,
        num_workers=train_config.get("num_workers", 4),
        pin_memory=True,
        drop_last=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=train_config.get("batch_size", 8),
        shuffle=False,
        num_workers=train_config.get("num_workers", 4),
        pin_memory=True,
    )

    return train_loader, val_loader
