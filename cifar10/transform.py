import torch
import albumentations as A
from albumentations.pytorch import ToTensorV2

from torchvision import transforms

torch.manual_seed(1)


def get_a_train_transform():
    """Get transformer for training data

    Returns:
        Compose: Composed transformations
    """
    return A.Compose([
        A.HorizontalFlip(p=0.5),
        A.ShiftScaleRotate(shift_limit=0.09, scale_limit=0.09, rotate_limit=7, p=0.5),
        #A.RandomBrightnessContrast(p=0.5),
        A.CoarseDropout(max_holes = 1, max_height=16, max_width=16, min_holes = 1, min_height=16, min_width=16, fill_value=(0.4914, 0.4822, 0.4465), mask_fill_value = None),
        A.Normalize(mean=(0.4914, 0.4822, 0.4465), std=(0.247, 0.243, 0.261)),
        ToTensorV2()
    ])


def get_a_test_transform():
    """Get transformer for test data

    Returns:
        Compose: Composed transformations
    """
    return A.Compose([
        A.Normalize(mean=(0.4914, 0.4822, 0.4465), std=(0.247, 0.243, 0.261)),
        ToTensorV2(),
    ])


def get_p_train_transform():
    """Get Pytorch Transform function for train data

    Returns:
        Compose: Composed transformations
    """
    random_rotation_degree = 5
    img_size = (28, 28)
    random_crop_percent = (0.85, 1.0)
    return transforms.Compose([
        transforms.RandomResizedCrop(img_size, random_crop_percent),
        transforms.RandomRotation(random_rotation_degree),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.4914, 0.4822, 0.4465), std=(0.247, 0.243, 0.261)),
    ])


def get_p_test_transform():
    """Get Pytorch Transform function for test data

    Returns:
        Compose: Composed transformations
    """
    return transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.4914, 0.4822, 0.4465), std=(0.247, 0.243, 0.261)),
    ])
