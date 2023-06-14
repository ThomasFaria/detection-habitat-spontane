import albumentations as album
from albumentations.pytorch.transforms import ToTensorV2
from torch.utils.data import random_split
import torch


def instantiate_dataset_test(config):
    # charger les exemples tests sur le datalab
    # change detection ou segmentation et / Sentinele 2 / PLeiade  et ho
    # récupérer tout le jeu de test
    # le splitter et le laisser dans l'ordre
    # save les noms etc et s'arranger pour reconstruire les masques totaux
    # sur grosses images
    return None


def split_dataset(dataset, prop_val):
    """
    Splits a given dataset into training and
    validation sets based on a given proportion.

    Args:
        dataset (torch.utils.data.Dataset): The dataset to split.
        prop_val (float): The proportion of the dataset to use for validation,
        should be between 0 and 1.

    Returns:
        (torch.utils.data.Dataset, torch.utils.data.Dataset):
        A tuple containing the training and validation datasets.

    """
    val_size = int(prop_val * len(dataset))
    train_size = len(dataset) - val_size

    dataset_train, dataset_val = random_split(dataset, [train_size, val_size])

    dataset_train = dataset_train.dataset
    dataset_val = dataset_val.dataset

    return dataset_train, dataset_val


def generate_transform(tile_size, augmentation, task: str):
    """
    Generates PyTorch transforms for data augmentation and preprocessing.

    Args:
        tile_size (int): The size of the image tiles.
        augmentation (bool): Whether or not to include data augmentation.
        task (str): Task.

    Returns:
        (albumentations.core.composition.Compose,
        albumentations.core.composition.Compose):
        A tuple containing the augmentation and preprocessing transforms.

    """
    image_size = (tile_size, tile_size)

    transforms_augmentation = None

    if augmentation:
        transforms_list = [
            album.Resize(300, 300, always_apply=True),
            album.RandomResizedCrop(
                *image_size, scale=(0.7, 1.0), ratio=(0.7, 1)
            ),
            album.HorizontalFlip(),
            album.VerticalFlip(),
            album.Normalize(),
            ToTensorV2(),
        ]
        if task == "detection":
            transforms_augmentation = album.Compose(
                transforms_list,
                bbox_params=album.BboxParams(
                    format="pascal_voc",
                    label_fields=['class_labels']
                )
            )
        else:
            transforms_augmentation = album.Compose(
                transforms_list
            )

    test_transforms_list = [
        album.Resize(*image_size, always_apply=True),
        album.Normalize(),
        ToTensorV2(),
    ]
    if task == "detection":
        transforms_preprocessing = album.Compose(
            test_transforms_list,
            bbox_params=album.BboxParams(
                format="pascal_voc",
                label_fields=['class_labels']
            )
        )
    else:
        transforms_preprocessing = album.Compose(
            test_transforms_list
        )

    return transforms_augmentation, transforms_preprocessing


def collate_fn(batch):
    """
    Collate function for object detection Dataloader.
    """
    images = []
    targets = []
    metadatas = []

    for i, t, m in batch:
        images.append(i)
        targets.append(t)
        metadatas.append(m)
    images = torch.stack(images, dim=0)
    return images, targets, metadatas
