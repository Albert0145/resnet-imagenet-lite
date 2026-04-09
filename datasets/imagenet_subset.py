from pathlib import Path
from torchvision import datasets


def build_classification_datasets(cfg, train_transform, val_transform):
    train_root = Path(cfg.data.train_dir)
    val_root = Path(cfg.data.val_dir)

    train_set = datasets.ImageFolder(train_root, transform=train_transform)
    val_set = datasets.ImageFolder(val_root, transform=val_transform)

    return train_set, val_set