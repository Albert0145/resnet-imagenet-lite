from torchvision import transforms


def build_train_transform(cfg):
    if getattr(cfg.data, "use_custom_preprocess", False):
        # 以后在这里接你旧的 preprocess 逻辑
        # 当前先保留接口，不启用复杂自定义流程
        pass

    return transforms.Compose([
        transforms.RandomResizedCrop(cfg.data.image_size),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=cfg.data.mean, std=cfg.data.std),
    ])


def build_val_transform(cfg):
    if getattr(cfg.data, "use_custom_preprocess", False):
        # 以后在这里接你旧的 preprocess 逻辑
        pass

    resize_size = int(cfg.data.image_size / 0.875)
    return transforms.Compose([
        transforms.Resize(resize_size),
        transforms.CenterCrop(cfg.data.image_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=cfg.data.mean, std=cfg.data.std),
    ])