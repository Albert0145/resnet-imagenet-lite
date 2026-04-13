from utils.config import load_config
from scripts.train import build_iters_with_old_dataiter, build_model
import torch


def main():
    cfg = load_config("configs/res18.yaml")

    if torch.cuda.is_available() and "cuda" in cfg.train.device:
        device = torch.device(cfg.train.device)
    else:
        device = torch.device("cpu")

    print("Device:", device)

    train_iter, val_iter, num_classes = build_iters_with_old_dataiter(cfg, device)
    print("Iter built.")
    print("Num classes:", num_classes)

    model = build_model(cfg, num_classes=num_classes).to(device)
    print("Model built.")

    batch = next(iter(train_iter))
    X, y = batch
    print("Type of X:", type(X))
    print("Len of X:", len(X))
    print("X[0].shape:", X[0].shape)
    print("y.shape:", y.shape)

    images = X[0].to(device)
    targets = y.to(device).long()

    logits = model(images)
    print("logits.shape:", logits.shape)
    print("targets.shape:", targets.shape)

    print("Smoke test passed.")


if __name__ == "__main__":
    main()