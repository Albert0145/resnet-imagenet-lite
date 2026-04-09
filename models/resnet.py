import torchvision.models as tvm
from torchvision.models import ResNet18_Weights, ResNet34_Weights


def build_resnet(depth: int, num_classes: int, pretrained: bool = False):
    if depth == 18:
        weights = ResNet18_Weights.DEFAULT if pretrained else None
        model = tvm.resnet18(weights=weights)
    elif depth == 34:
        weights = ResNet34_Weights.DEFAULT if pretrained else None
        model = tvm.resnet34(weights=weights)
    else:
        raise ValueError(f"Unsupported depth: {depth}")

    model.fc = __import__("torch").nn.Linear(model.fc.in_features, num_classes)
    return model