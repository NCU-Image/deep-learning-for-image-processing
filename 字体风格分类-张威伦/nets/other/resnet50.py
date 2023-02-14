from torch import nn
from torchvision import models


class Resnet50(nn.Module):
    """
    需要考虑采用BN和不采用BN的结果
    """

    def __init__(self) -> None:
        super(Resnet50, self).__init__()
        self.resnet50 = models.resnet50()
        self.classifier = nn.Linear(1000, 4)

    def forward(self, x):
        x = self.resnet50(x)
        cls = self.classifier(x)
        return cls
