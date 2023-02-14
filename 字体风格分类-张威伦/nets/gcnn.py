from torch import nn


class GCNN(nn.Module):
    """
    需要考虑采用BN和不采用BN的结果
    """

    def __init__(self) -> None:
        super(GCNN, self).__init__()
        self.c1 = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=64, kernel_size=(4, 4), bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU6(),
            nn.MaxPool2d(kernel_size=(2, 2), stride=1)
        )
        self.c2 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=(4, 4), bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU6(),
            nn.MaxPool2d(kernel_size=(2, 2), stride=1)
        )
        self.c3 = nn.Sequential(
            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=(4, 4), bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU6(),
            nn.MaxPool2d(kernel_size=(2, 2), stride=1)
        )

        self.c4 = nn.Sequential(
            nn.Conv2d(in_channels=256, out_channels=512, kernel_size=(4, 4), bias=False),
            nn.BatchNorm2d(512),
            nn.ReLU6(),
            nn.MaxPool2d(kernel_size=(2, 2), stride=1)
        )
        self.avg = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Linear(512, 4)

    def forward(self, x):
        x = self.c1(x)
        x = self.c2(x)
        x = self.c3(x)
        x = self.c4(x)
        x = self.avg(x)
        x = x.view(x.size(0), -1)
        cls = self.classifier(x)
        return cls
