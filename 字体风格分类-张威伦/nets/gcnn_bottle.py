from torch import nn


class GcnnBottle(nn.Module):
    """
    需要考虑采用BN和不采用BN的结果
    """

    def __init__(self) -> None:
        super(GcnnBottle, self).__init__()
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
        self.Bottleneck = nn.Linear(512, 128, bias=False)
        self.last_bn = nn.BatchNorm1d(128, eps=0.001, momentum=0.1, affine=True)
        self.classifier = nn.Linear(128, 4)
        # gram矩阵操作
        self.gram_model = nn.Sequential(
            nn.ZeroPad2d(padding=1),
            nn.Conv2d(in_channels=512, out_channels=32, kernel_size=(19, 19), bias=False),
            # nn.Conv2D(kernel_size=19, filters=32, padding='same'),
            nn.BatchNorm2d(32),
            nn.ReLU6(),
            nn.MaxPool2d(kernel_size=(2, 2), stride=1)
        )
        # 全连接分类层
        self.full = nn.Sequential(
            nn.Linear(512, 2048),
            nn.ReLU6(),
            nn.Linear(2048, 512),
            nn.ReLU(),
            nn.Linear(512, 4),
            nn.Softmax()
        )

    def forward(self, x):
        x = self.c1(x)
        x = self.c2(x)
        x = self.c3(x)
        x = self.c4(x)
        x = self.avg(x)
        x = x.view(x.size(0), -1)
        x = self.Bottleneck(x)
        before_normalize = self.last_bn(x)
        cls = self.classifier(before_normalize)
        return cls
