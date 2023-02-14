import torch
from torch import nn
from torchvision import models
import torch.nn.functional as F

EPSILON = 1e-12
import nets


class GenBone(nn.Module):
    def __init__(self):
        super(GenBone, self).__init__()

        self.c1 = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=64, kernel_size=(3, 3), padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU6(),
            nn.MaxPool2d(kernel_size=(3, 3), stride=2, padding=1)
        )
        self.c2 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=(3, 3), padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU6(),
            nn.MaxPool2d(kernel_size=(3, 3), stride=2, padding=1)
        )
        self.c3 = nn.Sequential(
            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=(3, 3), padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU6(),
            nn.MaxPool2d(kernel_size=(3, 3), stride=2, padding=1)
        )

        self.c4 = nn.Sequential(
            nn.Conv2d(in_channels=256, out_channels=512, kernel_size=(3, 3), padding=1, bias=False),
            nn.BatchNorm2d(512),
            nn.ReLU6(),
            nn.MaxPool2d(kernel_size=(3, 3), stride=2, padding=1)
        )
        # 放大两倍 使用邻近填充
        self.upsample = nn.Upsample(scale_factor=2, mode="nearest")

    def forward(self, x):
        x = self.c1(x)
        x1 = x
        x = self.c2(x)
        x2 = x
        x = self.c3(x)
        x3 = x
        x = self.c4(x)
        x4 = x
        return x4


# Bilinear Attention Pooling
class BAP(nn.Module):
    def __init__(self, pool='GAP'):
        super(BAP, self).__init__()
        assert pool in ['GAP', 'GMP']
        if pool == 'GAP':
            self.pool = None
        else:
            self.pool = nn.AdaptiveMaxPool2d(1)

    def forward(self, features, attentions):
        B, C, H, W = features.size()
        _, M, AH, AW = attentions.size()
        # match size
        if AH != H or AW != W:
            attentions = F.upsample_bilinear(attentions, size=(H, W))
        # feature_matrix: (B, M, C) -> (B, M * C)
        if self.pool is None:
            feature_matrix = (torch.einsum('imjk,injk->imn', (attentions, features)) / float(H * W)).view(B, -1)
        else:
            feature_matrix = []
            for i in range(M):
                AiF = self.pool(features * attentions[:, i:i + 1, ...]).view(B, -1)
                feature_matrix.append(AiF)
            feature_matrix = torch.cat(feature_matrix, dim=1)
        # sign-sqrt
        feature_matrix = torch.sign(feature_matrix) * torch.sqrt(torch.abs(feature_matrix) + EPSILON)
        # l2 normalization along dimension M and C
        feature_matrix = F.normalize(feature_matrix, dim=-1)
        return feature_matrix


class BasicConv2d(nn.Module):

    def __init__(self, in_channels, out_channels, **kwargs):
        super(BasicConv2d, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, bias=False, **kwargs)
        self.bn = nn.BatchNorm2d(out_channels, eps=0.001)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        return F.relu(x, inplace=True)


class GcnnCbam(nn.Module):
    """
    需要考虑采用BN和不采用BN的结果
    """

    def __init__(self, backbone="gcnn", num_classes=4, dropout_keep_prob=0.5, embedding_size=128) -> None:
        super(GcnnCbam, self).__init__()
        if backbone == "gcnn":
            self.back_bone = GenBone()
            self.flat_shape = 512
        self.avg = nn.AdaptiveAvgPool2d((1, 1))
        self.cbam_1 = nets.CBAMLayer(512)
        self.avg = nn.AdaptiveAvgPool2d((1, 1))

        # Attention Maps
        self.attentions_1 = BasicConv2d(512, 32, kernel_size=1)
        # Bilinear Attention Pooling
        self.bap = BAP(pool='GAP')
        self.Dropout = nn.Dropout(1 - dropout_keep_prob)
        self.Bottleneck = nn.Linear(self.flat_shape, embedding_size, bias=False)
        self.last_bn = nn.BatchNorm1d(embedding_size, eps=0.001, momentum=0.1, affine=True)
        self.classifier = nn.Linear(512, num_classes)

    def forward(self, x):
        x = self.back_bone(x)
        x = self.cbam_1(x)
        x = self.avg(x)
        x = x.view(x.size(0), -1)
        cls = self.classifier(x)
        return cls
