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
        x4 = self.upsample(x4)
        p1 = torch.cat([x3, x4], 1)
        x3 = self.upsample(x3)
        p2 = torch.cat([x2, x3], 1)
        x2 = self.upsample(x2)
        p3 = torch.cat([x1, x2], 1)
        return p1, p2, p3


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


class FineGrain(nn.Module):
    """
    需要考虑采用BN和不采用BN的结果
    """

    def __init__(self, backbone="gcnn", num_classes=4, dropout_keep_prob=0.5, embedding_size=128) -> None:
        super(FineGrain, self).__init__()
        if backbone == "gcnn":
            self.back_bone = GenBone()
            self.flat_shape = 43008
        elif backbone == "res50":
            self.flat_shape = 1280
        self.avg = nn.AdaptiveAvgPool2d((1, 1))
        self.cbam_1 = nets.CBAMLayer(768)
        self.cbam_2 = nets.CBAMLayer(384)
        self.cbam_3 = nets.CBAMLayer(192)

        # Attention Maps
        self.attentions_1 = BasicConv2d(768, 32, kernel_size=1)
        self.attentions_2 = BasicConv2d(384, 32, kernel_size=1)
        self.attentions_3 = BasicConv2d(192, 32, kernel_size=1)
        # Bilinear Attention Pooling
        self.bap = BAP(pool='GAP')
        self.Dropout = nn.Dropout(1 - dropout_keep_prob)
        self.Bottleneck = nn.Linear(self.flat_shape, embedding_size, bias=False)
        self.last_bn = nn.BatchNorm1d(embedding_size, eps=0.001, momentum=0.1, affine=True)
        self.classifier = nn.Linear(embedding_size, num_classes)

    def forward(self, x):
        x1, x2, x3 = self.back_bone(x)
        x1 = self.cbam_1(x1)
        x2 = self.cbam_2(x2)
        x3 = self.cbam_3(x3)
        attention_maps_1 = self.attentions_1(x1)
        attention_maps_2 = self.attentions_2(x2)
        attention_maps_3 = self.attentions_3(x3)
        feature_matrix_1 = self.bap(x1, attention_maps_1)
        feature_matrix_2 = self.bap(x2, attention_maps_2)
        feature_matrix_3 = self.bap(x3, attention_maps_3)
        # x1 = self.avg(x1)
        # x2 = self.avg(x2)
        # x3 = self.avg(x3)
        # x1 = x1.view(x1.size(0), -1)
        # x2 = x2.view(x2.size(0), -1)
        # x3 = x2.view(x3.size(0), -1)
        x = torch.cat([feature_matrix_1, feature_matrix_2, feature_matrix_3], 1)
        x = self.Bottleneck(x)
        before_normalize = self.last_bn(x)
        cls = self.classifier(before_normalize)
        return cls
