import torch
import torch.nn as nn
import math
from torchvision import models
from torchsummary import summary
from triplet_net.effinet import efficientnetv2_s


class PaperMethod(nn.Module):
    def __init__(self, pretrained=False):
        super(PaperMethod, self).__init__()
        # self.cnn = models.vgg16_bn(pretrained=pretrained).features
        self.cnn = efficientnetv2_s()
        self.conv = nn.Conv2d(256, 512, kernel_size=(3, 3), stride=(1, 1), padding=1, bias=False)
        self.fusion = Bottle2neck(512, 128, baseWidth=64)
        self.pas = PSA(512, 512)
        # self.cls = nn.Sequential(
        #     nn.Conv2d(512, 384, 3, padding=1),
        #     nn.BatchNorm2d(384),
        #     nn.ReLU(),
        #     nn.Conv2d(384, 256, 3, padding=1),
        #     nn.BatchNorm2d(256),
        #     nn.ReLU(),
        #     nn.Conv2d(256, num_classes, 1),
        #     nn.AdaptiveMaxPool2d((1, 1)),
        #     nn.Flatten(1)
        # )

    def forward(self, x):
        x = self.cnn.stem(x)
        x = self.cnn.blocks(x)
        x = self.conv(x)
        x = self.fusion(x)
        x = self.pas(x)
        # return self.cls(x)
        return x


class Bottle2neck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None, baseWidth=26, scale=4):
        super(Bottle2neck, self).__init__()

        width = int(math.floor(planes * (baseWidth / 64.0)))
        self.conv1 = nn.Conv2d(inplanes, width * scale,
                               kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(width * scale)

        if scale == 1:
            self.nums = 1
        else:
            self.nums = scale - 1
        convs = []
        bns = []
        for i in range(self.nums):
            convs.append(nn.Conv2d(width, width, kernel_size=3,
                                   stride=stride, padding=1, bias=False))
            bns.append(nn.BatchNorm2d(width))
        self.convs = nn.ModuleList(convs)
        self.bns = nn.ModuleList(bns)

        self.conv3 = nn.Conv2d(width * scale, planes *
                               self.expansion, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * self.expansion)

        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.scale = scale
        self.width = width

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        spx = torch.split(out, self.width, 1)
        for i in range(self.nums):
            if i == 0:
                sp = spx[0]
            else:
                sp = sp + spx[i]
            sp = self.convs[i](sp)
            sp = self.relu(self.bns[i](sp))
            if i == 0:
                out = sp
            else:
                out = torch.cat((out, sp), 1)
        out = torch.cat((out, spx[self.nums]), 1)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class ResidualBlock(nn.Module):

    def __init__(self, input_channels, output_channels, stride=1):
        super(ResidualBlock, self).__init__()
        self.input_channels = input_channels
        self.output_channels = output_channels
        self.stride = stride
        self.bn1 = nn.BatchNorm2d(input_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv2d(
            input_channels, output_channels // 4, 1, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(output_channels // 4)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(
            output_channels // 4, output_channels // 4, 3, stride, padding=1, bias=False)
        self.bn3 = nn.BatchNorm2d(output_channels // 4)
        self.relu = nn.ReLU(inplace=True)
        self.conv3 = nn.Conv2d(
            output_channels // 4, output_channels, 1, 1, bias=False)
        self.conv4 = nn.Conv2d(
            input_channels, output_channels, 1, stride, bias=False)

    def forward(self, x):
        residual = x
        out = self.bn1(x)
        out1 = self.relu(out)
        out = self.conv1(out1)
        out = self.bn2(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn3(out)
        out = self.relu(out)
        out = self.conv3(out)
        if (self.input_channels != self.output_channels) or (self.stride != 1):
            residual = self.conv4(out1)
        out += residual
        return out


class SEWeight(nn.Module):
    def __init__(self, channel, ratio=16):
        super(SEWeight, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // ratio, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // ratio, channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, h, w = x.size()
        avg_pool_out = self.avg_pool(x).view([b, c])
        fc_out = self.fc(avg_pool_out).view([b, c, 1, 1])
        return fc_out


class PSA(nn.Module):
    def __init__(self, inplanes, planes, stride=1, conv_kernels=[3, 5, 7, 9], conv_groups=[1, 4, 8, 16]):
        super(PSA, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes // 4, kernel_size=conv_kernels[0], padding=conv_kernels[0] // 2,
                               stride=stride, groups=conv_groups[0])
        self.conv2 = nn.Conv2d(inplanes, planes // 4, kernel_size=conv_kernels[1], padding=conv_kernels[1] // 2,
                               stride=stride, groups=conv_groups[1])
        self.conv3 = nn.Conv2d(inplanes, planes // 4, kernel_size=conv_kernels[2], padding=conv_kernels[2] // 2,
                               stride=stride, groups=conv_groups[2])
        self.conv4 = nn.Conv2d(inplanes, planes // 4, kernel_size=conv_kernels[3], padding=conv_kernels[3] // 2,
                               stride=stride, groups=conv_groups[3])
        self.se = SEWeight(planes // 4)
        self.split_channel = planes // 4
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        b, c, h, w = x.size()
        x1 = self.conv1(x)
        x2 = self.conv2(x)
        x3 = self.conv3(x)
        x4 = self.conv4(x)
        feats = torch.cat([x1, x2, x3, x4], dim=1)
        feats = feats.view([b, 4, self.split_channel, h, w])

        x1_se = self.se(x1)
        x2_se = self.se(x2)
        x3_se = self.se(x3)
        x4_se = self.se(x4)

        x_se = torch.cat([x1_se, x2_se, x3_se, x4_se], dim=1)
        att_vector = x_se.view(b, 4, self.split_channel, 1, 1)
        att_vector = self.softmax(att_vector)
        feats_vector = feats * att_vector

        for i in range(4):
            x_se_weight_fp = feats_vector[:, i, :, :]
            if i == 0:
                outs = x_se_weight_fp
            else:
                outs = torch.cat((x_se_weight_fp, outs), dim=1)
        return outs


if __name__ == '__main__':
    p = PaperMethod(10)
    summary(p, input_size=(3, 224, 224), batch_size=-1)
