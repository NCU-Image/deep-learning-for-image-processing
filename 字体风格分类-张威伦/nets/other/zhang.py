import math

import torch
import torch.nn as nn
from torchvision import models


# zhang的论文


class ConvLayer(nn.Module):
    def __init__(self, in_channels, out_channels, conv_k, conv_s):
        super(ConvLayer, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.conv_k = conv_k
        self.conv_s = conv_s
        # self.b = nn.Parameter(torch.full(size=(in_channels,), fill_value=0.01), requires_grad=True)
        self.conv = nn.Conv2d(self.in_channels, self.out_channels, conv_k, conv_s, bias=True, padding=1)
        nn.init.xavier_uniform_(self.conv.weight)
        nn.init.constant_(self.conv.bias, 0.01)

    def forward(self, x):
        x = self.conv(x)
        return x


class SEBlock(nn.Module):
    def __init__(self, in_channels):
        super(SEBlock, self).__init__()
        self.in_channels = in_channels
        self.global_avg_pooling = nn.AdaptiveAvgPool2d((1, 1))
        self.fc_1 = nn.Linear(self.in_channels, self.in_channels // 16, bias=True)
        self.fc_2 = nn.Linear(self.in_channels // 16, self.in_channels, bias=True)
        self.act_1 = nn.ReLU()
        self.act_2 = nn.Sigmoid()

    def forward(self, x):
        y = self.global_avg_pooling(x)
        y = torch.squeeze(y, dim=-1)
        y = torch.squeeze(y, dim=-1)
        y = self.act_1(self.fc_1(y))
        y = self.act_2(self.fc_2(y))
        y.view(-1, x.shape[1])
        y = torch.unsqueeze(y, dim=-1)
        y = torch.unsqueeze(y, dim=-1)
        # print('se shape: {}'.format((x * y).shape))
        return x * y


class HaarWaveletBlock(nn.Module):
    def __init__(self):
        super(HaarWaveletBlock, self).__init__()
        self.global_avg_pooling = nn.AdaptiveAvgPool2d((1, 1))

    def forward(self, x):
        feature_map_size = x.shape[1]
        x = torch.squeeze(self.global_avg_pooling(x))
        length = feature_map_size // 2
        temp = torch.reshape(x, (-1, length, 2))
        a = (temp[:, :, 0] + temp[:, :, 1]) / 2
        detail = (temp[:, :, 0] - temp[:, :, 1]) / 2
        length = length // 2
        while length != 16:  # 一级：32，acc：97.5， 二级：16，acc：97.875，三级：8, acc: 98.628, 四级：4，acc: 97.625, 五级：2，acc：97.5，六级：1，acc：97.375
            a = torch.reshape(a, (-1, length, 2))
            detail = torch.cat(((a[:, :, 0] - a[:, :, 1]) / 2, detail), dim=1)
            a = (a[:, :, 0] + a[:, :, 1]) / 2
            length = length // 2
        haar_info = torch.cat((a, detail), dim=1)
        # print('haar shape: {}'.format(haar_info.shape))
        return haar_info


class CAM(nn.Module):

    def __init__(self, in_channels):
        super(CAM, self).__init__()
        self.in_channels = in_channels
        self.max_pool = nn.AdaptiveMaxPool2d((1, 1))
        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.mlp = nn.Sequential(
            nn.Linear(self.in_channels, self.in_channels // 16, bias=True),
            nn.ReLU(),
            nn.Linear(self.in_channels // 16, self.in_channels, bias=True),
            nn.Sigmoid()
        )
        self.sig = nn.Sigmoid()

    def forward(self, x):
        y1 = self.avg_pool(x)
        y2 = self.max_pool(x)
        y1, y2 = torch.squeeze(y1), torch.squeeze(y2)
        y1 = self.mlp(y1)
        y2 = self.mlp(y2)
        y = y1 + y2
        y = torch.unsqueeze(y, dim=-1)
        y = torch.unsqueeze(y, dim=-1)
        y = self.sig(y)
        return x * y


class SAM(nn.Module):

    def __init__(self):
        super(SAM, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(2, 1, 7, 1, 3, bias=True),
            nn.Sigmoid()
        )

    def forward(self, x):
        y1, _ = torch.max(x, dim=1, keepdim=True)
        y2 = torch.mean(x, dim=1, keepdim=True)
        y = torch.cat((y1, y2), dim=1)
        y = self.conv(y)
        return x * y


class BasicNet(nn.Module):
    """
    A novel CNN structure for fine-grained classification of Chinese calligraphy styles 2019
    """
    def __init__(self):
        super(BasicNet, self).__init__()
        self.conv_1 = nn.Sequential(
            ConvLayer(3, 32, 5, 1),
            nn.MaxPool2d(3, 2, padding=1),
            nn.BatchNorm2d(32, eps=1e-3),
            nn.ReLU(inplace=True)
        )
        self.conv_2 = nn.Sequential(
            ConvLayer(32, 32, 5, 1),
            nn.MaxPool2d(3, 2, padding=1),
            nn.BatchNorm2d(32, eps=1e-3),
            nn.ReLU(inplace=True)
        )
        self.conv_3 = nn.Sequential(
            ConvLayer(32, 64, 5, 1),
            nn.MaxPool2d(3, 2, padding=1),
            nn.BatchNorm2d(64, eps=1e-3),
            nn.ReLU(inplace=True),
            SEBlock(64)
        )
        self.conv_4 = nn.Sequential(
            ConvLayer(64, 128, 5, 1),
            nn.MaxPool2d(3, 2, padding=1),
            nn.BatchNorm2d(128, eps=1e-3),
            nn.ReLU(inplace=True),
            SEBlock(128)
        )
        self.haar = HaarWaveletBlock()
        self.fc = nn.Linear(128, 4)
        nn.init.xavier_uniform_(self.fc.weight)
        nn.init.constant_(self.fc.bias, 0.01)
        # self.maxpool = nn.AdaptiveMaxPool2d((1, 1))

    def forward(self, x):
        # print(x.shape)
        x = self.conv_1(x)
        x = self.conv_2(x)
        x = self.conv_3(x)
        x = self.conv_4(x)
        x = self.haar(x)
        x = self.fc(x)
        return x


class AttentionNet(nn.Module):
    """
    Attention-Enhanced CNN for Chinese Calligraphy Styles Classification
    """
    def __init__(self):
        super(AttentionNet, self).__init__()
        self.conv_1 = nn.Sequential(
            ConvLayer(3, 32, 5, 1),
            nn.MaxPool2d(3, 2, padding=1),
            nn.BatchNorm2d(32, eps=1e-3),
            nn.ReLU(inplace=True)
        )
        self.conv_2 = nn.Sequential(
            ConvLayer(32, 32, 5, 1),
            nn.MaxPool2d(3, 2, padding=1),
            nn.BatchNorm2d(32, eps=1e-3),
            nn.ReLU(inplace=True)
        )
        self.conv_3 = nn.Sequential(
            ConvLayer(32, 64, 5, 1),
            nn.MaxPool2d(3, 2, padding=1),
            nn.BatchNorm2d(64, eps=1e-3),
            nn.ReLU(inplace=True),
            CAM(64),
            SAM()
        )
        self.conv_4 = nn.Sequential(
            ConvLayer(64, 128, 5, 1),
            nn.MaxPool2d(3, 2, padding=1),
            nn.BatchNorm2d(128, eps=1e-3),
            nn.ReLU(inplace=True),
            CAM(128),
            SAM()
        )
        self.haar = HaarWaveletBlock()
        self.fc = nn.Linear(128, 5)
        nn.init.xavier_uniform_(self.fc.weight)
        nn.init.constant_(self.fc.bias, 0.01)

    def forward(self, x):
        x = self.conv_1(x)
        x = self.conv_2(x)
        x = self.conv_3(x)
        x = self.conv_4(x)
        x = self.haar(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x


class SiameseNet(nn.Module):

    def __init__(self):
        super(SiameseNet, self).__init__()
        self.net = BasicNet()

    def forward(self, x1, x2):
        output1 = self.net(x1)
        output2 = self.net(x2)
        return output1, output2

    def get_feature(self, x):
        x = self.net(x)
        return x


class SEVGG(nn.Module):

    def __init__(self, model=models.vgg16(pretrained=True)):
        super(SEVGG, self).__init__()
        # print(self.net)
        # self.net = model
        self.conv1 = nn.Sequential(*list(model.children())[0][0:4],
                                   SEBlock(64))
        self.conv2 = nn.Sequential(*list(model.children())[0][5:9],
                                   SEBlock(128))
        self.conv3 = nn.Sequential(*list(model.children())[0][10:16],
                                   SEBlock(256))
        self.conv4 = nn.Sequential(*list(model.children())[0][17:23],
                                   SEBlock(512))
        self.conv5 = nn.Sequential(*list(model.children())[0][24:30],
                                   SEBlock(512))
        self.avgpool = model.avgpool
        self.haar = HaarWaveletBlock()
        self.fc = nn.Sequential(nn.Linear(in_features=512, out_features=4096),
                                *list(model.children())[2][1:5],
                                nn.Linear(in_features=4096, out_features=4))

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)
        x = self.avgpool(x)
        x = self.haar(x)

        x = self.fc(x)
        return x


class SEBasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(SEBasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=3, stride=stride, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.se = SEBlock(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.se(out)

        if self.downsample is not None:
            residual = self.downsample(x)
        print(out.shape, residual.shape)
        out += residual
        out = self.relu(out)

        return out


class SEBottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(SEBottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * 4)
        self.relu = nn.ReLU(inplace=True)
        # SE
        # self.global_pool = nn.AdaptiveAvgPool2d(1)
        # self.conv_down = nn.Conv2d(
        #     planes * 4, planes // 4, kernel_size=1, bias=False)
        # self.conv_up = nn.Conv2d(
        #     planes // 4, planes * 4, kernel_size=1, bias=False)
        # self.sig = nn.Sigmoid()
        # self.se = SEBlock(planes * 4).
        self.cam = CAM(planes * 4)
        self.sam = SAM()
        # Downsample
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        # out1 = self.global_pool(out)
        # out1 = self.conv_down(out1)
        # out1 = self.relu(out1)
        # out1 = self.conv_up(out1)
        # out1 = self.sig(out1)
        out = self.cam(out)
        out = self.sam(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        res = out + residual
        res = self.relu(res)

        return res


class SEResNet(nn.Module):

    def __init__(self, block, layers, num_classes=4):
        self.inplanes = 64
        super(SEResNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        self.avgpool = nn.AvgPool2d(7)
        self.haar = HaarWaveletBlock()
        self.fc = nn.Linear(512 * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = self.haar(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)

        return x


class VGG16(nn.Module):
    def __init__(self, net=models.vgg16_bn()):
        super(VGG16, self).__init__()
        # self.net = net
        self.conv = net.features
        # self.avgpool = net.avgpool
        self.haar = HaarWaveletBlock()
        self.classifier = nn.Sequential(
            nn.Linear(512, 128, bias=True),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5, inplace=False),
            nn.Linear(128, 32, bias=True),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5, inplace=False),
            nn.Linear(32, 5, bias=True)
        )

    def forward(self, x):
        x = self.conv(x)
        # x = self.avgpool(x)
        x = self.haar(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x


if __name__ == '__main__':
    net = models.vgg16_bn()
    print(net)
