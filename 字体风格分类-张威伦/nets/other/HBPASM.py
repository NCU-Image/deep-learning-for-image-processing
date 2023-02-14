import math
import re
from collections import OrderedDict

import torch
from torch import nn

model_urls = {
    'resnet34': 'https://download.pytorch.org/models/resnet34-333f7ec4.pth',
}


def conv3x3(in_planes, out_planes, stride=1):
    # "3x3 convolution with padding"
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        m = OrderedDict()
        m['conv1'] = conv3x3(inplanes, planes, stride)
        m['bn1'] = nn.BatchNorm2d(planes)
        m['relu1'] = nn.ReLU(inplace=True)
        m['conv2'] = conv3x3(planes, planes)
        m['bn2'] = nn.BatchNorm2d(planes)
        self.group1 = nn.Sequential(m)

        self.relu = nn.Sequential(nn.ReLU(inplace=True))
        self.downsample = downsample

    def forward(self, x):
        if self.downsample is not None:
            residual = self.downsample(x)
        else:
            residual = x

        out = self.group1(x) + residual

        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        m = OrderedDict()
        m['conv1'] = nn.Conv2d(inplanes, planes, kernel_size=1, bias=True)
        m['bn1'] = nn.BatchNorm2d(planes)
        m['relu1'] = nn.ReLU(inplace=True)
        m['conv2'] = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=True)
        m['bn2'] = nn.BatchNorm2d(planes)
        m['relu2'] = nn.ReLU(inplace=True)
        m['conv3'] = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=True)
        m['bn3'] = nn.BatchNorm2d(planes * 4)
        self.group1 = nn.Sequential(m)

        self.relu = nn.Sequential(nn.ReLU(inplace=True))
        self.downsample = downsample

    def forward(self, x):
        if self.downsample is not None:
            residual = self.downsample(x)
        else:
            residual = x

        out = self.group1(x) + residual
        out = self.relu(out)

        return out


class ResNet(nn.Module):
    def __init__(self, block, layers):
        self.inplanes = 64
        super(ResNet, self).__init__()

        m = OrderedDict()
        m['conv1'] = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=True)
        m['bn1'] = nn.BatchNorm2d(64)
        m['relu1'] = nn.ReLU(inplace=True)
        m['maxpool'] = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.group1 = nn.Sequential(m)

        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)

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
                nn.Conv2d(self.inplanes, planes * block.expansion, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.group1(x)

        fea1 = self.layer1(x)
        fea2 = self.layer2(fea1)
        fea3 = self.layer3(fea2)
        fea4_1 = self.layer4[0:1](fea3)
        fea4_2 = self.layer4[1:2](fea4_1)
        fea4_3 = self.layer4[2:3](fea4_2)

        return fea4_1, fea4_2, fea4_3


def load_state_dict(model, model_root):
    own_state_old = model.state_dict()
    own_state = OrderedDict()  # remove all 'group' string
    for k, v in own_state_old.items():
        k = re.sub('group\d+\.', '', k)
        own_state[k] = v

    state_dict = torch.load(model_root)

    for name, param in state_dict.items():
        if name not in own_state:
            if 'fc' in name:
                continue
            print(own_state.keys())
            raise KeyError('unexpected key "{}" in state_dict'
                           .format(name))
        if isinstance(param, nn.Parameter):
            # backwards compatibility for serialized parameters
            param = param.data
        own_state[name].copy_(param)


def resnet34(pretrained=False, model_root=None):
    model = ResNet(BasicBlock, [3, 4, 6, 3])
    if pretrained:
        load_state_dict(model, model_root=model_root)
    return model




def mask_Generation(feature, alpha):
    # B N h w

    batch_size = feature.size(0)
    kernel = feature.size(2)

    # detach()分离tensor

    sum = torch.sum(feature.detach(), dim=1)

    avg = torch.sum(torch.sum(sum, dim=1), dim=1) / kernel ** 2

    mask = torch.where(sum > alpha * avg.view(batch_size, 1, 1),

                       torch.ones(sum.size()),
                       (torch.zeros(sum.size()) + 0.1))

    mask = mask.unsqueeze(1)
    return mask


class Net(nn.Module):
    def __init__(self, model_path):
        super(Net, self).__init__()

        self.proj0 = nn.Conv2d(512, 8192, kernel_size=1, stride=1)
        self.proj1 = nn.Conv2d(512, 8192, kernel_size=1, stride=1)
        self.proj2 = nn.Conv2d(512, 8192, kernel_size=1, stride=1)

        self.bn0 = nn.BatchNorm2d(512)
        self.bn1 = nn.BatchNorm2d(512)
        self.bn2 = nn.BatchNorm2d(512)

        # fc layer
        self.fc_concat = torch.nn.Linear(8192 * 3, 4)

        self.softmax = nn.LogSoftmax(dim=1)
        self.avgpool = nn.AvgPool2d(kernel_size=2)

        # base-model
        self.features = resnet34(pretrained=False, model_root=model_path)

    def forward(self, x):
        batch_size = x.size(0)
        feature4_0, feature4_1, feature4_2 = self.features(x)

        slack_mask1 = mask_Generation(feature4_0, alpha=0.6)
        slack_mask2 = mask_Generation(feature4_1, alpha=0.6)
        slack_mask3 = mask_Generation(feature4_2, alpha=0.6)

        Aggregated_mask = slack_mask1 * slack_mask2 * slack_mask3

        feature4_0 = self.bn0(feature4_0 * Aggregated_mask)
        feature4_1 = self.bn1(feature4_1 * Aggregated_mask)
        feature4_2 = self.bn2(feature4_2 * Aggregated_mask)

        feature4_0 = self.proj0(feature4_0)
        feature4_1 = self.proj1(feature4_1)
        feature4_2 = self.proj2(feature4_2)

        inter1 = feature4_0 * feature4_1
        inter2 = feature4_0 * feature4_2
        inter3 = feature4_1 * feature4_2
        # print(inter1.shape)

        inter1 = self.avgpool(inter1).view(batch_size, -1)
        inter2 = self.avgpool(inter2).view(batch_size, -1)
        inter3 = self.avgpool(inter3).view(batch_size, -1)

        result1 = torch.nn.functional.normalize(torch.sign(inter1) * torch.sqrt(torch.abs(inter1) + 1e-10))
        result2 = torch.nn.functional.normalize(torch.sign(inter2) * torch.sqrt(torch.abs(inter2) + 1e-10))
        result3 = torch.nn.functional.normalize(torch.sign(inter3) * torch.sqrt(torch.abs(inter3) + 1e-10))

        result = torch.cat((result1, result2, result3), 1)
        result = self.fc_concat(result)
        # print(result.shape)
        return self.softmax(result)
