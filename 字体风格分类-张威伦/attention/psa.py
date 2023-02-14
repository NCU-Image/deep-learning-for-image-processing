import torch
import torch.nn as nn


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