from turtle import forward
import torch.nn as nn
from torch.hub import load_state_dict_from_url
from torch.nn import functional as F

from triplet_net.inception_resnetv1 import InceptionResnetV1
from triplet_net.mobilenet import MobileNetV1
from triplet_net.effinet import efficientnetv2_s
from triplet_net.convnet import convnext_tiny
from triplet_net.paper_res2net1 import PaperMethod


class effinet(nn.Module):
    def __init__(self, pretrainde) -> None:
        super(effinet, self).__init__()
        self.model = efficientnetv2_s()

    def forward(self, x):
        x = self.model.stem(x)
        x = self.model.blocks(x)
        return x


class Paper(nn.Module):
    def __init__(self) -> None:
        super(Paper, self).__init__()
        self.model = PaperMethod()

    def forward(self, x):
        x = self.model.forward(x)
        return x


class ConvNet(nn.Module):
    def __init__(self, num_classes):
        super(ConvNet, self).__init__()
        self.model = convnext_tiny(num_classes)

        del self.model.head

    def forward(self, x):
        x = self.model.forward_features(x)
        return x


class mobilenet(nn.Module):
    def __init__(self, pretrained):
        super(mobilenet, self).__init__()
        self.model = MobileNetV1()
        if pretrained:
            state_dict = load_state_dict_from_url(
                "https://github.com/bubbliiiing/facenet-pytorch/releases/download/v1.0/backbone_weights_of_mobilenetv1.pth",
                model_dir="model_data",
                progress=True)
            self.model.load_state_dict(state_dict)

        del self.model.fc
        del self.model.avg

    def forward(self, x):
        x = self.model.stage1(x)
        x = self.model.stage2(x)
        x = self.model.stage3(x)
        return x


class inception_resnet(nn.Module):
    def __init__(self, pretrained):
        super(inception_resnet, self).__init__()
        self.model = InceptionResnetV1()
        if pretrained:
            state_dict = load_state_dict_from_url(
                "https://github.com/bubbliiiing/facenet-pytorch/releases/download/v1.0/backbone_weights_of_inception_resnetv1.pth",
                model_dir="model_data",
                progress=True)
            self.model.load_state_dict(state_dict)

    def forward(self, x):
        x = self.model.conv2d_1a(x)
        x = self.model.conv2d_2a(x)
        x = self.model.conv2d_2b(x)
        x = self.model.maxpool_3a(x)
        x = self.model.conv2d_3b(x)
        x = self.model.conv2d_4a(x)
        x = self.model.conv2d_4b(x)
        x = self.model.repeat_1(x)
        x = self.model.mixed_6a(x)
        x = self.model.repeat_2(x)
        x = self.model.mixed_7a(x)
        x = self.model.repeat_3(x)
        x = self.model.block8(x)
        return x


class NormalNet(nn.Module):
    """
    单loss训练模型
    """

    def __init__(self, backbone="mobilenet", dropout_keep_prob=0.5, embedding_size=128, num_classes=None, mode="train",
                 pretrained=False):
        super(NormalNet, self).__init__()
        if backbone == "mobilenet":
            self.backbone = mobilenet(pretrained)
            flat_shape = 1024
        elif backbone == "inception_resnetv1":
            self.backbone = inception_resnet(pretrained)
            flat_shape = 1792
        elif backbone == "effinet":
            self.backbone = effinet(pretrained)
            flat_shape = 256
        elif backbone == "conv_net":
            self.backbone = ConvNet(num_classes)
            flat_shape = 768
        elif backbone == "paper":
            self.backbone = Paper()
            flat_shape = 512
        else:
            raise ValueError('Unsupported backbone - `{}`, Use mobilenet, inception_resnetv1.'.format(backbone))
        self.avg = nn.AdaptiveAvgPool2d((1, 1))
        self.Dropout = nn.Dropout(1 - dropout_keep_prob)
        self.Bottleneck = nn.Linear(flat_shape, embedding_size, bias=False)
        self.last_bn = nn.BatchNorm1d(embedding_size, eps=0.001, momentum=0.1, affine=True)
        self.classifier = nn.Linear(embedding_size, num_classes)

    def forward(self, x):
        x = self.backbone(x)
        x = self.avg(x)
        x = x.view(x.size(0), -1)
        x = self.Dropout(x)
        x = self.Bottleneck(x)
        before_normalize = self.last_bn(x)
        # x = F.normalize(before_normalize, p=2, dim=1)
        cls = self.classifier(before_normalize)
        return cls

    def forward_feature(self, x):
        x = self.backbone(x)
        x = self.avg(x)
        x = x.view(x.size(0), -1)
        x = self.Dropout(x)
        x = self.Bottleneck(x)
        before_normalize = self.last_bn(x)
        x = F.normalize(before_normalize, p=2, dim=1)
        return before_normalize, x

    def forward_classifier(self, x):
        x = self.classifier(x)
        return x
