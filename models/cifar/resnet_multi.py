from __future__ import absolute_import

'''Resnet for cifar dataset.
Ported form
https://github.com/facebook/fb.resnet.torch
and
https://github.com/pytorch/vision/blob/master/torchvision/models/resnet.py
(c) Baoyun, Peng
'''
import torch.nn as nn
import math
from models.cifar.slimmable_ops import *
#from .slimmable_ops import SwitchableBatchNorm2d
#from .slimmable_ops import SlimmableConv2d, SlimmableLinear


__all__ = ['resnet_1multi', 'resnet_3multi', 'resnet_4multi', 'resnet_2multi']
slimmable_init()

def SwitchableConv3x3(in_planes, out_planes, stride=1):
    "3x3 convolution with padding"
    return SlimmableConv2d(in_planes, out_planes, 3, stride=stride, padding=1, bias=False)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, outplanes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = SwitchableConv3x3(inplanes, outplanes, stride)
        self.bn1 = SwitchableBatchNorm2d(outplanes)
        self.relu = SlimmableReLU()
        self.conv2 = SwitchableConv3x3(outplanes, outplanes)
        self.bn2 = SwitchableBatchNorm2d(outplanes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x_list):
        residual_list = x_list
        out_list = self.conv1(x_list)
        out_list = self.bn1(out_list)
        out_list = self.relu(out_list)

        out_list = self.conv2(out_list)
        out_list = self.bn2(out_list)

        if self.downsample is not None:
            residual_list = self.downsample(x_list)
        out = []
        for idx, (_x,_y) in enumerate(zip(out_list, residual_list)):
            out.append(_x + _y)
        out = self.relu(out)
        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = SlimmableConv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = SwitchableBatchNorm2d(planes)
        self.conv2 = SlimmableConv2d(planes, planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn2 = SwitchableBatchNorm2d(planes)
        self.conv3 = SlimmableConv2d(planes, planes * 4, kernel_size=1, bias=False)
        self.bn3 = SwitchableBatchNorm2d(planes * 4)
        self.relu = SlimmableReLU()
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

        if self.downsample is not None:
            residual = self.downsample(x)

        out_list = []
        for idx, (_x,_y) in enumerate(zip(out, residual)):
            out_list.append(_x + _y)
        #out = self.relu(out)
        return out_list


class ResNet(nn.Module):

    def __init__(self, depth, num_classes=100):
        super(ResNet, self).__init__()
        # Model type specifies number of layers for CIFAR-10 model
        assert (depth - 2) % 6 == 0, 'depth should be 6n+2'
        n = (depth - 2) // 6

        #block = Bottleneck if depth >=44 else BasicBlock
        block = BasicBlock

        self.inplanes = 16
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, padding=1,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(16)
        self.relu = nn.ReLU(inplace=True)
        self.layer1 = self._make_layer(block, 16, n)
        self.layer2 = self._make_layer(block, 32, n, stride=2)
        self.layer3 = self._make_layer(block, 64, n, stride=2)
        self.avgpool = SlimmableAvgPool2d(8)
        self.fc = SlimmableLinear(64 * block.expansion, num_classes)

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
                SlimmableConv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                SwitchableBatchNorm2d(planes * block.expansion),
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
        x = self.relu(x)    # 32x32
        x_list = []
        for width_multi in get_value('width_multi_list'):
            x_list.append(x[:, 0:int(width_multi*x.size(1)), :, :])
        x = self.layer1(x_list)  # 32x32
        x = self.layer2(x)  # 16x16
        x = self.layer3(x)  # 8x8

        x = self.avgpool(x)
        out = []
        for _x in x:
            _x = _x.view(_x.size(0), -1)
            out.append(_x)
        
        out = self.fc(out)

        return out

def resnet_1multi(**kwargs):
    """
    Constructs a ResNet model.
    """
    set_value('width_multi_list', [1.0])
    return ResNet(**kwargs)


def resnet_3multi(**kwargs):
    """
    Constructs a ResNet model.
    """
    set_value('width_multi_list', [1.0, 0.8, 0.6])
    return ResNet(**kwargs)

def resnet_2multi(**kwargs):
    set_value('width_multi_list', [1.0, 0.8])
    """
    Constructs a ResNet model.
    """
    return ResNet(**kwargs)

def resnet_4multi(**kwargs):
    set_value('width_multi_list', [1.0, 0.8, 0.6, 0.5])
    """
    Constructs a ResNet model.
    """
    return ResNet(**kwargs)
