from __future__ import absolute_import

'''Resnet for cifar dataset.
Ported form
https://github.com/facebook/fb.resnet.torch
and
https://github.com/pytorch/vision/blob/master/torchvision/models/resnet.py
(c) Baoyun Peng
'''
import torch.nn as nn
import torch
import pdb
import torch.nn.functional as F
import math

from models.cifar.slimmable_ops import slimmable_init, set_value, get_value

__all__ = ['resnet_channel_rdi']

class GatedConv_BN(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1):
        super(GatedConv_BN, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size,
                              stride=stride, padding=padding)
        self.bn = nn.BatchNorm2d(out_channels)
        gate = []
        # Gate layers
        self.gate = nn.Sequential(nn.Linear(in_channels, 10),
                            nn.ReLU(),
                            nn.Linear(10, out_channels))
        for m in self.modules():
            if isinstance(m, nn.Linear):
                m.weight.data.normal_(0, math.sqrt(2. / m.weight.data.size(-1)))
                m.bias.data.zero_()

    def forward(self, x):
        upsampled = F.avg_pool2d(torch.abs(x), x.shape[2])
        ss = upsampled.view(x.shape[0], x.shape[1])
        gate_out = self.gate(ss.detach())
        gate_out = 1.5*F.sigmoid(gate_out)

        index = torch.ones(gate_out.size()).cuda()
        inactive_channels = int(self.conv.out_channels - round(self.conv.out_channels * get_value('ratio')))
        inactive_idx = (-gate_out).topk(inactive_channels, 1)[1]
        #pdb.set_trace()
        index.scatter_(1, inactive_idx, 0)
        active_idx = (gate_out*index).unsqueeze(2).unsqueeze(3)

        x = self.conv(x)
        x = self.bn(x)
        x = x * active_idx
        return x


def conv3x3_BN(in_planes, out_planes, stride=1):
    return GatedConv_BN(in_planes, out_planes, kernel_size=3, stride=stride, padding=1)
    "3x3 convolution with padding"

class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1_bn = conv3x3_BN(inplanes, planes, stride)
        self.relu = nn.ReLU(inplace=True)
        self.conv2_bn = conv3x3_BN(planes, planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x
        exec_probs_list = []
        out= self.conv1_bn(x)
        out = self.relu(out)
        out = self.conv2_bn(out)
        if self.downsample is not None:
            residual = self.downsample(x)
        out += residual
        out = self.relu(out)
        return out

class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None, gated=False):
        super(Bottleneck, self).__init__()
        self.conv1_bn = GatedConv_BN(inplanes, planes, kernel_size=1, bias=False)
        self.conv2_bn = GatedConv_BN(planes, planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.conv3_bn = GatedConv_BN(planes, planes * 4, kernel_size=1, bias=False)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x
        out = self.conv1_bn(x)
        out = self.relu(out)

        out = self.conv2_bn(out)
        out = self.relu(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        #out = self.relu(out)
        return out


class ResNet(nn.Module):

    def __init__(self, depth, num_classes=100, gated=False, ratio=1.0):
        super(ResNet, self).__init__()
        # Model type specifies number of layers for CIFAR-10 model
        assert (depth - 2) % 6 == 0, 'depth should be 6n+2'
        n = (depth - 2) // 6

        block = Bottleneck if depth >=44 else BasicBlock

        self.inplanes = 16
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, padding=1,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(16)
        self.relu = nn.ReLU(inplace=True)
        self.layer1 = self._make_layer(block, 16, n)
        self.layer2 = self._make_layer(block, 32, n, stride=2)
        self.layer3 = self._make_layer(block, 64, n, stride=2)
        self.avgpool = nn.AvgPool2d(8)
        self.fc = nn.Linear(64 * block.expansion, num_classes)

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
                GatedConv_BN(self.inplanes, planes * block.expansion,
                            kernel_size=1, stride=stride, padding=0)
                #nn.Conv2d(self.inplanes, planes * block.expansion,
                #          kernel_size=1, stride=stride, bias=False),
                #nn.BatchNorm2d(planes * block.expansion),
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
        x = self.layer1(x)  # 32x32
        x = self.layer2(x)  # 16x16
        x = self.layer3(x)  # 8x8

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x


def resnet_channel_rdi(depth, num_classes=100, ratio=1.0):
    """
    Constructs a ResNet model.
    """
    slimmable_init()
    set_value('ratio', ratio)
    return ResNet(depth, num_classes=num_classes)
