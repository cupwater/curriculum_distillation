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

from models.cifar.slimmable_ops import *

__all__ = ['resnet_1multi_gate', 'resnet_2multi_gate', 'resnet_3multi_gate', 'resnet_4multi_gate']

slimmable_init()
class GatedSlimmableConv_BN(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size=3, stride=1, padding=0, bias=False):
        super(GatedSlimmableConv_BN, self).__init__()
        self.conv = nn.Conv2d(in_planes, out_planes, kernel_size, stride=stride, padding=padding)

        self.in_planes = in_planes
        self.out_planes = out_planes
        self.stride = stride
        self.bias = bias
        self.padding = padding
        # add regurization for the gate, l1, l2 norm
        self.width_multi_list = get_value('width_multi_list')
        bns = []
        # adding unique batchnorm for each path
        for _idx in range(len(self.width_multi_list)):
            bns.append(nn.BatchNorm2d(out_planes))
        self.bn = nn.ModuleList(bns)

        
        self.in_planes_list = []
        self.out_planes_list = []
        gates = []
        # eadding unique gate for each path
        for _idx in range(len(self.width_multi_list)):
            _in_planes = int(self.width_multi_list[_idx]*in_planes)
            _out_planes = int(self.width_multi_list[_idx]*out_planes)
            self.in_planes_list.append( _in_planes )
            self.out_planes_list.append( _out_planes )
            gates.append(nn.Linear(in_planes, out_planes))
        self.gates = nn.ModuleList(gates)
        
        # init the parameters of gates
        for _idx in range(len(self.gates)):
            self.gates[_idx].weight.data.normal_(0, math.sqrt(2. / out_planes))
            nn.init.ones_(self.gates[_idx].bias)

    def regurizer(self, x):
        loss = torch.sum(torch.abs(x))
        return loss

    def forward(self, x_list):
        out = []
        rloss = 0
        for _idx in range(len(x_list)):
            x = x_list[_idx]
            upsampled = F.avg_pool2d(torch.abs(x), x.shape[2])
            ss = upsampled.view(x.shape[0], x.shape[1])
            o_gate = self.gates[_idx](ss.detach())
            o_gate = 1.5*F.sigmoid(o_gate)
            rloss += self.regurizer(o_gate)

            index = torch.ones(o_gate.size()).cuda()
            inactive_channels = int(self.out_planes - round(self.out_planes * self.width_multi_list[_idx]))
            if inactive_channels > 0:
                inactive_idx = (-o_gate).topk(inactive_channels, 1)[1]
                index.scatter_(1, inactive_idx, 0)
            
            x = self.conv(x)
            x = self.bn[_idx](x)
            active_idx = (o_gate*index).unsqueeze(2).unsqueeze(3)
            x = active_idx * x
            out.append(x)
        return out, rloss


def conv3x3_BN(in_planes, out_planes, stride=1):
    return GatedSlimmableConv_BN(in_planes, out_planes, kernel_size=3, stride=stride, padding=1)
    "3x3 convolution with padding"

class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1_bn = conv3x3_BN(inplanes, planes, stride)
        self.relu = SlimmableReLU()
        self.conv2_bn = conv3x3_BN(planes, planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x_list):
        residual = x_list
        out, rloss1 = self.conv1_bn(x_list)
        out = self.relu(out)
        out, rloss2 = self.conv2_bn(out)
        if self.downsample is not None:
            residual, rloss = self.downsample(x_list)
            rloss1 += rloss
        out_list = []
        for _idx, (_x, _y) in enumerate(zip(out, residual)):
            out_list.append(_x+_y)
        out_list = self.relu(out_list)
        return out_list, rloss1+rloss2

# currently the slimmable bottleneck is not implemented
class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1_bn = GatedSlimmableConv_BN(inplanes, planes, kernel_size=1, bias=False)
        self.conv2_bn = GatedSlimmableConv_BN(planes, planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.conv3_bn = GatedSlimmableConv_BN(planes, planes * 4, kernel_size=1, bias=False)
        self.relu = SlimmableReLU()
        self.downsample = downsample
        self.stride = stride

    def forward(self, x_list):
        residual = x_list
        out, rloss1 = self.conv1_bn(x_list)
        out = self.relu(out)
        out, rloss2 = self.conv2_bn(out)
        out = self.relu(out)
        out, rloss3 = self.conv3_bn(out)
        if self.downsample is not None:
            residual, rloss = self.downsample(x_list)
            rloss1 += rloss
        out_list = []
        for _idx, (_x, _y) in enumerate(zip(out, residual)):
            out_list.append(_x+_y)
        #out_list = self.relu(out_list)
        return out_list, rloss1+rloss2+rloss3


class ResNet(nn.Module):

    def __init__(self, depth, num_classes=100):
        super(ResNet, self).__init__()
        # Model type specifies number of layers for CIFAR-10 model
        assert (depth - 2) % 6 == 0, 'depth should be 6n+2'
        n = (depth - 2) // 6

        self.width_multi_list = get_value('width_multi_list')

        block = Bottleneck if depth >=44 else BasicBlock

        self.inplanes = 16
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, padding=1,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(16)
        self.relu = nn.ReLU(inplace=True)
        
        self.layer1 = self._make_layer(block, 16, n)
        self.layer2 = self._make_layer(block, 32, n, stride=2)
        self.layer3 = self._make_layer(block, 64, n, stride=2)
        self.avgpool = SlimmableAvgPool2d(8)
        fc = []
        for i in range(len(self.width_multi_list)):
            fc.append(nn.Linear(64*block.expansion, num_classes))
        self.fc = nn.ModuleList(fc)

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
            downsample =  GatedSlimmableConv_BN(self.inplanes, planes * block.expansion,
                            kernel_size=1, stride=stride, padding=0, bias=False)
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
        for i in range(len(get_value('width_multi_list'))):
            x_list.append(x)

        x = x_list
        regurize_loss_sum = 0
        for i in range(len(self.layer1)):
            x, rloss = self.layer1[i](x)
            regurize_loss_sum += rloss
        for i in range(len(self.layer2)):
            x, rloss = self.layer2[i](x)
            regurize_loss_sum += rloss
        for i in range(len(self.layer3)):
            x, rloss = self.layer3[i](x)
            regurize_loss_sum += rloss
        x = self.avgpool(x)
        out = []
        for _idx in range(len(x)):
            _x = x[_idx].view(x[_idx].size(0), -1)
            _x = self.fc[_idx](_x)
            out.append(_x)
        if not self.training:
            return out
        return out, regurize_loss_sum


def resnet_1multi_gate(depth, num_classes=100):
    """
    Constructs a ResNet model.
    """
    set_value('width_multi_list', [1.0])
    return ResNet(depth, num_classes=num_classes)


def resnet_3multi_gate(depth, num_classes=100):
    """
    Constructs a ResNet model.
    """
    set_value('width_multi_list', [1.0, 0.8, 0.6])
    return ResNet(depth, num_classes=num_classes)

def resnet_2multi_gate(depth, num_classes=100):
    set_value('width_multi_list', [1.0, 0.8])
    """
    Constructs a ResNet model.
    """
    return ResNet(depth, num_classes=num_classes)

def resnet_4multi_gate(depth, num_classes=100):
    set_value('width_multi_list', [1.0, 0.8, 0.6, 0.5])
    """
    Constructs a ResNet model.
    """
    return ResNet(depth, num_classes=num_classes)

