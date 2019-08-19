from __future__ import division
""" 
Creates a ResNeXt Model as defined in:
Xie, S., Girshick, R., Dollar, P., Tu, Z., & He, K. (2016). 
Aggregated residual transformations for deep neural networks. 
arXiv preprint arXiv:1611.05431.
import from https://github.com/facebookresearch/ResNeXt/blob/master/models/resnext.lua
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

import pdb

__all__ = ['resnet_mb']

class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion*planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_planes, planes, stride=1):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, self.expansion*planes, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(self.expansion*planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion*planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class ResNet(nn.Module):
    def __init__(self, block, num_blocks, num_classes=10, num_branch=4):
        super(ResNet, self).__init__()
        self.in_planes = 64
        self.branch_planes = 0
        self.num_branch=num_branch

        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=2)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)

        layer4 = []
        linear = []
        # n copies of block4 
        for i in range(self.num_branch):
            layer4.append(  self._make_layer(block, 512, num_blocks[3], stride=2, is_branch=True).cuda() )
            linear.append(nn.Linear(512*block.expansion, num_classes).cuda() )
        self.layer4 = nn.ModuleList(layer4)        
        self.linear = nn.ModuleList(linear)        

    def _make_layer(self, block, planes, num_blocks, stride, is_branch=False):
        if is_branch :
            self.in_planes = self.branch_planes

        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        if is_branch == False:
            self.branch_planes = self.in_planes
        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        
        outs = []
        for i in range( self.num_branch ):
            temp = self.layer4[i](out)
            temp = F.avg_pool2d(temp, 14)
            temp = temp.view(temp.size(0), -1)
            temp = self.linear[i](temp)
            outs.append( temp )
        return outs, out


def ResNet18_mb(num_classes=1000, num_branch=1):
    return ResNet(BasicBlock, [2,2,2,2], num_classes=num_classes, num_branch=num_branch)

def ResNet34_mb(num_classes=1000, num_branch=1):
    return ResNet(BasicBlock, [3,4,6,3], num_classes=num_classes, num_branch=num_branch)

def ResNet50_mb(num_classes=1000, branch_num=1):
    return ResNet(Bottleneck, [3,4,6,3], num_classes=num_classes, num_branch=num_branch)

def ResNet101_mb(num_classes=1000, num_branch=1):
    return ResNet(Bottleneck, [3,4,23,3], num_classes=num_classes, num_branch=num_branch)

def ResNet152_mb(num_classes=1000, num_branch=1):
    return ResNet(Bottleneck, [3,8,36,3], num_classes=num_classes, num_branch=num_branch)


def resnet_mb(num_classes=1000, depth=18, num_branch=1):
    if depth == 18:
        return ResNet18_mb(num_classes=num_classes, num_branch=num_branch)
    elif depth == 34:
        return ResNet34_mb(num_classes=num_classes, num_branch=num_branch)
    elif depth == 50:
        return ResNet50_mb(num_classes=num_classes, num_branch=num_branch)
    elif depth == 101:
        return ResNet101_mb(num_classes=num_classes, num_branch=num_branch)
    elif depth == 152:
        return ResNet152_mb(num_classes=num_classes, num_branch=num_branch)

# def test():
#     net = ResNet18_mb()
#     y = net(torch.randn(1,3,32,32))
#     print(y.size())
