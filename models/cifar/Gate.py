'''
Training script for CIFAR-10/100
Copyright (c) Baoyun Peng, 2018
'''

import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = ['Gate']

class Gate(nn.Module):
    def __init__(self, in_planes=1024, out_planes=1, d=28):
        super(Gate, self).__init__()
        self.in_planes  = in_planes
        self.out_planes = out_planes
        self.avgpool = nn.AvgPool2d(d)
        self.fc = nn.Linear(self.in_planes, self.out_planes)
        # self.fc = nn.Linear(128, 3)
        self.bn = nn.BatchNorm1d(self.out_planes)
        self.softmax = nn.Softmax()
    def forward(self, x):
        #print(x.size())
        x = self.avgpool(x)
        x = x.view( x.size(0), -1)
        out = self.fc(x)
        out = F.relu(self.bn(out))
        out = self.softmax(out)
        return out
#class Gate(nn.Module):
#    def __init__(self, in_planes=1024, out_planes=1, d=28):
#        super(Gate, self).__init__()
#        self.in_planes  = in_planes
#        self.out_planes = out_planes
#        self.avgpool = nn.AvgPool2d(d)
#        self.fc = nn.Linear(self.in_planes, self.out_planes)
#        # self.fc = nn.Linear(128, 3)
#        self.bn = nn.BatchNorm1d(self.out_planes)
#        self.softmax = nn.Softmax()
#    def forward(self, x):
#        #print(x.size())
#        x = self.avgpool(x)
#        x = x.view( x.size(0), -1)
#        out = self.fc(x)
#        out = F.relu(self.bn(out))
#        out = self.softmax(out)
#        return out
