import torch.nn as nn
import math
import torch
import numpy as np
import torch.nn.functional as F
import random
from torch.autograd import Variable
import os
import pdb

__all__ = ['vgg_rnp_rl']

class pruning_model(nn.Module):
    def __init__(self, extra_layers_encode):
        super(pruning_model, self).__init__()
        self.extra_layers_encode = extra_layers_encode
        self.rnncell = nn.GRUCell(256, 4, bias=True)
        self._initialize_weights()
        self.relu = nn.ReLU()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.weight.data.normal_(0, 0.01)
                m.bias.data.zero_()

    def forward(self, x, h, ct):
        encode = self.extra_layers_encode[2*ct]
        x = encode(x)
        return self.rnncell(x, h)


class vgg_RPN(nn.Module):
    def __init__(self, num_classes=100, greedyP=0.9, group_num=4):
        super(vgg_RPN, self).__init__()

        self.cfg = [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M']
        self.group_num=group_num
        conv_layers, extra_layers_encode = self._make_layers(self.cfg)
        self.conv_layers = conv_layers
        self.extra_layers_encode = extra_layers_encode
        self.classifier = nn.Sequential(
            nn.Linear(512, 512),
            nn.ReLU(True),
            nn.Dropout(p=0.5),
            nn.Linear(512, num_classes),
        )
        self.rnncell = nn.GRUCell(256, 4, bias=True)
        self._initialize_weights()
        self.pnet = pruning_model(self.extra_layers_encode)
        # mode: 0 for VGG baseline, 1 for random pruning, 2 for RNP training
        self.group = []
        self.greedyP = greedyP

    def divide_conv(self, group_num=4):
        for layer in self.conv_layers:
            if isinstance(layer, nn.Conv2d):
                weights = layer.weight.data
                weights = weights.view(weights.size(0), weights.size(1), -1)
                norm = torch.mean(torch.norm(weights,2,2), 1)
                order = torch.argsort(norm).cuda()
                glen = int(order.shape[0] / group_num)
                group_index = order[-1] * torch.ones((group_num, order.shape[0]), dtype=torch.long).cuda()
                for _idx in range(group_num):
                    group_index[_idx, 0:(_idx+1)*glen] = torch.sort(order[( group_num - _idx - 1 )*glen:])[0].cuda()
                self.group += [group_index]

    def forward(self, x):
        y = []
        ct = 0
        bs = x.size(0)
        former_state = Variable(torch.zeros(bs, 4)).cuda()
        random_explo = int(bs*(1-self.greedyP))
        for layer in self.conv_layers:
            if isinstance(layer, nn.Conv2d) and ct > 0:
                x_pool = x.mean(3).mean(2)
                x = layer(x)
                mask = torch.zeros(x.size(0), x.size(1)).cuda()
                h = self.pnet(x_pool, former_state, ct)
                former_state = h
                # exploration using random select
                exploration_select = torch.randint(0, self.group_num, [random_explo], dtype=torch.long).cuda()
                mask[0:random_explo,:].scatter_(1, self.group[ct][exploration_select, :], 1)

                # exploitation using greedy strategy
                exploitation_select = torch.argmax(h[random_explo:], 1)
                mask[exploitation_select, :].scatter_(1, self.group[ct][exploitation_select, :], 1)

                mask = Variable(mask, requires_grad=False).cuda()
                x = mask.unsqueeze(2).unsqueeze(3)*x
                #pdb.set_trace()
                y += [[h,  torch.cat((exploration_select, exploitation_select), 0)]]
                ct += 1
            elif isinstance(layer, nn.Conv2d) and ct == 0:
                x = layer(x)
                ct += 1
            else:
                x = layer(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x, y
    
    def _make_layers(self, cfg, batch_norm=False):
        layers = []
        extra_layers_encode = []
        in_channels = 3
        for v in cfg:
            if v == 'M':
                layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
            else:
                conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
                layers += [conv2d, nn.ReLU()]
                extra_layers_encode += [nn.Linear(in_channels, 256), nn.ReLU()]
                in_channels = v
        layers = nn.ModuleList(layers)
        extra_layers_encode = nn.ModuleList(extra_layers_encode)
        return layers, extra_layers_encode


    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.weight.data.normal_(0, 0.01)
                m.bias.data.zero_()
        for m in self.conv_layers:
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()     

def vgg_rnp_rl(num_classes=100):
    """
    Constructs a vgg_RNP model with reinforcement learning training.
    """
    net = vgg_RPN(num_classes=num_classes)
    return net
