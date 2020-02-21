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

__all__ = ['resnet_gate']

class GatedSlimmableConv_BN(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size=3, stride=1, padding=1, bias=False):
        super(GatedSlimmableConv_BN, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride=stride, padding=padding)
        #super(GatedSlimmableConv2d, self).__init__( in_planes, out_planes,
        #    kernel_size, stride=stride, padding=padding, bias=bias
        #)
        self.bn = SlimmableBatchNorm2d(out_planes)

        self.in_planes = in_planes
        self.out_planes = out_planes
        self.stride = stride
        self.bias = bias
        self.padding = padding

        # add regurization for the gate, l1, l2 norm
        self.width_multi_list = get_value('width_multi_list')
        self.in_planes_list = []
        self.out_planes_list = []
        gate = []

        for _idx in range(len(self.width_multi_list)):
            _in_planes = int(self.width_multi_list[_idx]*in_planes)
            _out_planes = int(self.width_multi_list[_idx]*out_planes)
            self.in_planes_list.append( _in_planes )
            self.out_planes_list.append( _out_planes )
            # each gate correspond to unique switch path
            gates.append(nn.Linear(_in_planes, out_planes))

        self.gates = nn.ModuleList(gates)
        # init the parameters of gates
        for _idx in range(len(self.gates)):
            self.gates[_idx].weight.data.normal_(0, math.sqrt(2. / out_planes))
            nn.init.ones_(self.gates[_idx].bias)

    def regurizer(self, x):
        loss = torch.sum(torch.abs(x))
        return loss

    def forward(self, input_list):
        out = []
        for _idx in range(len(input_list)):
            _in_planes = self.in_planes_list[_idx]
            _out_planes = self.out_planes_list[_idx]
            weight = self.conv.weight[:_out_planes, :_in_planes, :, :]
            if self.bias is not None:
                bias = self.conv.bias[:_in_planes]
            else:
                bias = self.conv.bias
            _out = nn.functional.conv2d(input_list[_idx], weight, bias, self.stride, self.padding)
            out.append(_out)
        out = self.bn(out)
        return out

    def forward_gate(self, x_list):
        out = []
        r_loss = 0
        for _idx in range(len(x_list)):
            x = x_list[_idx]
            upsampled = F.avg_pool2d(torch.abs(x), x.shape[2])
            ss = upsampled.view(x.shape[0], x.shape[1])
            o_gate = self.gates[_idx](ss.detach())
            o_gate = 1.5*F.sigmoid(o_gate)
            rloss = self.regurizer(o_gate)

            #pdb.set_trace()
            index = torch.ones(o_gate.size()).cuda()
            inactive_channels = int(self.out_channels - round(self.out_channels * self.width_multi_list[_idx]))
            if inactive_channels > 0:
                inactive_idx = (-o_gate).topk(inactive_channels, 1)[1]
                index.scatter_(1, inactive_idx, 0)
            
            x = self.conv(x)
            active_idx = (o_gate*index).unsqueeze(2).unsqueeze(3)
            x = active_idx * x
        x = self.bn(x)
        return x, rloss


class TorchGraph(object):
    def __init__(self):
        self._graph = {}

    def add_tensor_list(self, name):
        self._graph[name] = []

    def append_tensor(self, name, val):
        self._graph[name].append(val)

    def clear_tensor_list(self, name):
        self._graph[name].clear()

    def get_tensor_list(self, name):
        return self._graph[name]

    def set_global_var(self, name, val):
        self._graph[name] = val

    def get_global_var(self, name):
        return self._graph[name]


_Graph = TorchGraph()
_Graph.add_tensor_list('gate_values')

regurize_loss_sum = 0

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
        out= self.conv1_bn(x)
        out = self.relu(out)
        out= self.conv2_bn(out)
        if self.downsample is not None:
            residual = self.downsample(x)
        out += residual
        out = self.relu(out)
        return out

    def forward_gate(self, x):
        residual = x
        out, rloss = self.conv1_bn.forward_gate(x)
        out = self.relu(out)
        out, rloss = self.conv2_bn.forward_gate(out)
        if self.downsample is not None:
            residual = self.downsample(x)
        out += residual
        out = self.relu(out)
        return out, rloss


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None, gated=False):
        super(Bottleneck, self).__init__()
        self.conv1_bn = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2_bn = GatedConv_BN(planes, planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * 4)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride


    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2_bn(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)
        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        #out = self.relu(out)
        return out

    def forward_gate(self, x):
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out, rloss = self.conv2_bn.forward_gate(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)
        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        #out = self.relu(out)
        return out, rloss



class ResNet(nn.Module):

    def __init__(self, depth, num_classes=100, gated=False, ratio=1.0):
        super(ResNet, self).__init__()
        # Model type specifies number of layers for CIFAR-10 model
        assert (depth - 2) % 6 == 0, 'depth should be 6n+2'
        n = (depth - 2) // 6

        _Graph.set_global_var('ratio', ratio)

        self.gated = gated

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
        if self.gated:
            return self.forward_gate(x)
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

    def forward_gate(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)    # 32x32

        regurize_loss_sum = 0
        for i in range(len(self.layer1)):
            x, rloss = self.layer1[i].forward_gate(x)
            regurize_loss_sum += rloss
        for i in range(len(self.layer2)):
            x, rloss = self.layer2[i].forward_gate(x)
            regurize_loss_sum += rloss
        for i in range(len(self.layer3)):
            x, rloss = self.layer3[i].forward_gate(x)
            regurize_loss_sum += rloss

        #x, rloss = self.layer1(x)  # 32x32
        #x, rloss = self.layer2(x)  # 16x16
        #x, rloss = self.layer3(x)  # 8x8

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        if not self.training:
            return x
        return x, regurize_loss_sum

def resnet_gate(depth, num_classes=100, gated=False, ratio=1.0):
    """
    Constructs a ResNet model.
    """
    return ResNet(depth, num_classes=num_classes, gated=gated, ratio=ratio)
