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

__all__ = ['resnet_rnp_rl']


class BlockSkip(nn.Module):
    def __init__(self, block_inplanes_list):
        super(BlockSkip, self).__init__()
        self.block_inplanes_list = block_inplanes_list
        embedding_list = []
        for inplane in self.block_inplanes_list:
            embedding_list.append(nn.Linear(inplane, get_value('group_num')))
        self.embeddings = nn.ModuleList(embedding_list)
        self.rnncell = nn.GRUCell(get_value('group_num'), get_value('group_num'), bias=True)
        self._initialize_weights()
        self.relu = nn.ReLU()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                m.weight.data.normal_(0, 0.1)
                #m.bias.data = torch.ones(m.bias.data.size(0))

    def forward(self, x, former_state, block_idx):
        if not self.training:
            random_explo = 0
        else:
            random_explo = int(x.size(0)*(1-get_value('greedyP')))
        # get execution ratio in this block through pruning network
        upsampled = F.avg_pool2d(x, x.shape[2])
        ss = upsampled.view(x.shape[0], x.shape[1])
        h_state = self.rnncell(self.relu(self.embeddings[block_idx](ss.detach())), former_state)
        random_actions = torch.randint(0, get_value('group_num'), [random_explo], dtype=torch.long).cuda()
        greedy_actions = torch.argmax(h_state[random_explo:], 1).cuda()
        return torch.cat((random_actions, greedy_actions), 0), h_state


class GatedConv_BN(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1):
        super(GatedConv_BN, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size,
                              stride=stride, padding=padding)
        self.bn = nn.BatchNorm2d(out_channels)
        # Gate layers
        self.gate = nn.Sequential(nn.Linear(in_channels, 10),
                            nn.ReLU(),
                            nn.Linear(10, out_channels))
        for m in self.modules():
            if isinstance(m, nn.Linear):
                m.weight.data.normal_(0, math.sqrt(2. / m.weight.data.size(-1)))
                m.bias.data.zero_()

    def forward(self, x, ratios):
        upsampled = F.avg_pool2d(torch.abs(x), x.shape[2])
        ss = upsampled.view(x.shape[0], x.shape[1])
        gate_out = self.gate(ss.detach())
        gate_out = 1.5*F.sigmoid(gate_out)
        sort_channels_idx = torch.argsort(gate_out, 1)
        topks = torch.round(ratios * self.conv.out_channels).type(torch.cuda.LongTensor)
        mask = torch.sort(sort_channels_idx)[1] > (self.conv.out_channels - 1 - topks.view(-1, 1))
        active_idx = (gate_out*mask.type(torch.cuda.FloatTensor)).unsqueeze(2).unsqueeze(3)
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
        
    def forward(self, x, ratios):
        residual = x
        out = self.conv1_bn(x, ratios)
        out = self.relu(out)
        out = self.conv2_bn(out, ratios)
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

    def forward(self, x, ratios):
        residual = x
        out = self.conv1_bn(x, ratios)
        out = self.relu(out)
        out = self.conv2_bn(out, ratios)
        out = self.relu(out)
        out = self.conv3_bn(out, ratios)
        if self.downsample is not None:
            residual = self.downsample(x)
        out += residual
        return out


class ResNet(nn.Module):

    def __init__(self, depth, num_classes=100):
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

        block_inplanes_list = []
        self.layer1, inplanes_list = self._make_layer(block, 16, n)
        block_inplanes_list += inplanes_list
        self.layer2, inplanes_list = self._make_layer(block, 32, n, stride=2)
        block_inplanes_list += inplanes_list
        self.layer3, inplanes_list = self._make_layer(block, 64, n, stride=2)
        block_inplanes_list += inplanes_list
        self.blockskipnet = BlockSkip(block_inplanes_list)

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
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )
        layers = []
        inplanes_list = []
        inplanes_list.append(self.inplanes)
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))
            inplanes_list.append(self.inplanes)

        return nn.Sequential(*layers), inplanes_list


    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)    # 32x32
        
        prob_action_list = []
        block_idx = 0
        former_state=torch.zeros(x.size(0), get_value('group_num'), dtype=torch.float32).cuda()
        action=torch.zeros(x.size(0), dtype=torch.int32).cuda()
        for i in range(len(self.layer1)):
            action, former_state = self.blockskipnet(x, former_state, block_idx)
            x = self.layer1[i](x, action.type(torch.cuda.FloatTensor) / get_value('group_num'))
            prob_action_list.append([former_state, action])
            block_idx += 1
        for i in range(len(self.layer2)):
            action, former_state = self.blockskipnet(x, former_state, block_idx)
            x = self.layer2[i](x, action.type(torch.cuda.FloatTensor) / get_value('group_num'))
            prob_action_list.append([former_state, action])
            block_idx += 1
        for i in range(len(self.layer3)):
            action, former_state = self.blockskipnet(x, former_state, block_idx)
            x = self.layer3[i](x, action.type(torch.cuda.FloatTensor) / get_value('group_num'))
            prob_action_list.append([former_state, action])
            block_idx += 1

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x, prob_action_list


def resnet_rnp_rl(depth, num_classes=100, greedyP=0.9, group_num=8, min_exec_ratio=0.4):
    """
    Constructs a ResNet model.
    """
    slimmable_init()
    set_value('group_num', group_num)
    set_value('greedyP', greedyP)
    set_value('min_exec_ratio', min_exec_ratio)
    return ResNet(depth, num_classes=num_classes)
