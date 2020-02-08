import torch.nn as nn
import torch.nn.functional as F
import torch
import math
import pdb

__all__ = ['cifarnet_gate']

def conv2d_same_padding(input, weight, bias=None, stride=[1, 1], dilation=[1, 1], groups=1):
    input_rows = input.size(2)
    filter_rows = weight.size(2)
    out_rows = (input_rows + stride[0] - 1) // stride[0]
    padding_rows = max(0, (out_rows - 1) * stride[0] +
                        (filter_rows - 1) * dilation[0] + 1 - input_rows)
    rows_odd = (padding_rows % 2 != 0)
    padding_cols = max(0, (out_rows - 1) * stride[0] +
                        (filter_rows - 1) * dilation[0] + 1 - input_rows)
    cols_odd = (padding_rows % 2 != 0)

    if rows_odd or cols_odd:
        input = F.pad(input, [0, int(cols_odd), 0, int(rows_odd)])

    return F.conv2d(input, weight, bias, stride,
                  padding=(padding_rows // 2, padding_cols // 2),
                  dilation=dilation, groups=groups)


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


class GatedConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1, gated=True):
        super(GatedConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size,
                              stride=stride, padding=padding)
        self.bn = nn.BatchNorm2d(out_channels)
        #self.bn = nn.BatchNorm2d(out_channels, affine=False)

        self.gate = nn.Linear(in_channels, out_channels)
        #self.beta = nn.Parameter(torch.Tensor(out_channels))
        #nn.init.zeros_(self.beta)

        self.ratio = _Graph.get_global_var('ratio')
        # init the parameters of gate
        self.gate.weight.data.normal_(0, math.sqrt(2. / out_channels))
        nn.init.ones_(self.gate.bias)

    # add regurization for the gate, l1, l2 norm
    def regurizer(self, x):
        loss = torch.sum(torch.abs(x)) 
        return loss

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = F.relu(x)
        return x        


    def gate_forward(self, x):
        upsampled = F.avg_pool2d(torch.abs(x), x.shape[2])
        ss = upsampled.view(x.shape[0], x.shape[1])

        o_gate = self.gate(ss.detach())
        o_gate = F.relu(o_gate)

        regurize_loss = self.regurizer(o_gate)

        inactive_channels = int(self.conv.out_channels - round(self.conv.out_channels * self.ratio))
        if inactive_channels > 0:
            inactive_idx = (-o_gate).topk(inactive_channels, 1)[1]
            o_gate.scatter_(1, inactive_idx, 0)  # set inactive channels as zeros

        x1 = self.conv(x)
        x = self.bn(x1)

        x = o_gate.unsqueeze(2).unsqueeze(3) * x
        x = F.relu(x)
        return x, regurize_loss


class CifarNet(nn.Module):
    def __init__(self, num_classes=100, gated=True, ratio=1.0):
        _Graph.set_global_var('ratio', ratio)
        super(CifarNet, self).__init__()
        self.gconv0 = GatedConv(3, 64, padding=0, gated=gated)
        self.gconv1 = GatedConv(64, 64, gated=gated)
        self.gconv2 = GatedConv(64, 128, stride=2, gated=gated)
        self.gconv3 = GatedConv(128, 128, gated=gated)
        self.drop3 = nn.Dropout2d()
        self.gconv4 = GatedConv(128, 128, gated=gated)
        self.gconv5 = GatedConv(128, 192, stride=2, gated=gated)
        self.gconv6 = GatedConv(192, 192, gated=gated)
        self.drop6 = nn.Dropout2d()
        self.gconv7 = GatedConv(192, 192, gated=gated)
        self.pool = nn.AvgPool2d(8)
        self.fc = nn.Linear(192, num_classes)

        self.gated = gated


    def forward(self, x):
        if self.gated:
            return self.forward_gated(x)
        else:
            return self.forward_origin(x)


    def forward_gated(self, x):
        x, regurize_loss0 = self.gconv0.gate_forward(x)
        x, regurize_loss1 = self.gconv1.gate_forward(x)
        x, regurize_loss2 = self.gconv2.gate_forward(x)
        x, regurize_loss3 = self.gconv3.gate_forward(x)
        x = self.drop3(x)
        x, regurize_loss4 = self.gconv4.gate_forward(x)
        x, regurize_loss5 = self.gconv5.gate_forward(x)
        x, regurize_loss6 = self.gconv6.gate_forward(x)
        x = self.drop6(x)
        x, regurize_loss7 = self.gconv7.gate_forward(x)
        x = self.pool(x)
        x = x.view(-1, 192)
        x = self.fc(x)
        if self.training:
            regurize_loss = regurize_loss0 + regurize_loss1 + regurize_loss2 + regurize_loss3 + regurize_loss4 + \
                                regurize_loss5 + regurize_loss6 + regurize_loss7
            return  x, regurize_loss
        return x

    def forward_origin(self, x):
        x = self.gconv0(x)
        x = self.gconv1(x)
        x = self.gconv2(x)
        x = self.gconv3(x)
        x = self.drop3(x)
        x = self.gconv4(x)
        x = self.gconv5(x)
        x = self.gconv6(x)
        x = self.drop6(x)
        x = self.gconv7(x)
        x = self.pool(x)
        x = x.view(-1, 192)
        x = self.fc(x)
        return  x

def cifarnet_gate(num_classes=100, gated=False, ratio=1.0):
    """
    Constructs a ResNet model.
    """
    return CifarNet(num_classes=num_classes, gated=gated, ratio=ratio)
