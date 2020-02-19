import torch.nn as nn
import torch.nn.functional as F
import pdb

def slimmable_init():
    global _global_dict
    _global_dict = {}
 
 
def set_value(key, value):
    _global_dict[key] = value
 
 
def get_value(key, defValue=None):
    try:
        return _global_dict[key]
    except KeyError:
        print('no such data {}'.format(key))


class SwitchableBatchNorm2d(nn.Module):
    def __init__(self, num_feature):
        super(SwitchableBatchNorm2d, self).__init__()
        self.width_multi_list = get_value('width_multi_list')
        self.num_features_list = []
        for width_multi in self.width_multi_list:
            self.num_features_list.append( int(width_multi*num_feature) )
        bns = []
        for i in self.num_features_list:
            bns.append(nn.BatchNorm2d(i))
        self.bn = nn.ModuleList(bns)
        self.ignore_model_profiling = True

    def forward(self, input_list):
        out = []
        for _idx in range(len(input_list)):
            out.append( self.bn[_idx]( input_list[_idx] ) )
        return out


class SlimmableReLU(nn.ReLU):
    def __init__(self):
        super(SlimmableReLU, self).__init__()
    def forward(self, input_list):
        out = []
        for input in input_list:
            out.append(F.relu(input))
        return out


class SlimmableAvgPool2d(nn.AvgPool2d):
    def __init__(self, kernel_size):
        super(SlimmableAvgPool2d, self).__init__(kernel_size)
        self.kernel_size=kernel_size
    def forward(self, input_list):
        out = []
        for input in input_list:
            out.append(F.avg_pool2d(input, self.kernel_size))
        return out


class SlimmableConv2d(nn.Conv2d):
    def __init__(self, in_planes, out_planes, kernel_size=3, stride=1, padding=0, bias=True):
        super(SlimmableConv2d, self).__init__( in_planes, out_planes,
            kernel_size, stride=stride, padding=padding, bias=bias
        )
        
        self.width_multi_list = get_value('width_multi_list')
        self.in_planes_list = []
        self.out_planes_list = []
        for width_multi in self.width_multi_list:
            self.in_planes_list.append( int(width_multi*in_planes) )
            self.out_planes_list.append( int(width_multi*out_planes) )

    def forward(self, input_list):
        out = []
        for _idx in range(len(input_list)):
            self.in_planes = self.in_planes_list[_idx]
            self.out_planes = self.out_planes_list[_idx]
            weight = self.weight[:self.out_planes, :self.in_planes, :, :]
            if self.bias is not None:
                bias = self.bias[:self.in_planes]
            else:
                bias = self.bias
            _out = nn.functional.conv2d(input_list[_idx], weight, bias, self.stride, self.padding)
            out.append(_out)
        return out

# the out_features is fixed
class SlimmableLinear(nn.Linear):
    def __init__(self, in_features, out_features, bias=True):
        super(SlimmableLinear, self).__init__(in_features, out_features, bias=bias)
        self.width_multi_list = get_value('width_multi_list')
        self.in_features_list = []
        self.out_features_list = []
        for width_multi in self.width_multi_list:
            self.in_features_list.append( int(width_multi*in_features) )
            self.out_features_list.append( out_features )

    def forward(self, input_list):
        out = []
        for _idx in range(len(input_list)):
            self.in_features = self.in_features_list[_idx]
            self.out_features = self.out_features_list[_idx]
            weight = self.weight[:self.out_features, :self.in_features]
            if self.bias is not None:
                bias = self.bias[:self.out_features]
            else:
                bias = self.bias
            _out = nn.functional.linear(input_list[_idx], weight, bias)
            out.append(_out)
        return out

