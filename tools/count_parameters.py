from models.mobilenet_112 import *
#from models.mobilenet_96 import *
from models.mobilenet import *
from models.mobilenet_112_custom import *
from models.mobilenet_112_tiny import *
from models.mobilenet_64_tiny import *
from models.mobilenet_64_tiny_tiny import *
from models.mobilenet_64_tiny_gray import *
from models.custom_quality import *
from thop import profile
import torch
model = mobilenet_v2_t4_w05_f0_64_tiny_gray(1)
input = torch.randn(1, 1, 64, 64)
model1_flops, model1_params = profile(model, inputs=(input, ))

model = mobilenet_v2_t3_w025_f0_64_tiny_gray(1)
input = torch.randn(1, 1, 64, 64)
model2_flops, model2_params = profile(model, inputs=(input, ))

model = custom_quality(1)
input = torch.randn(1, 1, 112, 112)
model3_flops, model3_params = profile(model, inputs=(input, ))

print('mobilenet_v2_t4_w05_64_tiny_gray flops is: {}, params is: {}'.format(str(model1_flops), str(model1_params)))
print('mobilenet_v2_t3_w025_64_tiny flops is: {}, parmas is: {}'.format(str(model2_flops), str(model2_params)))
print('custom_quality flops is: {}, params: {}'.format(str(model3_flops), str(model3_params)))
#model = mobilenet_v2_basic_fc0(1)
#input = torch.randn(1, 3, 224, 224)
#flops, params = profile(model, inputs=(input, ))
#print('mobilenet_v2_basic flops is: {}'.format(str(flops)))
