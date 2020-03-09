import torch.nn as nn
import math
import torch
import numpy as np
import torch.nn.functional as F
import random
from torch.autograd import Variable
import torchvision
import torchvision.transforms as transforms
import torch.optim as optim

import os
import pdb

from utils import progress_bar

def adjust_learning_rate(optimizer, lr):
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

def make_layers(cfg, batch_norm=False):
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

class vgg_RPN(nn.Module):
    def __init__(self, cfg, num_classes=1000, group_num=4):
        super(vgg_RPN, self).__init__()
        self.cfg = cfg
        self.group_num = group_num
        self.conv_layers, self.extra_layers_encode = make_layers(cfg)
        self.classifier = nn.Sequential(
            nn.Linear(512, 512),
            nn.ReLU(True),
            nn.Dropout(p=0.5),
            nn.Linear(512, num_classes),
        )
        self.rnncell = nn.GRUCell(256, 4, bias=True)
        self._initialize_weights()
        # mode = 0: VGG baseline
        # mode = 1: random pruning
        # mode = 2: RNP training
        self.mode = 1
        self.group = []
        self.greedyP = 0.9

    def new_divide_conv(self, group_num=4):
        for layer in self.conv_layers:
            if isinstance(layer, nn.Conv2d):
                weights = layer.weight.data
                weights = weights.view(weights.size(0), weights.size(1), -1)
                norm = torch.mean(torch.norm(weights,2,2), 1)
                order = torch.argsort(norm).cuda()
                glen = int(order.shape[0] / group_num)
                group_index = order[-1] * torch.ones((group_num, order.shape[0]), dtype=torch.long).cuda()
                for _idx in range(group_num):
                    group_index[_idx, 0:(_idx+1)*glen] = torch.sort(order[( group_num - _idx - 1 )*glen:])[0]
                self.group += [group_index]


    def divide_conv(self):
        for layer in self.conv_layers:
            if isinstance(layer, nn.Conv2d):
                weights = layer.weight.data
                #print(weights.size())
                weights = weights.view(weights.size(0), weights.size(1), -1)
                norm = torch.norm(weights,2,2).cpu().numpy()
                norm = np.mean(norm, 1)
                order = np.argsort(norm)
                glen = int(order.shape[0]/4)
                g0 = torch.from_numpy(np.sort(order[3*glen:]))
                g1 = torch.from_numpy(np.sort(np.hstack((order[3*glen:], order[2*glen:3*glen]))))
                g2 = torch.from_numpy(np.sort(np.hstack((order[3*glen:], order[2*glen:3*glen], order[glen:2*glen]))))
                g3 = torch.from_numpy(np.sort(np.hstack((order[3*glen:], order[2*glen:3*glen], order[glen:2*glen], order[0:glen]))))
                self.group += [[g0, g1, g2, g3]]

    def forward(self, x):
        if self.mode == 0:
            for layer in self.conv_layers:
                #print layer
                x = layer(x)
            x = x.view(x.size(0), -1)
            x = self.classifier(x)

        if self.mode == 1:
            ct = 0
            bs = x.size(0)
            for layer in self.conv_layers:
                if isinstance(layer, nn.Conv2d) and ct > 0:
                    x = layer(x)
                    mask = torch.zeros(x.size(0), x.size(1)).cuda()
                    #pdb.set_trace()
                    mask.scatter_(1, self.group[ct][torch.randint(0, self.group_num, [bs], dtype=torch.long), :].cuda(), 1)
                    #for i in range(bs):
                    #    choice = random.randint(0, 3)
                    #    now_group = self.group[ct][choice]
                    #    mask[i, now_group, :, :].fill_(1.0)
                    mask = Variable(mask)
                    x = mask.unsqueeze(2).unsqueeze(3)*x
                    ct += 1
                elif isinstance(layer, nn.Conv2d) and ct == 0:
                    x = layer(x)
                    ct += 1
                else:
                    x = layer(x)
            x = x.view(x.size(0), -1)
            x = self.classifier(x)

        if self.mode == 2:
            ct = 0
            bs = x.size(0)
            former_state = Variable(torch.zeros(bs, 4))
            for layer in conv_layers:
                if isinstance(layer, nn.Conv2d) and ct > 0:
                    #choice = random.randint(0, 3)
                    #now_group = self.group[ct][choice]
                    #x = F.conv2d(x, layer.weight[former_group, now_group, :, :], layer.bias[now_group], kernel_size=3, padding=1)
                    x_pool = x.mean(3).mean(2)
                    x = layer(x)
                    mask = torch.zeros_like(x.data)
                    x_input = self.extra_layers_encode[ct](x_pool)
                    h = self.rnncell(x_input, former_state)
                    former_state = h
                    h_softmax = F.softmax(h)
                    h_softmax_np = h_softmax.data.cpu().numpy()
                    for i in range(bs):
                        choice = np.argmax(h_softmax_np[i])
                        if random.random() > self.greedyP:
                            choice = random.randint(0, 3)
                        now_group = self.group[ct][choice]
                        mask[i, now_group, :, :].fill_(1.0)
                    mask = Variable(mask)
                    x = mask*x
                    ct += 1
                elif isinstance(layer, nn.Conv2d) and ct == 0:
                    x = layer(x)
                    ct += 1
                else:
                    x = layer(x)
            x = x.view(x.size(0), -1)
            x = self.classifier(x)
        return x

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
        for m in self.extra_layers_encode:
            if isinstance(m, nn.Linear):
                m.weight.data.normal_(0, 0.01)
                m.bias.data.zero_()

cfg = [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M']

net = vgg_RPN(cfg, 100)
net.cuda()

transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

trainset = torchvision.datasets.CIFAR100(root='./data', train=True, download=True, transform=transform_train)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=128, shuffle=True, num_workers=2)

testset = torchvision.datasets.CIFAR100(root='./data', train=False, download=True, transform=transform_test)
testloader = torch.utils.data.DataLoader(testset, batch_size=100, shuffle=False, num_workers=2)

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=0.01, momentum=0.9, weight_decay=5e-5)

def train(epoch):
    print('\nEpoch: %d' % epoch)
    net.train()
    train_loss = 0
    raw_loss = 0
    correct = 0
    total = 0
    for batch_idx, (inputs, targets) in enumerate(trainloader):
        inputs, targets = inputs.cuda(), targets.cuda()
        optimizer.zero_grad()
        inputs, targets = Variable(inputs), Variable(targets)
        outputs = net(inputs)
        loss = criterion(outputs, targets)
        raw_loss += loss.item()
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        _, predicted = torch.max(outputs.data, 1)
        total += targets.size(0)
        correct += predicted.eq(targets.data).cpu().sum()

        progress_bar(batch_idx, len(trainloader), 'Loss: %.3f Row_loss: %.3f | Acc: %.3f%% (%d/%d)'
            % (train_loss/(batch_idx+1), raw_loss/(batch_idx+1), 100.*correct/total, correct, total))

def test(epoch):
    global best_acc
    net.eval()
    test_loss = 0
    correct = 0
    total = 0
    for batch_idx, (inputs, targets) in enumerate(testloader):
        inputs, targets = inputs.cuda(), targets.cuda()
        inputs, targets = Variable(inputs, volatile=True), Variable(targets)
        outputs = net(inputs)
        loss = criterion(outputs, targets)

        test_loss += loss.item()
        _, predicted = torch.max(outputs.data, 1)
        total += targets.size(0)
        correct += predicted.eq(targets.data).cpu().sum()

        progress_bar(batch_idx, len(testloader), 'Loss: %.3f  | Acc: %.3f%% (%d/%d)'
            % (test_loss/(batch_idx+1), 100.*correct/total, correct, total))


#nlr = 0.01
#for epoch in range(0, 300):
#    if epoch > 0 and epoch % 25 == 0:
#        nlr = nlr / 2.0
#        adjust_learning_rate(optimizer, nlr)
#    
#    train(epoch)
#    test(epoch)
#    save_prefix = 'experiments/cifar100/dynamic_inference/RNP_RL/vgg_RNP/'
#    if not os.path.isdir(save_prefix):
#        os.makedirs(save_prefix)
#    torch.save(net.state_dict(), os.path.join(save_prefix, 'vgg16-cifar100-random.pth'))


nlr = 0.001
save_prefix = 'experiments/cifar100/dynamic_inference/RNP_RL/vgg_RNP/'
net.load_state_dict(torch.load(save_prefix + 'vgg16-cifar100.pth'))
net.mode = 1
net.new_divide_conv()

#net.load_state_dict(torch.load('vgg16-cifar100-random.pth'))

for epoch in range(0, 100):
    if epoch % 25 == 0:
        nlr = nlr / 2.0
        adjust_learning_rate(optimizer, nlr)

    train(epoch)
    test(epoch)
    torch.save(net.state_dict(), 'vgg16-cifar100-random.pth')


