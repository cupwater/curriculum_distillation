'''
Training script for CIFAR-10/100
Copyright (c) Baoyun Peng, 2018
'''
from __future__ import print_function

import argparse
import os
import shutil
import time
import random
import yaml
import numpy as np
import pdb


import torch
import torch.nn.functional as F
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data as data
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import models.cifar as models
from utils import Logger, AverageMeter, accuracy, cal_accuracy_each_class, \
            cal_accuracy_confidence, cal_samples_confidence, mkdir_p, progress_bar

model_names = sorted(name for name in models.__dict__
    if name.islower() and not name.startswith("__")
    and callable(models.__dict__[name]))

parser = argparse.ArgumentParser(description='PyTorch CIFAR100 Training')
parser.add_argument('--dataset', type=str, default='cifar100', help='dataset')
# Optimization options
parser.add_argument('--epochs', default=300, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('--train-batch', default=64, type=int, metavar='N',
                    help='train batchsize')
parser.add_argument('--test-batch', default=32, type=int, metavar='N',
                    help='test batchsize')
parser.add_argument('--lr', '--learning-rate', default=0.1, type=float,
                    metavar='LR', help='initial learning rate')
parser.add_argument('--drop', '--dropout', default=0, type=float,
                    metavar='Dropout', help='Dropout ratio')
parser.add_argument('--schedule', type=int, nargs='+', default=[150, 225],
                        help='Decrease learning rate at these epochs.')
parser.add_argument('--gamma', type=float, default=0.1, help='LR is multiplied by gamma on schedule.')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum')
parser.add_argument('--weight-decay', '--wd', default=5e-4, type=float,
                    metavar='W', help='weight decay (default: 1e-4)')
# Checkpoints
parser.add_argument('-c', '--checkpoint', default='checkpoint', type=str, metavar='PATH',
                    help='path to save checkpoint (default: checkpoint)')
# Architecture
parser.add_argument('--arch', '-a', metavar='ARCH', default='resnet20',
                    choices=model_names,
                    help='model architecture: ' +
                        ' | '.join(model_names) +
                        ' (default: resnet18)')
parser.add_argument('--depth', type=int, default=29, help='Model depth.')
parser.add_argument('--cardinality', type=int, default=8, help='Model cardinality (group).')
parser.add_argument('--widen-factor', type=int, default=4, help='Widen factor. 4 -> 64, 8 -> 128, ...')
parser.add_argument('--growthRate', type=int, default=12, help='Growth rate for DenseNet.')
parser.add_argument('--compressionRate', type=int, default=2, help='Compression Rate (theta) for DenseNet.')
# Miscs
parser.add_argument('--manualSeed', type=int, help='manual seed')
parser.add_argument('-r', '--resume', dest='resume', action='store_true',
                    help='resume for continue training')
parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true',
                    help='evaluate model on validation set')

parser.add_argument('-f', '--finetune', dest='finetune', action='store_true',
                    help='finetune a pretrained_model')
parser.add_argument('--model-path', type=str, default='n', help='path of pretrained model')

parser.add_argument('--greedyP', type=float, default=0.9, help='LR is multiplied by gamma on schedule.')

args = parser.parse_args()
state = {k: v for k, v in args._get_kwargs()}

# Validate dataset
assert args.dataset == 'cifar10' or args.dataset == 'cifar100', 'Dataset can only be cifar10 or cifar100.'
# when specify_path is true, use the specify path
args.save_path = 'experiments/' + args.dataset + '/dynamic_inference/RNP/' + args.arch + str(args.depth) + '_wd' + str(args.weight_decay) + '_greedyP' + str(args.greedyP) 
if not os.path.isdir(args.save_path):
    os.makedirs(args.save_path)

# Use CUDA
use_cuda = torch.cuda.is_available()

# Random seed
if args.manualSeed is None:
    args.manualSeed = random.randint(1, 10000)
random.seed(args.manualSeed)
torch.manual_seed(args.manualSeed)
if use_cuda:
    torch.cuda.manual_seed_all(args.manualSeed)

best_acc = 0  # best test accuracy


def main():
    global best_acc
    start_epoch = args.start_epoch  # start from epoch 0 or last checkpoint epoch

    if not os.path.isdir(args.checkpoint):
        mkdir_p(args.checkpoint)

    # Data
    print('==> Preparing dataset %s' % args.dataset)
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
    if args.dataset == 'cifar10':
        dataloader = datasets.CIFAR10
        num_classes = 10
    elif args.dataset == 'cifar100':
        dataloader = datasets.CIFAR100
        num_classes = 100


    trainset = dataloader(root='./data', train=True, download=True, transform=transform_train)
    trainloader = data.DataLoader(trainset, batch_size=args.train_batch, shuffle=True, num_workers=1)

    testset = dataloader(root='./data', train=False, download=False, transform=transform_test)
    testloader = data.DataLoader(testset, batch_size=args.test_batch, shuffle=False, num_workers=1)

    # Model
    print("==> creating model '{}'".format(args.arch))

    model = models.__dict__[args.arch](num_classes=num_classes, greedyP=args.greedyP)
    model.load_state_dict(torch.load('experiments/cifar100/dynamic_inference/RNP_RL/vgg_RNP/vgg16-cifar100.pth'), strict=False)
    model.divide_conv()
    model.load_state_dict(torch.load('experiments/cifar100/dynamic_inference/RNP_RL/vgg_RNP/vgg16-cifar100-random.pth'), strict=False)

    model = torch.nn.DataParallel(model).cuda()
    #pmodel = torch.nn.DataParallel(pmodel).cuda()
    cudnn.benchmark = True
    print('    Total params: %.2fM' % (sum(p.numel() for p in model.parameters())/1000000.0))
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)
    criterion_rl = nn.MSELoss()
    optimizer_rl = optim.Adam(model.module.pnet.parameters(), lr=0.0001, weight_decay=5e-5)

    # Resume
    title = 'cifar-10-' + args.arch
    if args.resume:
        # Load checkpoint.
        print('==> Resuming from checkpoint..')
        print(args.save_path + '/' + 'checkpoint.pth.tar')
        assert os.path.isfile(args.save_path + '/' + 'checkpoint.pth.tar'), 'Error: no checkpoint directory found!'
        checkpoint = torch.load(args.save_path + '/' + 'checkpoint.pth.tar')
        best_acc = checkpoint['best_acc']
        start_epoch = checkpoint['epoch']

        # recover learning rate
        for epoch in args.schedule:
            if start_epoch > epoch:
                state['lr'] *= args.gamma

        model.load_state_dict(checkpoint['state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        logger = Logger(os.path.join(args.save_path, 'log.txt'), title=title, resume=True)
    elif args.evaluate:
        print('\nEvaluation train')
        checkpoint = torch.load(args.save_path + '/' + 'model_best.pth.tar')
        model.load_state_dict(checkpoint['state_dict'])

        print('\nEvaluation test')
        test_loss, test_top1, test_top5 = test(testloader, model, criterion, start_epoch, use_cuda)
        print(' Test Loss:  %.8f, Test top1:  %.2f, Test top5' % (test_loss, test_top1, test_top5))
        return
    elif args.finetune:
        # Load checkpoint.
        print('==> finetune a pretrained model..')
        print(args.model_path)
        assert os.path.isfile(args.model_path), 'Error: no checkpoint directory found!'
        checkpoint = torch.load(args.model_path)
        model.load_state_dict(checkpoint['state_dict'])
    else:
        logger = Logger(os.path.join(args.save_path, 'log.txt'), title=title)
        logger.set_names(['Learning Rate', 'Train Loss', 'Valid Loss', 'Train top1', 'Train top5', 'Valid top1.', 'Valid top5'])
    
    # Train and val
    for epoch in range(start_epoch, args.epochs):
        adjust_learning_rate(optimizer, epoch)

        print('\nEpoch: [%d | %d] LR: %f' % (epoch + 1, args.epochs, state['lr']))

        train_loss, train_top1, train_top5 = train(trainloader, model, criterion, criterion_rl, optimizer, optimizer_rl, epoch, use_cuda)
        test_loss, test_top1, test_top5 = test(testloader, model, criterion, epoch, use_cuda)
        # append logger file
        logger.append([state['lr'], train_loss, test_loss, train_top1, train_top5, test_top1, test_top5])
        print(args.save_path)
        # save model
        is_best = test_top1 > best_acc
        best_acc = max(test_top1, best_acc)
        save_checkpoint({
                'epoch': epoch + 1,
                'state_dict': model.state_dict(),
                'acc': test_top1,
                'best_acc': best_acc,
                'optimizer' : optimizer.state_dict(),
            }, is_best, save_path=args.save_path)

    logger.close()

    print('Best acc:')
    print(best_acc)

def train(trainloader, model, criterion, criterion_rl, optimizer, optimizer_rl, epoch, use_cuda):
    # switch to train mode
    model.train()

    raw_loss = 0
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    losses_rl = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()
    end = time.time()

    for batch_idx, (inputs, targets) in enumerate(trainloader):
        # measure data loading time
        data_time.update(time.time() - end)
        if use_cuda:
            inputs, targets = inputs.cuda(), targets.cuda(async=True)
        inputs, targets = torch.autograd.Variable(inputs), torch.autograd.Variable(targets)
        # compute output
        optimizer.zero_grad()
        outputs, y  = model(inputs)
        loss = criterion(outputs, targets)
        raw_loss += loss.item()
        # measure accuracy and record loss
        prec1, prec5 = accuracy(outputs.data, targets.data, topk=(1, 5))
        losses.update(loss.item(), inputs.size(0))
        top1.update(prec1.item(), inputs.size(0))
        top5.update(prec5.item(), inputs.size(0))
        # compute gradient and do SGD step
        loss.backward(retain_graph=True)
        optimizer.step()


        rtargets = 0
        state_action_values = 0
        for i in range(len(y)):
            optimizer_rl.zero_grad()
            action = y[i][1]
            state_action_values = y[i][0].gather(1, action.unsqueeze(1))
            if i < len(y) - 1:
                rtargets = -action.type(torch.cuda.FloatTensor)*0.1 #+  torch.max(y[i+1][0].detach(), 1)[0].type(torch.cuda.FloatTensor)
            else:
                rtargets = - action.type(torch.cuda.FloatTensor)*0.1 - raw_loss

            loss_rl = criterion_rl(state_action_values, rtargets)
            loss_rl.backward(retain_graph=True)
            losses_rl.update(loss_rl.item(), inputs.size(0))
            optimizer_rl.step()

        progress_bar(batch_idx, len(trainloader), 'Loss: %.2f | Top1: %.2f | Top5: %.2fi | loss_rl: %.2f'
                    % (losses.avg, top1.avg, top5.avg, losses_rl.avg))

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

    return (losses.avg, top1.avg, top5.avg)

def test(testloader, model, criterion, epoch, use_cuda):
    global best_acc

    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    # switch to evaluate mode
    model.eval()
    end = time.time()
    start_index = 0
    for batch_idx, (inputs, targets) in enumerate(testloader):
        # measure data loading time
        data_time.update(time.time() - end)

        if use_cuda:
            inputs, targets = inputs.cuda(), targets.cuda()
        with torch.no_grad():
            inputs, targets = torch.autograd.Variable(inputs), torch.autograd.Variable(targets)
        # compute output
        outputs, y = model(inputs)
        loss = criterion(outputs, targets)
        # measure accuracy and record loss
        prec1, prec5 = accuracy(outputs.data, targets.data, topk=(1, 5))

        losses.update(loss.item(), inputs.size(0))
        top1.update(prec1.item(), inputs.size(0))
        top5.update(prec5.item(), inputs.size(0))

        progress_bar(batch_idx, len(testloader), 'Loss: %.2f | Top1: %.2f | Top5: %.2f'
                    % (losses.avg, top1.avg, top5.avg))
        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

    return (losses.avg, top1.avg, top5.avg)

def save_checkpoint(state, is_best, save_path='experiment/template', filename='checkpoint.pth.tar'):
    filepath = os.path.join(save_path, filename)
    torch.save(state, filepath)
    if is_best:
        shutil.copyfile(filepath, os.path.join(save_path, 'model_best.pth.tar'))

def adjust_learning_rate(optimizer, epoch):
    global state
    if epoch in args.schedule:
        state['lr'] *= args.gamma
        for param_group in optimizer.param_groups:
            param_group['lr'] = state['lr']

if __name__ == '__main__':
    main()

