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

from utils import accuracy, progress_bar, Logger, AverageMeter, mkdir_p
import loss

parser = argparse.ArgumentParser(description='PyTorch CIFAR10/100 Training')

model_names = sorted(name for name in models.__dict__
    if name.islower() and not name.startswith("__")
    and callable(models.__dict__[name]))


loss_dict = {
    'kd': 'KDLoss',
    'cl_kd': 'CurriculumKDLoss',
    'anti_cl_kd': 'AntiCurriculumKDLoss',
    'pce': 'PartCElossbyRemoveLowConfidenceData',
    'pkd': 'PartKDlossbyRemoveLowConfidenceData',
    'wce': 'WeightedCrossEntropy',
    'cwce': 'CorrectWeightedCrossEntropy',
    'dkce': 'DKwithCrossEntropy',
    'spkd': 'SPKDCrossEntropy',
}


parser.add_argument('--specify-path', type=str, default='', help='specify the path of model')
# parameters for teacher model
parser.add_argument('--teacher-path', default='template/checkpoint.pth.tar', type=str)
parser.add_argument('--teacher-depth', default=32, type=int)
parser.add_argument('--teacher-arch', default='resnet', type=str)
parser.add_argument('--teacher-growthRate', type=int, default=12, help='Growth rate for DenseNet.')
parser.add_argument('--temperature', default=5, type=float)
parser.add_argument('--pce-threshold', default=0.1, type=float)
parser.add_argument('--loss-fun', default='kd', type=str)

parser.add_argument('--save-path', default='experiments/cifar100/template', type=str)
# Datasets
parser.add_argument('-d', '--dataset', default='cifar10', type=str)
parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                    help='number of data loading workers (default: 4)')
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
parser.add_argument('--weight-decay', '--wd', default=1e-4, type=float,
                    metavar='W', help='weight decay (default: 1e-4)')
# Checkpoints
parser.add_argument('-c', '--checkpoint', default='checkpoint', type=str, metavar='PATH',
                    help='path to save checkpoint (default: checkpoint)')
#parser.add_argument('--resume', default='', type=str, metavar='PATH',
#                    help='path to latest checkpoint (default: none)')
parser.add_argument('-r', '--resume', dest='resume', action='store_true',
                    help='resume for continue training')
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
parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true',
                    help='evaluate model on validation set')
#Device options
parser.add_argument('--gpu-id', default='0', type=str,
                    help='id(s) for CUDA_VISIBLE_DEVICES')

args = parser.parse_args()
state = {k: v for k, v in args._get_kwargs()}

# get the loss function
kd_loss_fun = loss.__dict__[loss_dict[args.loss_fun]](T=args.temperature, num_classes=10 if args.dataset=='cifar10' else 100, threshold=args.pce_threshold)
kd_loss_fun.cuda()

# when specify_path is true, use the specify path
if args.specify_path != '':
    args.save_path = args.specify_path
else :
    args.save_path = 'experiments/' + args.dataset + '/reverse_kd/' + args.teacher_arch + str(args.teacher_depth)  + '_' + args.loss_fun \
                    + '_' + args.arch + str(args.depth) + '_wd' + str(args.weight_decay) + '_T' + str(args.temperature) + '_lr' + str(args.lr)
if not os.path.isdir(args.save_path):
    os.makedirs(args.save_path)
print(args.save_path)

# Validate dataset
assert args.dataset == 'cifar10' or args.dataset == 'cifar100', 'Dataset can only be cifar10 or cifar100.'
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
    else:
        dataloader = datasets.CIFAR100
        num_classes = 100

    trainset = dataloader(root='./data', train=True, download=True, transform=transform_train)
    trainloader = data.DataLoader(trainset, batch_size=args.train_batch, shuffle=True, num_workers=args.workers)

    testset = dataloader(root='./data', train=False, download=False, transform=transform_test)
    testloader = data.DataLoader(testset, batch_size=args.test_batch, shuffle=False, num_workers=args.workers)

    print('class number is :{}'.format(str(num_classes)))
    # Model
    print("==> creating model '{}'".format(args.arch))

    if args.teacher_arch.startswith('densenet'):
        print(args.teacher_growthRate)
        t_model = models.__dict__[args.teacher_arch](
                    num_classes=num_classes,
                    depth=args.teacher_depth,
                    growthRate=args.teacher_growthRate,
                    compressionRate=args.compressionRate,
                    dropRate=args.drop,
                )
    elif args.teacher_arch.startswith('wrn'):
        t_model = models.__dict__[args.teacher_arch](
                    num_classes=num_classes,
                    depth=args.teacher_depth,
                    widen_factor=args.widen_factor,
                    dropRate=args.drop,
                )
    elif args.teacher_arch.endswith('resnet'):
        t_model = models.__dict__[args.teacher_arch](
                    num_classes=num_classes,
                    depth=args.teacher_depth,
                )
    else:
        print('only support densenet/wrn/resnet')
        exit(-1)
    s_model = models.__dict__[args.arch](
                num_classes=num_classes,
                depth=args.depth,
            )

    t_model = torch.nn.DataParallel(t_model).cuda()
    s_model = torch.nn.DataParallel(s_model).cuda()
    cudnn.benchmark = True
    print('Teacher model total params: %.2fM' % (sum(p.numel() for p in t_model.parameters())/1000000.0))
    print('Student model total params: %.2fM' % (sum(p.numel() for p in s_model.parameters())/1000000.0))

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(s_model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=1e-4)
    title = 'cifar-100' + args.arch
    # Load checkpoint.
    t_checkpoint = torch.load(args.teacher_path)
    t_model.load_state_dict(t_checkpoint['state_dict'])
    if args.evaluate:
        print('\nEvaluation train')
        checkpoint = torch.load(args.save_path + '/' + 'model_best.pth.tar')
        s_model.load_state_dict(checkpoint['state_dict'])

        print('\nEvaluation test')
        test_loss, test_top1, top5= test(testloader, s_model, criterion, start_epoch, use_cuda, args)
        print(' Test Loss:  %.8f, Test top1:  %.2f, Test topt: %.2f' % (test_loss, test_top1, test_top5))
        return
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
        s_model.load_state_dict(checkpoint['state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        logger = Logger(os.path.join(args.save_path, 'log.txt'), title=title, resume=True)
    else:
        logger = Logger(os.path.join(args.save_path, 'log.txt'), title=title)
        logger.set_names(['Learning Rate', 'Train Loss', 'Train ce loss', 'Valid Loss', 'Train top1.', 'train top5', 'Valid top1.', 'valid top5'])

    # Train and val
    for epoch in range(start_epoch, args.epochs):
        adjust_learning_rate(optimizer, epoch)
        #adjust_learning_rate(gate_optimizer, epoch)

        print('\nEpoch: [%d | %d] LR: %f' % (epoch + 1, args.epochs, state['lr']))
        train_loss, train_loss_kl, train_loss_ce, train_top1, train_top5 = train(trainloader, t_model, s_model, criterion, optimizer, epoch, use_cuda, args)
        test_loss, test_top1, test_top5= test(testloader, s_model, criterion, epoch, use_cuda, args)

        logger.append([state['lr'], train_loss_kl, train_loss_ce, test_loss, train_top1, train_top5, test_top1, test_top5])
        # save model
        is_best = test_top1 > best_acc
        best_acc = max(test_top1, best_acc)
        print(args.save_path)
        save_checkpoint({
                'epoch': epoch + 1, 
                'state_dict': s_model.state_dict(),
                'acc': test_top1,
                'best_acc': best_acc,
                'optimizer' : optimizer.state_dict(),
            }, is_best, checkpoint=args.save_path)

    logger.close()
    print('Best acc:')
    print(best_acc)

def train(trainloader, t_model, s_model, criterion, optimizer, epoch, use_cuda, args):
    # switch to train mode
    global kd_loss_fun, cmclloss_v1, indeploss, mclloss
    t_model.eval()
    s_model.train()

    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    losses_kl = AverageMeter()
    losses_ce = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()
    end = time.time()

    for batch_idx, (batch_data) in enumerate(trainloader):
        # measure data loading time
        if len(batch_data) == 2:
            inputs, targets = batch_data
        else:
            inputs, targets, indexes = batch_data
        
        data_time.update(time.time() - end)
        if use_cuda:
            inputs, targets = inputs.cuda(), targets.cuda(async=True)
        inputs, targets = torch.autograd.Variable(inputs), torch.autograd.Variable(targets)
        # compute output
        t_outputs = t_model(inputs)
        s_outputs = s_model(inputs)
        t_prec1, t_prec5 = accuracy(t_outputs.data, targets.data, topk=(1, 5))
        # measure accuracy and record loss
        prec1, prec5 = accuracy(s_outputs.data, targets.data, topk=(1, 5))
        loss_kl = kd_loss_fun(s_outputs, t_outputs.detach(), targets)
        loss_ce = criterion(s_outputs, targets)
        loss = loss_kl
        losses.update(loss.item(), inputs.size(0))
        losses_kl.update(loss_kl.item(), inputs.size(0))
        losses_ce.update(loss_ce.item(), inputs.size(0))
        top1.update(prec1.item(), inputs.size(0))
        top5.update(prec5.item(), inputs.size(0))
        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        progress_bar(batch_idx, len(trainloader), 'Loss: %.2f | KLloss: %.2f | ce_loss: %.2f | Top1: %.2f | Top5: %.2f | t_top1: %.2f | t_top5: %.2f' 
                % (losses.avg, loss_kl, loss_ce, top1.avg, top5.avg, t_prec1, t_prec5))
        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

    return (losses.avg, losses_kl.avg, losses_ce.avg, top1.avg, top5.avg)

def test(testloader, model, criterion, epoch, use_cuda, args):
    global best_acc
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()
    model.eval()

    wholedata_num = testloader.dataset.__len__()

    fine_accuracy = np.zeros(100, dtype=float)       # 100*1
    classes_confidence = np.zeros((100, 100), dtype=float)  # 100*100
    samples_confidence = np.zeros((wholedata_num, 100), dtype=float) # n*100

    end = time.time()
    start_index = 0
    for batch_idx, (batch_data) in enumerate(testloader):
        if len(batch_data) == 2:
            inputs, targets = batch_data
        else:
            inputs, targets, indexes = batch_data
        # measure data loading time
        data_time.update(time.time() - end)
        if use_cuda:
            inputs, targets = inputs.cuda(), targets.cuda()
        with torch.no_grad():
            inputs, targets = torch.autograd.Variable(inputs), torch.autograd.Variable(targets)

        # compute output
        outputs = model(inputs)

        loss = criterion(outputs, targets)
        # measure accuracy and record loss
        prec1, prec5 = accuracy(outputs.data, targets.data, topk=(1, 5))
        losses.update(loss.item(), inputs.size(0))
        top1.update(prec1.item(), inputs.size(0))
        top5.update(prec5.item(), inputs.size(0))

        # measure accuracy and record loss
        batch_time.update(time.time() - end)
        end = time.time()

        progress_bar(batch_idx, len(testloader), 'Loss: %.2f | Top1: %.2f | Top5: %.2f' 
                % (losses.avg, top1.avg, top5.avg))

    return (losses.avg, top1.avg, top5.avg)

def save_checkpoint(state, is_best, checkpoint='checkpoint', filename='checkpoint.pth.tar'):
    filepath = os.path.join(checkpoint, filename)
    torch.save(state, filepath)
    if is_best:
        shutil.copyfile(filepath, os.path.join(checkpoint, 'model_best.pth.tar'))

def adjust_learning_rate(optimizer, epoch):
    global state
    if epoch in args.schedule:
        state['lr'] *= args.gamma
        for param_group in optimizer.param_groups:
            param_group['lr'] = state['lr']

if __name__ == '__main__':
    main()
