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
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data as data
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import models.cifar as models

from utils import accuracy, cal_accuracy_each_class, progress_bar, Logger, AverageMeter, mkdir_p
from loss import KDLoss, CMCLLoss_v1, MCLLoss, IndependentLoss
from models.cifar.Gate import Gate

parser = argparse.ArgumentParser(description='CIFAR10/100 Training')

model_names = sorted(name for name in models.__dict__
    if name.islower() and not name.startswith("__")
    and callable(models.__dict__[name]))


parser.add_argument('--beta', default=0.1, type=float)
parser.add_argument('--topk', default=1, type=int)
parser.add_argument('--num-branch', default=1, type=int)
parser.add_argument('--ensemble-scale', default=1, type=int)
parser.add_argument('--cmcloss-scale', default=0.3, type=float)
parser.add_argument('--kl-scale', default=1, type=int)
parser.add_argument('--branch-scale', default=1, type=int)
parser.add_argument('--save-path', default='experiments/cifar100/template', type=str)

# Datasets
parser.add_argument('-d', '--dataset', default='cifar10', type=str)
parser.add_argument('-j', '--workers', default=1, type=int, metavar='N',
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
parser.add_argument('--resume', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
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

args.save_path = 'experiments/' + args.dataset + '/' + args.arch + str(args.depth) + '_' + str(args.num_branch) + 'mb_wd' + str(args.weight_decay) + '_cmcscale' + str(args.cmcloss_scale) + '_kd_cmcl'
if not os.path.isdir(args.save_path):
    os.makedirs(args.save_path)

# Validate dataset
assert args.dataset == 'cifar10' or args.dataset == 'cifar100', 'Dataset can only be cifar10 or cifar100.'
# Use CUDA
os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu_id
use_cuda = torch.cuda.is_available()
# Random seed
if args.manualSeed is None:
    args.manualSeed = random.randint(1, 10000)
random.seed(args.manualSeed)
torch.manual_seed(args.manualSeed)
if use_cuda:
    torch.cuda.manual_seed_all(args.manualSeed)
best_acc = 0  # best test accuracy


kd_loss_fun = KDLoss(T=3, eps=1e-10)
cmclloss_v1 = CMCLLoss_v1(topk=1, beta = args.beta)
indeploss = IndependentLoss()
mclloss = MCLLoss(topk=2)

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
    if args.arch.startswith('resnext'):
        model = models.__dict__[args.arch](
                    cardinality=args.cardinality,
                    num_classes=num_classes,
                    depth=args.depth,
                    widen_factor=args.widen_factor,
                    dropRate=args.drop,
                )
    elif args.arch.startswith('densenet'):
        model = models.__dict__[args.arch](
                    num_classes=num_classes,
                    depth=args.depth,
                    growthRate=args.growthRate,
                    compressionRate=args.compressionRate,
                    dropRate=args.drop,
                )
    elif args.arch.startswith('wrn'):
        model = models.__dict__[args.arch](
                    num_classes=num_classes,
                    depth=args.depth,
                    widen_factor=args.widen_factor,
                    dropRate=args.drop,
                )
    elif args.arch.startswith('resnet'):
        model = models.__dict__[args.arch](
                    num_classes=num_classes,
                    depth=args.depth,
                    num_branch=args.num_branch,
                )
    else:
        model = models.__dict__[args.arch](num_classes=num_classes)

    if args.depth == 110:
        gate = Gate(in_planes=128, out_planes=args.num_branch, d=16)
    elif args.depth == 50:
        gate = Gate(in_planes=128, out_planes=args.num_branch, d=16)
    elif args.depth == 20:
        gate = Gate(in_planes=128, out_planes=args.num_branch, d=8)
    elif args.depth == 32 :
        gate = Gate(in_planes=128, out_planes=args.num_branch, d=8)
    gate = gate.cuda()
    model = torch.nn.DataParallel(model).cuda()
    cudnn.benchmark = True
    print('    Total params: %.2fM' % (sum(p.numel() for p in model.parameters())/1000000.0))

    criterion = nn.CrossEntropyLoss()
    #optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)
    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=1e-4)
    gate_optimizer = optim.SGD( gate.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay )

    # Resume
    title = 'cifar-10-' + args.arch
    logger = Logger(os.path.join(args.save_path, 'log.txt'), title=title)
    logger.set_names(['Learning Rate', 'Train Loss', 'Valid Loss', 'Train Acc.', 'Valid Acc.'])

    if args.evaluate:
        print('\nEvaluation train')
        checkpoint = torch.load(args.save_path + '/' + 'model_best.pth.tar')
        model.load_state_dict(checkpoint['state_dict'])
        test_loss, test_acc, predict_domain = test(testloader, model, criterion, start_epoch, use_cuda, args, gate)
        np.savetxt(os.path.join(args.save_path, 'test_domain.txt'), np.array(predict_domain), fmt='%d')
        print(' Test Loss:  %.8f, Test Acc:  %.2f' % (test_loss, test_acc))
        print('\nEvaluation test')
        trainloader = data.DataLoader(trainset, batch_size=args.train_batch, shuffle=False, num_workers=args.workers)
        test_loss, test_acc, predict_domain = test(trainloader, model, criterion, start_epoch, use_cuda, args, gate)
        np.savetxt(os.path.join(args.save_path, 'train_domain.txt'), np.array(predict_domain), fmt='%d')
        print(' Test Loss:  %.8f, Test Acc:  %.2f' % (test_loss, test_acc))
        return

    # Train and val
    for epoch in range(start_epoch, args.epochs):
        adjust_learning_rate(optimizer, epoch)

        print('\nEpoch: [%d | %d] LR: %f' % (epoch + 1, args.epochs, state['lr']))
        train_loss, train_acc = train(trainloader, model, criterion, optimizer, gate_optimizer, use_cuda, args, gate)
        test_loss, test_acc, predict_domain = test(testloader, model, criterion, use_cuda, args, gate)

        logger.append([state['lr'], train_loss, test_loss, train_acc, test_acc])
        # save model
        is_best = test_acc > best_acc
        best_acc = max(test_acc, best_acc)
        save_checkpoint({
                'epoch': epoch + 1, 
                'state_dict': model.state_dict(),
                'acc': test_acc,
                'best_acc': best_acc,
                'optimizer' : optimizer.state_dict(),
                'gate_optimizer' : gate_optimizer.state_dict(),
            }, is_best, checkpoint=args.save_path)

    logger.close()
    print('Best acc:')
    print(best_acc)

def train(trainloader, model, criterion, optimizer, gate_optimizer, use_cuda, args, gate):
    # switch to train mode
    global kd_loss_fun, cmclloss_v1, indeploss, mclloss
    model.train()

    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    top1_branch =[]
    for i in range(args.num_branch):
        top1_branch.append(AverageMeter())
    top5_branch =[]
    for i in range(args.num_branch):
        top5_branch.append(AverageMeter())
    losses_branch = []
    for i in range(args.num_branch): 
        losses_branch.append(AverageMeter())
    end = time.time()

    for batch_idx, (inputs, targets) in enumerate(trainloader):
        # measure data loading time
        data_time.update(time.time() - end)

        if use_cuda:
            inputs, targets = inputs.cuda(), targets.cuda()
        inputs, targets = torch.autograd.Variable(inputs), torch.autograd.Variable(targets)
        # compute output
        outputs, middle_output = model(inputs)
        cmc_loss, oracle_logits, min_index = cmclloss_v1(outputs, targets)
        # get weight for each branch
        weights = gate(middle_output)
        loss_branch = []
        loss_ce_branch =0
        ensemble_output = 0
        for i in range(args.num_branch):
            loss_branch.append(criterion(outputs[i], targets))
            loss_ce_branch += loss_branch[i]
            ensemble_output += weights[:, i:(i+1)] * outputs[i]
        ensemble_loss = criterion(ensemble_output, targets)

        # measure accuracy and record loss
        prec1, prec5 = accuracy(oracle_logits.data, targets.data, topk=(1, 5))
        top1.update(prec1[0], inputs.size(0))
        top5.update(prec5[0], inputs.size(0))

        losses_kl = 0
        losses_ = 0
        # measure accuracy and record loss
        for i in range(args.num_branch):
            prec1, prec5 = accuracy(outputs[i].data, targets.data, topk=(1, 5))
            losses_branch[i].update(loss_branch[i].data[0], inputs.size(0))
            losses_kl += kd_loss_fun(outputs[i], oracle_logits, targets)
            top1_branch[i].update(prec1[0], inputs.size(0))
            top5_branch[i].update(prec5[0], inputs.size(0))
        loss = args.ensemble_scale * ensemble_loss + args.kl_scale * losses_kl / 5 + args.branch_scale * loss_ce_branch + cmc_loss * args.cmcloss_scale
        #loss = cmc_loss
        losses.update(loss.data[0], inputs.size(0))

        # compute gradient and do SGD step
        optimizer.zero_grad()
        gate_optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        gate_optimizer.step()

        if args.num_branch == 1:
            progress_bar(batch_idx, len(trainloader), 'Loss: %.2f | KLloss: %.2f | Top1: %.2f | Top5: %.2f | CMCloss: %.2f ' 
                    % (losses.avg, losses_kl, top1.avg, top5.avg, cmc_loss.item()))
        elif args.num_branch == 2:
            progress_bar(batch_idx, len(trainloader), 'Loss: %.2f | KLloss: %.2f | Top1: %.2f | Top5: %.2f | Top1_1: %.2f | Top1_2: %.2f | CMCloss: %.2f ' 
                    % (losses.avg, losses_kl, top1.avg, top5.avg, top1_branch[0].avg, top1_branch[1].avg, cmc_loss.item()))
        else :
            progress_bar(batch_idx, len(trainloader), 'Loss: %.2f | KLloss: %.2f | Top1: %.2f | Top5: %.2f | Top1_1: %.2f | Top1_2: %.2f | Top1_3: %.2f | CMCloss: %.2f ' 
                    % (losses.avg, losses_kl, top1.avg, top5.avg, top1_branch[0].avg, top1_branch[1].avg, top1_branch[2].avg, cmc_loss.item()))

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

    return (losses.avg, top1.avg)

def test(testloader, model, criterion, use_cuda, args, gate):
    global best_acc
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    top1_branch =[]
    for i in range(args.num_branch):
        top1_branch.append(AverageMeter())
    top5_branch =[]
    for i in range(args.num_branch):
        top5_branch.append(AverageMeter())
    losses_branch = []
    for i in range(args.num_branch): 
        losses_branch.append(AverageMeter())
    # switch to evaluate mode
    model.eval()
    gate.eval()

    top1_branch_scalar = []
    top5_branch_scalar = []
    for i in range(args.num_branch):
        top1_branch_scalar.append(0)
        top5_branch_scalar.append(0)

    predict_domain = []
    end = time.time()
    for batch_idx, (inputs, targets) in enumerate(testloader):
        # measure data loading time
        data_time.update(time.time() - end)

        if use_cuda:
            inputs, targets = inputs.cuda(), targets.cuda()
        inputs, targets = torch.autograd.Variable(inputs, volatile=True), torch.autograd.Variable(targets)

        # compute output
        outputs, middle_output = model(inputs)

        cmc_loss, oracle_logits, min_index = cmclloss_v1(outputs, targets, is_validate=False)
        predict_domain.extend(list(min_index))

        # get weight for each branch
        weights = gate(middle_output)
        loss_branch = []
        ensemble_output = 0
        for i in range(args.num_branch):
            loss_branch.append(criterion(outputs[i], targets))
            ensemble_output += weights[:,i:(i+1)] * outputs[i]
        ensemble_loss = criterion(oracle_logits, targets)

        # measure accuracy and record loss
        prec1, prec5 = accuracy(ensemble_output.data, targets.data, topk=(1, 5))
        losses.update(ensemble_loss.data[0], inputs.size(0))
        top1.update(prec1[0], inputs.size(0))
        top5.update(prec5[0], inputs.size(0))

        # measure accuracy and record loss
        for i in range(args.num_branch):
            prec1, prec5 = accuracy(outputs[i].data, targets.data, topk=(1, 5))
            losses_branch[i].update(loss_branch[i].data[0], inputs.size(0))
            top1_branch[i].update(prec1[0], inputs.size(0))
            top5_branch[i].update(prec5[0], inputs.size(0))


        # measure accuracy and record loss
        for i in range(args.num_branch):
            prec1, prec5 = cal_accuracy_each_class(outputs[i].data, targets.data, topk=(1, 5))
            top1_branch_scalar[i] += prec1
            top5_branch_scalar[i] += prec5

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if args.num_branch == 1:
            progress_bar(batch_idx, len(testloader), 'Loss: %.2f | Top1: %.2f | Top5: %.2f' 
                    % (losses.avg, top1.avg, top5.avg))
        elif args.num_branch == 2:
            progress_bar(batch_idx, len(testloader), 'Loss: %.2f | Top1: %.2f | Top5: %.2f | Top1_1: %.2f | Top1_2: %.2f' 
                    % (losses.avg, top1.avg, top5.avg, top1_branch[0].avg, top1_branch[1].avg))
        else :
            progress_bar(batch_idx, len(testloader), 'Loss: %.2f | Top1: %.2f | Top5: %.2f | Top1_1: %.2f | Top1_2: %.2f | Top1_3: %.2f' 
                    % (losses.avg, top1.avg, top5.avg, top1_branch[0].avg, top1_branch[1].avg, top1_branch[2].avg))

    for i in range(args.num_branch):
        print('branch {}:'.format(str(i)))
        print(top1_branch_scalar[i])

    return (losses.avg, top1.avg, predict_domain)

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
