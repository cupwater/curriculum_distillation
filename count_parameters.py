'''
Training script for CIFAR-10/100
Copyright (c) Baoyun Peng, 2018
'''
from __future__ import print_function

import argparse
import random
import numpy as np
import torch
import models.cifar as models
from thop import profile
model_names = sorted(name for name in models.__dict__
    if name.islower() and not name.startswith("__")
    and callable(models.__dict__[name]))

def main():
    parser = argparse.ArgumentParser(description='count parameters of model')
    parser.add_argument('--arch', '-a', metavar='ARCH', default='resnet20',
                        choices=model_names,
                        help='model architecture: ' +
                            ' | '.join(model_names) +
                            ' (default: resnet18)')
    parser.add_argument('--depth', type=int, default=29, help='Model depth.')
    args = parser.parse_args()
    print("==> creating model '{}'".format(args.arch))
    if args.arch.startswith('resnet'):
        model = models.__dict__[args.arch](
                    num_classes=100,
                    depth=args.depth,
                )
    else:
        model = models.__dict__[args.arch](num_classes=100)

    input = torch.randn(1, 3, 32, 32)
    model_flops, model_params = profile(model, inputs=(input, ))
    print('{} flops is: {}, params is: {}'.format(args.arch + str(args.depth), str(model_flops), str(model_params)))

def count_parameters(arch, depth):
    print("==> creating model '{}'".format(arch))
    if arch.startswith('resnet'):
        model = models.__dict__[arch](
                    num_classes=100,
                    depth=depth,
                )
    else:
        model = models.__dict__[arch](num_classes=100)

    input = torch.randn(1, 3, 32, 32)
    model_flops, model_params = profile(model, inputs=(input, ))
    return model_flops, model_params
if __name__ == '__main__':
    main()
