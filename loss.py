#!/usr/bin/env python
# encoding: utf-8
'''
Training script for CIFAR-10/100
Copyright (c) Baoyun Peng, 2018
'''

import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np
import math
import pdb

eps = 1e-8

class L2Loss(nn.Module):
    def __init__(self, div_element = False):
        super(L2Loss, self).__init__()
        self.div_element = div_element
    def forward(self, output, target):
        loss = torch.sum(torch.pow(torch.add(output, -1, target), 2) )
        if self.div_element:
            loss = loss / output.numel()
        else:
            loss = loss / output.size(0) / 2
        return loss


# knowledge distillation
class KDLoss(nn.Module):
    def __init__(self, T=5, alpha=1.0, eps=1e-8):
        super(KDLoss, self).__init__()
        self.eps=eps
        self.T = T
        self.alpha = alpha
    def forward(self, pred_logits, gt_logits, labels):
        p = F.softmax(pred_logits / self.T, dim=1)
        q = F.softmax(gt_logits / self.T, dim=1)
        loss_kl = torch.mean( -torch.dot(q.view(-1), (torch.log((p+self.eps) / (q+self.eps))).view(-1))) * self.alpha + F.cross_entropy(pred_logits, labels) * (1. - self.alpha)
        return loss_kl

# label smoothing loss for cross-entropy
class LSRLoss(nn.Module):
    def __init__(self, eps=1e-10, gamma=1, K=100):
        super(LSRLoss, self).__init__()
        self.eps=eps
        self.uniforms = float(1.0/float(K))
        self.gamma = gamma
    def forward(self, pred_logits, gt_logits, labels):
        p = F.softmax(pred_logits, dim=1)
        q = F.softmax( self.uniforms * torch.ones(labels.size(0)), dim=1)
        loss_lsr = self.gamma * torch.mean( -torch.dot(q.view(-1), (torch.log((p+eps) / (q+eps))).view(-1))) + F.cross_entropy(pred_logits, labels)
        return loss_lsr

class WeightedCrossEntropy(nn.Module):
    def __init__(self, T=5):
        super(WeightedCrossEntropy, self).__init__()
        self.softmax = nn.Softmax()
    def forward(self, pred_logits, gt_logits, labels):
        ce_losses = F.cross_entropy(pred_logits, labels, reduction='none')
        gt_probs = self.softmax(gt_logits)
        index = labels.detach()
        weights = gt_probs[ range(labels.size(0)), index ]
        #weights = torch.topk(self.softmax(gt_logits), 1)[0]
        loss = torch.mean(weights.detach() * ce_losses)
        return loss

# self-paced knowledge distillation as weights cross entropy
class SPKDCrossEntropy(nn.Module):
    def __init__(self, T=5, alpha=1.0):
        super(SPKDCrossEntropy, self).__init__()
        self.softmax = nn.Softmax()
        self.alpha = alpha
    def forward(self, pred_logits, gt_logits, labels):
        ce_losses = F.cross_entropy(pred_logits, labels, reduction='none')
        weights = torch.topk(self.softmax(gt_logits), 1)[0]
        loss = torch.mean( (1-self.alpha)*weights.detach()*ce_losses + self.alpha*(1-weights.detach())*ce_losses )
        return loss