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
    def __init__(self, T=5, alpha=1.0, eps=1e-8, num_classes=10, threshold=0.1):
        super(KDLoss, self).__init__()
        self.eps=eps
        self.T = T
        self.alpha = alpha
    def forward(self, pred_logits, gt_logits, labels):
        p = F.softmax(pred_logits / self.T, dim=1)
        q = F.softmax(gt_logits / self.T, dim=1)
        loss_kl = torch.mean( -torch.dot(q.view(-1), (torch.log((p+self.eps) / (q+self.eps))).view(-1))) * self.alpha + F.cross_entropy(pred_logits, labels) * (1. - self.alpha)
        return loss_kl

# knowledge distillation
class PartCElossbyRemoveLowConfidenceData(nn.Module):
    def __init__(self, T=5, alpha=1.0, eps=1e-8, num_classes=10, threshold=0.1):
        super(PartCElossbyRemoveLowConfidenceData, self).__init__()
        self.eps=eps
        self.T = T
        self.alpha = alpha
        self.threshold = threshold
    def forward(self, pred_logits, gt_logits, labels):
        origin_q = F.softmax(gt_logits / self.T, dim=1)
        gt_prob = origin_q[range(labels.size(0)), labels.detach()]
        selected_index = gt_prob > self.threshold
        
        #pdb.set_trace()
        loss_pce = F.cross_entropy(pred_logits[selected_index, :], labels[selected_index])
        return loss_pce


# knowledge distillation
class PartKDlossbyRemoveLowConfidenceData(nn.Module):
    def __init__(self, T=5, alpha=1.0, eps=1e-8, num_classes=10, threshold=0.1):
        super(PartKDlossbyRemoveLowConfidenceData, self).__init__()
        self.eps=eps
        self.T = T
        self.alpha = alpha
        self.threshold = threshold
    def forward(self, pred_logits, gt_logits, labels):
        origin_q = F.softmax(gt_logits / self.T, dim=1)
        gt_prob = origin_q[range(labels.size(0)), labels.detach()]
        selected_index = gt_prob > self.threshold
        p = F.softmax(pred_logits[selected_index, :] / self.T, dim=1)
        q = F.softmax(gt_logits[selected_index, :] / self.T, dim=1)
        loss_pkd = torch.mean( -torch.dot(q.view(-1), (torch.log((p+self.eps) / (q+self.eps))).view(-1))) * self.alpha + F.cross_entropy(pred_logits, labels) * (1. - self.alpha)
        #pdb.set_trace()
        return loss_pkd

# knowledge distillation
class AntiCurriculumKDLoss(nn.Module):
    def __init__(self, T=5, alpha=1.0, eps=1e-8, num_classes=10):
        super(AntiCurriculumKDLoss, self).__init__()
        self.eps=eps
        self.T = T
        self.alpha = alpha
        self.num_classes=num_classes
    def forward(self, pred_logits, gt_logits, labels):
        # for each sample, compute the T
        temp = F.softmax(gt_logits/self.T, dim=1)
        index = labels.detach()
        #pdb.set_trace()
        cl_T = temp[ range(labels.size(0)), index ]
        cl_T = cl_T.view(-1, 1)
        cl_T = cl_T.repeat(1, gt_logits.size(1))
        cl_T = self.T / torch.mean(cl_T) * cl_T
        p = F.softmax(pred_logits / cl_T, dim=1)
        q = F.softmax(gt_logits / cl_T, dim=1)
        loss_kl = torch.mean( -torch.dot(q.view(-1), (torch.log((p+self.eps) / (q+self.eps))).view(-1))) * self.alpha + F.cross_entropy(pred_logits, labels) * (1. - self.alpha)
        return loss_kl

# knowledge distillation
class CurriculumKDLoss(nn.Module):
    def __init__(self, T=5, alpha=1.0, eps=1e-8, num_classes=10):
        super(CurriculumKDLoss, self).__init__()
        self.eps=eps
        self.T = T
        self.alpha = alpha
        self.num_classes=num_classes
    def forward(self, pred_logits, gt_logits, labels):
        # for each sample, compute the T
        temp = F.softmax(gt_logits/self.T, dim=1)
        index = labels.detach()
        #pdb.set_trace()
        cl_T = temp[ range(labels.size(0)), index ]
        cl_T = cl_T.view(-1, 1)
        cl_T = cl_T.repeat(1, gt_logits.size(1))
        cl_T = torch.mean(cl_T) * self.T / cl_T
        p = F.softmax(pred_logits / cl_T, dim=1)
        q = F.softmax(gt_logits / cl_T, dim=1)
        loss_kl = torch.mean( -torch.dot(q.view(-1), (torch.log((p+self.eps) / (q+self.eps))).view(-1))) * self.alpha + F.cross_entropy(pred_logits, labels) * (1. - self.alpha)
        return loss_kl

# label smoothing loss for cross-entropy
class LSRLoss(nn.Module):
    def __init__(self, eps=1e-10, gamma=1, K=100):
        super(LSRLoss, self).__init__()
        self.eps=eps
        self.uniforms = float(1.0/float(K))
        self.gamma = gamma
    def forward(self, pred_logits, labels):
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



# independent loss for ensemble leanring, used in ensemble methods
class IndependentLoss(nn.Module):
    def __init__(self, eps=1e-10):
        super(IndependentLoss, self).__init__()
        self.eps=eps

    def forward(self, pred_logits_list, targets):
        ce_loss = 0
        for i in range(len(pred_logits_list)):
            ce_loss.append( F.cross_entropy(pred_logits_list[i], targets) )
        total_loss = torch.sum(torch.stack(ce_loss))
        return total_loss

# multiple  choice learning loss, used in ensemble methods
class MCLLoss(nn.Module):
    def __init__(self, eps=1e-10, topk=1):
        super(MCLLoss, self).__init__()
        self.eps=eps
        self.topk=topk

    def forward(self, pred_logits_list, targets):
        ce_loss = 0
        for i in range(len(pred_logits_list)):
            ce_loss.append( F.cross_entropy(pred_logits_list[i], targets, reduction='none') )
        ce_loss = torch.stack(ce_loss)
        total_loss = torch.sum( torch.mean(ce_loss, dim=0) )
        min_values, min_index = torch.topk(-ce_loss.t(), self.topk)
        total_loss -= torch.sum(min_values) / targets.size(0)
        return total_loss

# confidence multiple  choice learning loss, used in ensemble methods
class CMCLLoss_v1(nn.Module):
    def __init__(self, eps=1e-10, topk=1, beta=1.0, num_classes=100):
        super(CMCLLoss_v1, self).__init__()
        self.eps=eps
        self.topk=topk
        self.beta = beta
        self.num_classes = num_classes
        self.curr_step = 0

    def forward(self, pred_logits_list, targets, is_validate=False):
        ce_loss = []
        for i in range(len(pred_logits_list)):
            if is_validate == False:
                ce_loss.append( F.cross_entropy(pred_logits_list[i], targets, reduction='none') )
            else :
                ce_loss.append(torch.mean(F.log_softmax(pred_logits_list[i]+self.eps, dim=1), dim=1))
        ce_loss = torch.stack(ce_loss)
        # entropy_list = [ torch.log(len(pred_logits_list)) + torch.mean(F.log_softmax(logits+self.eps, dim=1), dim=1)  for logits in pred_logits_list ]
        entropy_list = [ -math.log(self.num_classes) - torch.mean(F.log_softmax(logits+self.eps, dim=1), dim=1)  for logits in pred_logits_list ]
        loss_list = []
        for m in range(len(pred_logits_list)):
            if m == 0:
                loss_list.append(ce_loss[m] + self.beta*( torch.sum(torch.stack(entropy_list[(m+1):]), dim=0) ))
            elif m ==  len(pred_logits_list)-1:
                loss_list.append( ce_loss[m] + self.beta*( torch.sum(torch.stack(entropy_list[:m]), dim=0) ))
            else :
                loss_list.append( ce_loss[m] + self.beta*( torch.sum(torch.stack(entropy_list[:m] + entropy_list[(m+1):]), dim=0) ))
        loss_list = torch.stack(loss_list)
        min_values, min_index = torch.topk( -loss_list.t(), self.topk )
        min_index = min_index.cpu().numpy().T
        #min_index = min_index.detach()
        entropy_list = torch.stack(entropy_list)
        # here we return the oracle logits from multiple prediction
        logits_ensemble = torch.stack(pred_logits_list)
        oracle_logits = logits_ensemble[ min_index[0,:], range(targets.size(0)) ]

        mask = np.zeros((len(pred_logits_list), targets.size(0)))
        mask[ min_index.reshape(-1), np.tile(range(targets.size(0)), self.topk) ] = 1
        mask = torch.from_numpy(mask)
        mask = mask.type(torch.FloatTensor)
        new_loss = torch.sum(mask.cuda() * (ce_loss - self.beta * entropy_list )) / targets.size(0)
        new_loss += torch.sum(self.beta * entropy_list) / targets.size(0)

        min_index = torch.from_numpy(min_index[0])
        return new_loss, oracle_logits, min_index
