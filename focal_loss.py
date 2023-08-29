import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

device = torch.device('cuda:0')

class FocalLoss(nn.Module):

    def __init__(self, weight=None, reduction='mean', gamma=2, a=1,eps=1e-7):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.alpha = [5,1,1,1,1, 1,1,1,1,1, 1,1,5,1,1, 1,1]
        # self.alpha = torch.tensor([0.265, 0.03, 0.03, 0.04, 0.03, 0.04, 0.03, 0.03, 0.03, 0.03, 0.03, 0.03, 0.265, 0.03, 0.03, 0.03, 0.03]).to(device)
        # self.alpha = torch.tensor([a,1,1,a,1, 1,1,1,1,1, 1,1,a,a,1, 1,1]).to(device)
        # self.eps = eps
        # self.ce = torch.nn.CrossEntropyLoss(weight=weight, reduction=reduction)

    def forward(self, input, target):
        log_p = F.log_softmax(input)   # (128,17)
        # print('log_p',log_p.shape)
        # print('target',target.shape)
        pt = target * (log_p)  # 点乘     (128,17)
        # print('pt',pt.shape)
        sub_pt = 1-pt               # (128,)
        # print('sub',sub_pt.shape)

        fl = -(sub_pt) ** self.gamma * log_p
        # print('fl',fl.shape)
        return fl.mean()
        # logp = self.ce(input, target)
        # print(logp)
        # p = torch.exp(-logp)
        # print(p)
        # loss = (1 - p) ** self.gamma * logp
        # return loss.mean()

class FocalLoss2(nn.Module):

    def __init__(self, weight=None, reduction='mean', alpha=1,gamma=2, eps=1e-7):
        super(FocalLoss2, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.eps = eps
        self.ce = torch.nn.CrossEntropyLoss(weight=weight, reduction=reduction)

    def forward(self, input, target):
        logp = self.ce(input, target)
        p = torch.exp(-logp)  #因CE中取了log，所以要exp回来，得到概率，因为输入的并不是概率，celoss自带softmax转为概率形式
        loss = (1 - p) ** self.gamma * logp
        return loss.mean()
