import torch
import torch.nn as nn
import torchvision
import torch.nn.functional as F
from torch.autograd import Variable
from torch.autograd import Function


class FocalLoss2d(nn.modules.loss._WeightedLoss):

    def __init__(self, num_classes, device ,gamma=2, weight=None, size_average=None, ignore_index=-100,
                 reduce=None, reduction='mean', balance_param=0.25):
        super(FocalLoss2d, self).__init__(weight, size_average, reduce, reduction)
        self.num_classes = num_classes
        self.gamma = gamma
        self.weight = weight
        self.size_average = size_average
        self.ignore_index = ignore_index
        self.balance_param = balance_param
        self.device = device

    def forward(self, input, labels):
        target = one_hot_embedding(labels, self.num_classes)
        target = target.to(self.device)
        # inputs and targets are assumed to be BatchxClasses
        assert len(input.shape) == len(target.shape)
        assert input.size(0) == target.size(0)
        assert input.size(1) == target.size(1)
        
        #weight = Variable(self.weight)
        
        # compute the negative likelyhood
        logpt = - F.binary_cross_entropy_with_logits(input, target, reduction=self.reduction)
        pt = torch.exp(logpt)

        # compute the loss
        focal_loss = -( (1-pt)**self.gamma ) * logpt
        balanced_focal_loss = self.balance_param * focal_loss
        return balanced_focal_loss

