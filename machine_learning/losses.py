import torch
import torch.nn as nn
import torchvision
import torch.nn.functional as F
from torch.autograd import Variable
from torch.autograd import Function


def calculate_weights(weights, data_loader, device):
    if weights == None:
        loss_weights = None
        return loss_weights
    elif weights == "frequency":
        train_index = (data_loader.validation_dataset.df["set"] == "train")
        classes = (data_loader.validation_dataset.df.loc[train_index,"class"])
        loss_weights = []
        for cl in data_loader.classes:
            loss_weights.append( len(classes)/float((classes == cl).sum()) )
    else:
        loss_weights = []
        for cl in data_loader.classes:
            loss_weights.append( weights[cl]  )
    print("weights for the loss are: ", loss_weights)
    return torch.FloatTensor(loss_weights).to(device)

def one_hot_embedding(labels, num_classes):
    '''Embedding labels to one-hot form.
    Args:
      labels: (LongTensor) class labels, sized [N,].
      num_classes: (int) number of classes.
    Returns:
      (tensor) encoded labels, sized [N,#classes].
    '''
    y = torch.eye(num_classes)  # [D,D]
    return y[labels]            # [N,D]



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

def get_loss(ml_configs, data_loader):
    """
    function for getting the loss:
    Args:
        ml_config(dict): machine learning config
    """
    loss_function = ml_configs["loss_function"]
    
    if loss_function=="MSELoss":
        criterion = nn.MSELoss() 
    elif loss_function== "NLLLoss": 
        loss_weights = calculate_weights(   ml_configs["weights"] , 
                                        data_loader, 
                                        ml_configs["device"])
        criterion = nn.NLLLoss(weight = loss_weights)
    elif loss_function== "CrossEntropyLoss":
        loss_weights = calculate_weights(   ml_configs["weights"] , 
                                        data_loader, 
                                        ml_configs["device"])
        criterion = nn.CrossEntropyLoss(weight = loss_weights)
    elif loss_function== "FocalLoss":
        criterion = FocalLoss2d(len(data_loader.classes), ml_configs["device"] )
    return criterion