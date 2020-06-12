import torch
import torch.nn as nn
import torchvision
import torch.nn.functional as F
from torch.autograd import Variable
from torch.autograd import Function
from losses.FocalLoss2d import *
import logging

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
    logging.info("weights for the loss are: %s" % loss_weights)
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


def get_loss(loss_configs, data_loader, device):
    """
    function for getting the loss:
    Args:
        ml_config(dict): machine learning config
    """
    loss_function = loss_configs["loss_function"] 
    
    if loss_function=="MSELoss":
        criterion = nn.MSELoss() 
    
    elif loss_function== "NLLLoss": 
        loss_weights = calculate_weights(   loss_configs["weights"] , 
                                        data_loader, 
                                        device)
        criterion = nn.NLLLoss(weight = loss_weights)
    
    elif loss_function== "CrossEntropyLoss":
        loss_weights = calculate_weights(   loss_configs["weights"] , 
                                        data_loader, 
                                        device)
        criterion = nn.CrossEntropyLoss(weight = loss_weights)
    elif loss_function== "FocalLoss":
        criterion = FocalLoss2d(len(data_loader.classes), device )
    return criterion