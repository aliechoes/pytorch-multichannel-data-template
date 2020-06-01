import torch.nn as nn
import torchvision
import torch.nn.functional as F
  
from torch.autograd import Function

def get_loss(ml_configs):
    """
    function for getting the loss:
    Args:
        ml_config(dict): machine learning config
    """
    loss_function = ml_configs["loss_function"]
    
    if loss_function=="MSELoss":
        criterion = nn.MSELoss() 
    elif loss_function== "NLLLoss": 
        criterion = nn.NLLLoss()
    elif loss_function== "CrossEntropyLoss":
        criterion = nn.CrossEntropyLoss()
    return criterion