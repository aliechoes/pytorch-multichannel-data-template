import torch.nn as nn
import torchvision
import torch.nn.functional as F
  
def get_loss(loss_function):
    if loss_function=="mse":
        criterion = nn.MSELoss()
        return criterion
    if loss_function== "nll_loss": 
        criterion = nn.NLLLoss()
        return criterion