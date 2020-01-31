import torch.nn as nn
import torchvision
import torch

   
def get_optimizer(optimization_method, model,parameters): 
 
    if optimization_method=="adam": 
        lr = parameters["lr"] 
        weight_decay = parameters["weight_decay"] 
        optimizer = torch.optim.Adam(model.parameters(), 
                                        lr=lr,
                                        weight_decay=weight_decay) 
        return optimizer