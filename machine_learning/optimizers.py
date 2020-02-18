import torch.nn as nn
import torchvision
import torch

   
def get_optimizer(optimization_method, model,parameters,
                        checkpoint): 
 
    if optimization_method=="adam": 
        lr = parameters["lr"] 
        weight_decay = parameters["weight_decay"] 
        optimizer = torch.optim.Adam(model.parameters(), 
                                        lr=lr,
                                        weight_decay=weight_decay) 
    
    if checkpoint is not None:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    
    return optimizer