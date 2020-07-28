
import torch
import torchvision
import torch.nn.functional as F
import torch.nn as nn 
import logging
from torch.optim.lr_scheduler import ReduceLROnPlateau, StepLR

class Null_Scheduler():
    def __init__(self, optimizer):
        logging.info("No Scheduler is used")
        self.lr = optimizer.state_dict()["param_groups"][0]["lr"]
    def state_dict(self):
        return {"lr": self.lr}    
    def step(self):
        pass


class GetLRScheduler():
    def __init__(self,optimizer,lr_scheduler_config):
        self.scheduler_type = lr_scheduler_config["type"]
        self.parameters = lr_scheduler_config["args"]
        self.get_lr_scheduler(optimizer)

    def get_lr_scheduler(self, optimizer): 
        if self.scheduler_type == "StepLR":
            self.scheduler = StepLR(optimizer, **self.parameters)
        elif self.scheduler_type == "ReduceLROnPlateau":
            self.scheduler = ReduceLROnPlateau(optimizer, **self.parameters) 
        else:
           self. scheduler = Null_Scheduler(optimizer) 
        
    def step(self,value):
        if self.scheduler_type == "StepLR":
            self.scheduler.step()
        elif self.scheduler_type == "ReduceLROnPlateau":
            self.scheduler.step(value)
    