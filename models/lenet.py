
import torch
import torchvision
from torchvision.models import alexnet, resnet18, densenet121, squeezenet1_0
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init


class LeNet(nn.Module):
    def __init__(self,num_channels ,num_classes, droprate=0.5):
        super(LeNet, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(num_channels,  6, 3),
            nn.ReLU(inplace=True), 
            nn.MaxPool2d(2, 2),
            nn.Conv2d(6, 16, kernel_size=5, padding=2),
            nn.ReLU(inplace=True), 
            nn.MaxPool2d(2), 
            nn.Flatten() 
        )

        self.classifier= nn.Sequential(
            nn.Linear(784, 120),
            nn.ReLU(inplace=True),
            nn.Dropout(p=droprate),
            nn.Linear(120, 84),
            nn.ReLU(inplace=True),
            nn.Dropout(p=droprate),
            nn.Linear(84, num_classes)
        ) 
        
    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x

