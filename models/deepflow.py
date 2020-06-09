
import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F


class SimpleConvFactory(nn.Module):

    def __init__(self, in_channels, out_channel, kernel, pad, stride):
        super(SimpleConvFactory, self).__init__()
        self.simple_conv_factory = nn.Sequential(
            nn.Conv2d(in_channels, out_channel, kernel_size=kernel, padding=pad, stride=stride),
            nn.BatchNorm2d(out_channel),
            nn.ReLU(inplace=True)            
        )

    def forward(self, x):
        x = self.simple_conv_factory(x)
        return x

class DualFactory(nn.Module):

    def __init__(self, in_channels, out_channel_1, kernel_1, pad_1,stride_1,
                                    out_channel_3, kernel_3, pad_3,stride_3):
        super(DualFactory, self).__init__()
        self.simple_conv_factory_1 = nn.Sequential(
            nn.Conv2d(in_channels, out_channel_1, kernel_size=kernel_1, 
                                    padding=pad_1, stride=stride_1),
            nn.BatchNorm2d(out_channel_1),
            nn.ReLU(inplace=True)            
        )
        self.simple_conv_factory_3 = nn.Sequential(
            nn.Conv2d(in_channels, out_channel_3, kernel_size=kernel_3, padding=pad_3,stride=stride_3),
            nn.BatchNorm2d(out_channel_3),
            nn.ReLU(inplace=True)            
        )
    def forward(self, x):
        y_1 = self.simple_conv_factory_1(x)
        y_3 = self.simple_conv_factory_3(x)
        x = torch.cat((y_1, y_3), dim = 1)
        return x

class DualDownsampleFactory(nn.Module):

    def __init__(self, in_channels, out_channel_3, kernel_3 = 3, pad_3 = 1, stride_3 = 2):
        super(DualDownsampleFactory, self).__init__()
        self.simple_conv_factory_3 = nn.Sequential(
            nn.Conv2d(  in_channels, 
                        out_channel_3, 
                        kernel_size=kernel_3, 
                        padding=pad_3,
                        stride = stride_3),
            nn.BatchNorm2d(out_channel_3),
            nn.ReLU(inplace=True)            
        )
        self.pool_3 = nn.Sequential(
            nn.MaxPool2d(kernel_size = kernel_3, stride = stride_3, padding = pad_3)           
        )
    def forward(self, x):
        y_1 = self.simple_conv_factory_3(x)
        y_3 = self.pool_3(x)
        x = torch.cat((y_1, y_3), dim = 1)
        return x


class DeepFlow(nn.Module):
    def __init__(self,num_channels ,  num_classes):
        super(DeepFlow, self).__init__()
        
        self.conv1 = SimpleConvFactory(num_channels, 96, 3, 1, 1)

        self.in3a = DualFactory(96,     32, 1, 0,1,
                                        32, 3, 1,1)
        self.in3b = DualFactory(32+32,  32, 1, 0,1,
                                        48, 3, 1,1)
        self.in3c = DualDownsampleFactory(32+48, 80)

        self.in4a = DualFactory(80+80,     112, 1, 0,1,
                                        48, 3, 1,1)
        self.in4b = DualFactory(112+48,     96, 1, 0,1,
                                        64, 3, 1,1)
        self.in4c = DualFactory(96+64,     80, 1, 0,1,
                                        80, 3, 1,1)
        self.in4d = DualFactory(80+80,     48, 1, 0,1,
                                        96, 3, 1,1)
        self.in4e = DualDownsampleFactory(48+96, 96)

        self.in5a = DualFactory(240,  176, 1, 0,1,
                                        160, 3, 1,1)
        
        self.in5b = DualFactory(176+160,    176, 1, 0,1,
                                            160, 3, 1,1)

        self.in6a = DualDownsampleFactory(176+160, 96)

        self.in6b = DualFactory(432,  176, 1, 0,1,
                                        160, 3, 1,1)
        self.in6c = DualFactory(176+160,    176, 1, 0,1,
                                            160, 3, 1,1)
        
        self.classifier =nn.Sequential(
            nn.AvgPool2d(kernel_size = 8),
            nn.Flatten(),
            nn.Linear(336,num_classes)           
        )
    
    def forward(self,x):
        x = self.conv1(x)
        x = self.in3a(x)
        x = self.in3b(x)
        x = self.in3c(x)
        x = self.in4a(x)
        x = self.in4b(x)
        x = self.in4c(x)
        x = self.in4d(x)
        x = self.in4e(x)
        x = self.in5a(x)
        x = self.in5b(x)
        x = self.in6a(x)
        x = self.in6b(x)
        x = self.in6c(x)
        x = self.classifier(x)
        return x

