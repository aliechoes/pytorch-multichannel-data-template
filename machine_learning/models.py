
import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F



class ShallowNet(nn.Module):
    def __init__(self,num_channels ,num_classes):
        super(ShallowNet,self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(num_channels, 10, kernel_size=5),
            nn.MaxPool2d(kernel_size=2),
            nn.ReLU(inplace=True),
            nn.Conv2d(10, 20, kernel_size=5),
            nn.MaxPool2d(kernel_size=2),
            nn.ReLU(inplace=True),
            
        )
        self.embedding = nn.Sequential(
            nn.Dropout2d(),
            nn.Linear(320, 50),
            nn.ReLU(inplace=True),
            
        ) 
        self.classifier= nn.Sequential(
            nn.Linear(50, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        x = x.view(-1, 320)
        x = self.embedding(x)
        x = self.classifier(x)
        x = F.log_softmax(x, dim=1)
        return x

    def embedding_generator(self, x): 
        x = self.features(x)
        x = x.view(-1, 320)
        x = self.embedding(x)
        return x



class AlexNet(nn.Module):

    def __init__(self, num_channels ,num_classes):
        super(AlexNet, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(num_channels , 64, kernel_size=11, stride=4, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(64, 192, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(192, 384, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(384, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
        )
        self.avgpool = nn.AdaptiveAvgPool2d((6, 6))
        self.embedding = nn.Sequential(
            nn.Dropout(),
            nn.Linear(256 * 6 * 6, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(4096, 1024),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(1024, 128),
            nn.ReLU(inplace=True)
        )
        self.classifier = nn.Sequential(
            nn.Linear(128, num_classes),

        )
        
    def embedding_generator(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.embedding(x)
        return x

    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.embedding(x)
        x = self.classifier(x)
        x = F.log_softmax(x, dim=1)
        return x


def get_model(ml_config, checkpoint ,num_channels ,num_classes ):
    model_name = ml_config["model_name"]
    device = ml_config["device"]
    
    if model_name == "AlexNet":
        model = AlexNet(num_channels ,num_classes)
        model.image_size = 256
        model.intution_layer = "features"
    if model_name == "ShallowNet":

        model = ShallowNet(num_channels ,num_classes) 
        model.image_size = 28
        model.intution_layer = "features"
    
    # transfer learning
    if checkpoint is not None:
        model.load_state_dict(checkpoint['model_state_dict'])

    model = model.to(device)
    print(model)
    return model 