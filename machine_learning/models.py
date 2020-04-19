
import torch
import torchvision
from torchvision.models import alexnet, resnet18
import torch.nn as nn
import torch.nn.functional as F

def pretrained_weights(model, weights):

    # in our code, it is always the last layer will be classifier 
    # and before that it is embedding
    for w in list(weights.keys())[0:-1]:
        layer_name = w
        if 'classifier' in w:
            layer_name = layer_name.replace('classifier', 'embedding')
        model.state_dict()[layer_name] = weights[w]
    return model

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
            nn.Flatten() 
        )

        self.classifier= nn.Sequential(
            nn.Dropout2d(),
            nn.Linear(320, 50),
            nn.ReLU(inplace=True),
            nn.Linear(50, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x

def get_model(ml_config, checkpoint ,num_channels ,num_classes ):
    """
    function for getting the model:
    Args:
        ml_config(dict): machine learning config
        checkpoint(dict): in case of transfer learning
        num_channels(int): number of channels to initiate the network
        num_classes(int): number of classes to initiate the network

    remark: `intution_layer` is grad-cam. It will be added to the code later
    """
    model_name = ml_config["model_name"].lower()
    device = ml_config["device"]
    
    ## TODO: add grad-cam
    if model_name == "alexnet":
        model = alexnet(pretrained=True)
        model.image_size = 256
        model.intuition_layer = "features"
        ## loading the imagenet weights in case it is possible
        if num_channels != 3:
            model.features[0] = nn.Conv2d(num_channels, 64, 
                        kernel_size=(11, 11), stride=(4, 4), padding=(2, 2))
        num_ftrs = model.classifier[6].in_features
        model.classifier[6] = nn.Linear(num_ftrs, num_classes)
        model.embedding_generator = "nn.Sequential( model.features, \
                                                    model.avgpool, \
                                                    nn.Flatten(), \
                                                    model.classifier[:-1])"

    ## TODO: add grad-cam
    if model_name == "resnet18":
        model = resnet18(pretrained=True)
        model.image_size = 224
        model.intuition_layer = "features"
        ## loading the imagenet weights in case it is possible
        if num_channels != 3:
            model.conv1 = nn.Conv2d(num_channels, 64, kernel_size=(7, 7), 
                            stride=(2, 2), padding=(3, 3), bias=False)
        num_ftrs = model.fc.in_features
        model.fc = nn.Linear(num_ftrs, num_classes)
        model.embedding_generator = "nn.Sequential(*list(model.children())[:-1])"
        
    if model_name == "shallownet":
        model = ShallowNet(num_channels ,num_classes) 
        model.image_size = 28
        model.intution_layer = "features"
        model.embedding_generator = "nn.Sequential( model.features, \
                                                    model.classifier[:-1])"
    
    # transfer learning
    if checkpoint is not None:
        model.load_state_dict(checkpoint['model_state_dict'])

    model = model.to(device)
    print(model)
    return model 