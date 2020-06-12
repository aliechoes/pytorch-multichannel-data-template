import torch
import torchvision
from torchvision.models import alexnet, resnet18, densenet121, squeezenet1_0
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
from models.lenet import LeNet
from models.deepflow import DeepFlow
from models.weight_initialization import weight_init
import logging

def load_checkpoint(model, device, checkpoint):
    if torch.cuda.device_count() > 1:
        logging.info("Data Parallelism is on! %d GPUs are detected" % \
                                            torch.cuda.device_count())
    model = nn.DataParallel(model)
    model = model.to(device)
        # transfer learning
    if checkpoint is not None:
        model = model.load_state_dict(checkpoint['model_state_dict'])
        logging.info("preivously trained model is used")

    return model

def get_model(ml_config,device, checkpoint ,num_channels ,num_classes ):
    """
    function for getting the model:
    Args:
        ml_config(dict): machine learning config
        checkpoint(dict): in case of transfer learning
        num_channels(int): number of channels to initiate the network
        num_classes(int): number of classes to initiate the network

    remark: `intution_layer` is grad-cam. It will be added to the code later
    """
    model_name = ml_config["network"].lower() 
    
    ## TODO: add grad-cam
    if model_name == "alexnet":
        model = alexnet(pretrained=True)
        ## loading the imagenet weights in case it is possible
        if num_channels != 3:
            model.features[0] = nn.Conv2d(num_channels, 64, 
                        kernel_size=(11, 11), stride=(4, 4), padding=(2, 2))
        num_ftrs = model.classifier[6].in_features
        model.classifier[6] = nn.Linear(num_ftrs, num_classes)
        model = load_checkpoint(model, device, checkpoint)
        model.image_size = 256
        model.intuition_layer = "features"
        model.embedding_generator = "nn.Sequential( model.features, \
                                                    model.avgpool, \
                                                    nn.Flatten(), \
                                                    model.classifier[:-1], \
                                                    nn.Flatten())"

    ## TODO: add grad-cam
    if model_name == "resnet18":
        model = resnet18(pretrained=True)
        ## loading the imagenet weights in case it is possible
        if num_channels != 3:
            model.conv1 = nn.Conv2d(num_channels, 64, kernel_size=(7, 7), 
                            stride=(2, 2), padding=(3, 3), bias=False)
        num_ftrs = model.fc.in_features
        model.fc = nn.Linear(num_ftrs, num_classes)
        model = load_checkpoint(model, device, checkpoint)
        model.image_size = 224
        model.intuition_layer = "features"
        model.embedding_generator = "nn.Sequential(*list(model.children())[:-1], \
                                                    nn.Flatten())"

    if model_name == "lenet":
        model = LeNet(num_channels ,num_classes) 
        model = model.apply(weight_init)
        model = load_checkpoint(model, device, checkpoint)
        model.image_size = 32
        model.intution_layer = "features"
        model.embedding_generator = "nn.Sequential( model.features, \
                                                    model.classifier[:-1], \
                                                    nn.Flatten())"
    ## TODO: add grad-cam
    if model_name == "densenet121":
        model = densenet121(pretrained=True, drop_rate = 0.5)
        ## loading the imagenet weights in case it is possible
        if num_channels != 3:
            model.features.conv0 = nn.Conv2d(num_channels, 64, kernel_size=(7, 7), 
                            stride=(2, 2), padding=(3, 3), bias=False)
        num_ftrs = model.classifier.in_features
        model.classifier = nn.Linear(num_ftrs, num_classes)
        model = load_checkpoint(model, device, checkpoint)
        model.image_size = 224
        model.intuition_layer = "features"
        model.embedding_generator = "nn.Sequential(model.features), \
                                                    nn.Flatten())"
    
    if model_name == "squeezenet1_0":
        model = squeezenet1_0(pretrained=True)
        ## loading the imagenet weights in case it is possible
        if num_channels != 3:
            model.features.conv0 = nn.Conv2d(num_channels, 96, kernel_size=7, stride=2)
        model.classifier[1] = nn.Conv2d(512, num_classes, kernel_size=(1,1), stride=(1,1))
        model = load_checkpoint(model, device, checkpoint)
        model.num_classes = num_classes
        model.image_size = 224
        model.intuition_layer = "features"
        model.embedding_generator = "nn.Sequential(model.features), \
                                                    nn.Flatten())"

    if model_name == "deepflow":
        model = DeepFlow(num_channels ,num_classes) 
        model = load_checkpoint(model, device, checkpoint)
        model = model.apply(weight_init)
        model.image_size = 64
        model.intuition_layer = "features" 
        model.num_classes = num_classes 
        model.embedding_generator = "nn.Sequential( model.conv1, \
                                                    model.in3a, \
                                                    model.in3b, \
                                                    model.in3c, \
                                                    model.in4a, \
                                                    model.in4b, \
                                                    model.in4c, \
                                                    model.in4d, \
                                                    model.in4e, \
                                                    model.in5a, \
                                                    model.in5b, \
                                                    model.in6a, \
                                                    model.in6b, \
                                                    model.in6c, \
                                                    model.classifier[:-1], \
                                                    nn.Flatten())"


    logging.info(model)
    return model 