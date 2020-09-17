import torch
import torchvision
import torchvision.models
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
from models.lenet import LeNet
from models.deepflow import DeepFlow
from models.weight_initialization import weight_init
from torchvision.models.resnet import ResNet, BasicBlock
import logging

def model_info(model):
    logging.info(model)
    number_of_trainable_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)/1000000.
    number_of_nontrainable_parameters = sum(p.numel() for p in model.parameters() if not p.requires_grad)
    logging.info("Number of trainable parameters: %.2f M" % number_of_trainable_parameters)
    logging.info("Number of nontrainable parameters: %d" % number_of_nontrainable_parameters)
    return None

def load_checkpoint(model, device, checkpoint):
    logging.info("Data Parallelism is on! %d GPUs are detected" % \
                                        torch.cuda.device_count())
    model = nn.DataParallel(model)
    model = model.to(device)
    # transfer learning
    if checkpoint is not None:
        for m in checkpoint["model_state_dict"]:
            try:
                model.state_dict()[m].data.copy_(checkpoint["model_state_dict"][m])
            except:
                logging.info("for the layer %s it could not match state_dict" % m )
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
        model = torchvision.models.alexnet(pretrained=True)
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

    if model_name == "resnet9":
        model = ResNet(BasicBlock, [1, 1, 1, 1]) 
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
    if model_name == "resnet18":
        model = torchvision.models.resnet18(pretrained=True)
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

    if model_name == "wide_resnet50_2":
        model = torchvision.models.wide_resnet50_2(pretrained=True)
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

    if model_name == "resnet34":
        model = torchvision.models.resnet34(pretrained=True)
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

    if model_name == "resnext50_32x4d":
        model = torchvision.models.resnext50_32x4d(pretrained=True)
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
        model = torchvision.models.densenet121(pretrained=True, drop_rate = 0.5)
        ## loading the imagenet weights in case it is possible
        if num_channels != 3:
            model.features.conv0 = nn.Conv2d(num_channels, 64, kernel_size=(7, 7), 
                            stride=(2, 2), padding=(3, 3), bias=False)
        num_ftrs = model.classifier.in_features
        model.classifier = nn.Linear(num_ftrs, num_classes)
        model = load_checkpoint(model, device, checkpoint)
        model.image_size = 224
        model.intuition_layer = "features"
        model.embedding_generator = "nn.Sequential(model.features, \
                                                    nn.Flatten())"
    
    if model_name == "squeezenet1_0":
        model = torchvision.models.squeezenet1_0(pretrained=True)
        ## loading the imagenet weights in case it is possible
        if num_channels != 3:
            model.features[0] = nn.Conv2d(num_channels, 96, kernel_size=7, stride=2)
        model.classifier[1] = nn.Conv2d(512, num_classes, kernel_size=(1,1), stride=(1,1))
        model = load_checkpoint(model, device, checkpoint)
        model.num_classes = num_classes
        model.image_size = 180
        model.intuition_layer = "features"
        model.embedding_generator = "nn.Sequential(model.features, \
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

    if model_name == "vgg16":
        model = torchvision.models.vgg16(pretrained=True)
        ## loading the imagenet weights in case it is possible
        if num_channels != 3:
            model.features[0] = nn.Conv2d(num_channels, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        
        num_ftrs = model.classifier[6].in_features
        model.classifier[6] = nn.Linear(num_ftrs, num_classes)
        model = load_checkpoint(model, device, checkpoint)
        model.image_size = 224
        model.intuition_layer = "features"
        model.embedding_generator = "nn.Sequential( model.features, \
                                                    model.avgpool, \
                                                    nn.Flatten(), \
                                                    model.classifier[:-1], \
                                                    nn.Flatten())"
 
    
    model_info(model)
    return model 