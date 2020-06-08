
import torch
import torchvision
from torchvision.models import alexnet, resnet18, densenet121, squeezenet1_0
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init

def to_device(model, device):
    if torch.cuda.device_count() > 1:
        print("Data Parallelism is on! %d GPUs are detected" % \
                                            torch.cuda.device_count())
    model = nn.DataParallel(model)
    model = model.to(device)
    return model

def pretrained_weights(model, weights):

    # in our code, it is always the last layer will be classifier 
    # and before that it is embedding
    for w in list(weights.keys())[0:-1]:
        layer_name = w
        if 'classifier' in w:
            layer_name = layer_name.replace('classifier', 'embedding')
        model.state_dict()[layer_name] = weights[w]
    return model

class LeNet(nn.Module):
    def __init__(self,num_channels ,num_classes, droprate=0.5):
        super(LeNet, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(num_channels,  20, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            nn.Dropout2d(p=droprate),
            nn.MaxPool2d(2, stride=2),
            nn.Conv2d(20, 50, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            nn.Dropout2d(p=droprate),
            nn.MaxPool2d(2, stride=2), 
            nn.Flatten() 
        )

        self.classifier= nn.Sequential(
            nn.Linear(12800, 500),
            nn.ReLU(inplace=True),
            nn.Dropout(p=droprate),
            nn.Linear(500, num_classes)
        ) 
        
    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x

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




def weight_init(m):
    '''
    Usage:
        model = Model()
        model.apply(weight_init)
    '''
    if isinstance(m, nn.Conv1d):
        init.normal_(m.weight.data)
        if m.bias is not None:
            init.normal_(m.bias.data)
    elif isinstance(m, nn.Conv2d):
        init.xavier_normal_(m.weight.data)
        if m.bias is not None:
            init.normal_(m.bias.data)
    elif isinstance(m, nn.Conv3d):
        init.xavier_normal_(m.weight.data)
        if m.bias is not None:
            init.normal_(m.bias.data)
    elif isinstance(m, nn.ConvTranspose1d):
        init.normal_(m.weight.data)
        if m.bias is not None:
            init.normal_(m.bias.data)
    elif isinstance(m, nn.ConvTranspose2d):
        init.xavier_normal_(m.weight.data)
        if m.bias is not None:
            init.normal_(m.bias.data)
    elif isinstance(m, nn.ConvTranspose3d):
        init.xavier_normal_(m.weight.data)
        if m.bias is not None:
            init.normal_(m.bias.data)
    elif isinstance(m, nn.BatchNorm1d):
        init.normal_(m.weight.data, mean=1, std=0.02)
        init.constant_(m.bias.data, 0)
    elif isinstance(m, nn.BatchNorm2d):
        init.normal_(m.weight.data, mean=1, std=0.02)
        init.constant_(m.bias.data, 0)
    elif isinstance(m, nn.BatchNorm3d):
        init.normal_(m.weight.data, mean=1, std=0.02)
        init.constant_(m.bias.data, 0)
    elif isinstance(m, nn.Linear):
        init.xavier_normal_(m.weight.data)
        init.normal_(m.bias.data)
    elif isinstance(m, nn.LSTM):
        for param in m.parameters():
            if len(param.shape) >= 2:
                init.orthogonal_(param.data)
            else:
                init.normal_(param.data)
    elif isinstance(m, nn.LSTMCell):
        for param in m.parameters():
            if len(param.shape) >= 2:
                init.orthogonal_(param.data)
            else:
                init.normal_(param.data)
    elif isinstance(m, nn.GRU):
        for param in m.parameters():
            if len(param.shape) >= 2:
                init.orthogonal_(param.data)
            else:
                init.normal_(param.data)
    elif isinstance(m, nn.GRUCell):
        for param in m.parameters():
            if len(param.shape) >= 2:
                init.orthogonal_(param.data)
            else:
                init.normal_(param.data)



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
        ## loading the imagenet weights in case it is possible
        if num_channels != 3:
            model.features[0] = nn.Conv2d(num_channels, 64, 
                        kernel_size=(11, 11), stride=(4, 4), padding=(2, 2))
        num_ftrs = model.classifier[6].in_features
        model.classifier[6] = nn.Linear(num_ftrs, num_classes)
        model = to_device(model, device)
        model.image_size = 256
        model.intuition_layer = "features"
        model.embedding_generator = "nn.Sequential( model.features, \
                                                    model.avgpool, \
                                                    nn.Flatten(), \
                                                    model.classifier[:-1])"

    ## TODO: add grad-cam
    if model_name == "resnet18":
        model = resnet18(pretrained=True)
        ## loading the imagenet weights in case it is possible
        if num_channels != 3:
            model.conv1 = nn.Conv2d(num_channels, 64, kernel_size=(7, 7), 
                            stride=(2, 2), padding=(3, 3), bias=False)
        num_ftrs = model.fc.in_features
        model.fc = nn.Linear(num_ftrs, num_classes)
        model = to_device(model, device)
        model.image_size = 224
        model.intuition_layer = "features"
        model.embedding_generator = "nn.Sequential(*list(model.children())[:-1])"

        
    if model_name == "shallownet":
        model = ShallowNet(num_channels ,num_classes) 
        model = model.apply(weight_init)
        model = to_device(model, device)
        model.image_size = 28
        model.intution_layer = "features"
        model.embedding_generator = "nn.Sequential( model.features, \
                                                    model.classifier[:-1])"

    if model_name == "lenet":
        model = LeNet(num_channels ,num_classes) 
        model = model.apply(weight_init)
        model = to_device(model, device)
        model.image_size = 64
        model.intution_layer = "features"
        model.embedding_generator = "nn.Sequential( model.features, \
                                                    model.classifier[:-1])"
    ## TODO: add grad-cam
    if model_name == "densenet121":
        model = densenet121(pretrained=True, drop_rate = 0.5)
        ## loading the imagenet weights in case it is possible
        if num_channels != 3:
            model.features.conv0 = nn.Conv2d(num_channels, 64, kernel_size=(7, 7), 
                            stride=(2, 2), padding=(3, 3), bias=False)
        num_ftrs = model.classifier.in_features
        model.classifier = nn.Linear(num_ftrs, num_classes)
        model = to_device(model, device)
        model.image_size = 224
        model.intuition_layer = "features"
        model.embedding_generator = "nn.Sequential(model.features)"
    
    if model_name == "squeezenet1_0":
        model = squeezenet1_0(pretrained=True)
        ## loading the imagenet weights in case it is possible
        if num_channels != 3:
            model.features.conv0 = nn.Conv2d(num_channels, 96, kernel_size=7, stride=2)
        model.classifier[1] = nn.Conv2d(512, num_classes, kernel_size=(1,1), stride=(1,1))
        model = to_device(model, device)
        model.num_classes = num_classes
        model.image_size = 224
        model.intuition_layer = "features"
        model.embedding_generator = "nn.Sequential(model.features)"

    if model_name == "deepflow":
        model = DeepFlow(num_channels ,num_classes) 
        model = to_device(model, device)
        model = model.apply(weight_init)
        model.image_size = 64
        model.intuition_layer = "features" 
        model.num_classes = num_classes 
        model.embedding_generator = "nn.Sequential(  self.conv1(x), \
                                                    model.in3a(x), \
                                                    model.in3b(x), \
                                                    model.in3c(x), \
                                                    model.in4a(x), \
                                                    model.in4b(x), \
                                                    model.in4c(x), \
                                                    model.in4d(x), \
                                                    model.in4e(x), \
                                                    model.in5a(x), \
                                                    model.in5b(x), \
                                                    model.in6a(x), \
                                                    model.in6b(x), \
                                                    model.in6c(x), \
                                                    model.classifier(x)[:-1])"

    # transfer learning
    if checkpoint is not None:
        model.load_state_dict(checkpoint['model_state_dict'])
        print("preivously trained model is used")


    print(model)
    return model 