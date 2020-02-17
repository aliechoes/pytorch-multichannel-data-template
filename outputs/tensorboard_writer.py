import matplotlib.pyplot as plt
import numpy as np

import torch
import torchvision
from torch.utils.tensorboard import SummaryWriter
from torchvision import datasets, transforms
import torch.nn.functional as F

# helper function
def select_n_random(train_dataset , n=200):
    '''
    Selects n random datapoints and their corresponding labels from a dataset
    '''

    features_data_loader = torch.utils.data.DataLoader(train_dataset, batch_size=n,
                                        shuffle=True, num_workers=1)

    features_data = next(iter(features_data_loader)) 
    features_labels = features_data["label"]
    features_data = features_data["image"]
     
    return features_data, features_labels

# helper functions

def images_to_probs(net, images, classes):
    '''
    Generates predictions and corresponding probabilities from a trained
    network and a list of images
    '''
    output = net(images)
    # convert output probabilities to predicted class
    _, preds_tensor = torch.max(output, 1)
    preds = np.squeeze(preds_tensor.numpy())
    return preds, [F.softmax(el, dim=0)[i].item() for i, el in zip(preds, output)]


def plot_classes_preds(net, images, labels):
    '''
    Generates matplotlib Figure using a trained network, along with images
    and labels from a batch, that shows the network's top prediction along
    with its probability, alongside the actual label, coloring this
    information based on whether the prediction was correct or not.
    Uses the "images_to_probs" function.
    '''
    preds, probs = images_to_probs(net, images)
    # plot the images in the batch, along with predicted and true labels
    fig = plt.figure(figsize=(12, 48))
    for idx in np.arange(4):
        ax = fig.add_subplot(1, 4, idx+1, xticks=[], yticks=[])
        matplotlib_imshow(images[idx], one_channel=True)
        ax.set_title("{0}, {1:.1f}%\n(label: {2})".format(
            classes[preds[idx]],
            probs[idx] * 100.0,
            classes[labels[idx]]),
                    color=("green" if preds[idx]==labels[idx].item() else "red"))
    return fig

class TensorBoardSummaryWriter(object):
    def __init__(self, output_path ):
        self.writer = SummaryWriter(output_path )
    
    def add_loss(self,loss_type, loss_value, x_value):
         
        self.writer.add_scalar(loss_type ,
                     loss_value,
                     x_value)
        self.writer.close()

    def add_metrics(self,df, metrics_of_interest,epoch):
        for mt in metrics_of_interest:
            results = dict()
            for s in ["train","validation","test"]:
                
                indx = ((df["set"] == s) & (df["metric"] == mt)) & \
                    (df["epoch"] == (epoch + 1))
                
                results[s] = df.loc[indx,"value"].iloc[0]
            self.writer.add_scalars( mt,results ,epoch +1)

        self.writer.close()
        
    def add_image(self, image_name, images ):
        # create grid of images
        img_grid = torchvision.utils.make_grid(images)


        # get the class labels for each image
        class_labels = [data_loader.classes[lab] for lab in labels]
 
        # write to tensorboard
        self.writer.add_image(image_name, img_grid)
        self.writer.close()

    def add_graph(self, model, data_loader ):
        images, _ = select_n_random(data_loader.train_dataset )
        images = images.float() 

        self.writer.add_graph(model.cpu(), images.cpu())
        self.writer.close()
    
    def add_embedding(self, model, data_loader, epoch):
        images, labels = select_n_random(data_loader.train_dataset )
        images = images.float()
        # get the class labels for each image
        class_labels = [data_loader.classes[lab] for lab in labels]
 
        features = model.embedding_generator(images)
        images_shape = (images.shape[0], 1,  images.shape[2]  , images.shape[3] )
        for j in range(images.shape[1]):
            self.writer.add_embedding(tag = "Channel " + str(j+1),
                        mat = features ,
                        metadata=class_labels,
                        label_img=images[:,j,:,:].reshape(images_shape),
                        global_step = epoch + 1)
            self.writer.close()
    
    def pr_curve(self, model, data_loader, epoch): 
        pass