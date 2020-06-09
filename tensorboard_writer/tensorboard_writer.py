import matplotlib.pyplot as plt
import numpy as np
import os
import torch
import torchvision
from torch.utils.tensorboard import SummaryWriter
from torchvision import datasets, transforms
import torch.nn.functional as F
from sklearn.metrics import confusion_matrix

import itertools
import numpy as np
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt


def plot_confusion_matrix(cm, class_names):
    """
    Returns a matplotlib figure containing the plotted confusion matrix.

    Args:
        cm (array, shape = [n, n]): a confusion matrix of integer classes
        class_names (array, shape = [n]): String names of the integer classes
    """
    figure = plt.figure(figsize=(8, 8))
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title("Confusion matrix")
    #plt.colorbar()
    tick_marks = np.arange(len(class_names))
    plt.xticks(tick_marks, class_names, rotation=45)
    plt.yticks(tick_marks, class_names)

    # Normalize the confusion matrix.
    #cm = np.around(cm.astype('float') / cm.sum(axis=1)[:, np.newaxis], decimals=2)

    # Use white text if squares are dark; otherwise black.
    threshold = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        color = "white" if cm[i, j] > threshold else "black"
        plt.text(j, i, cm[i, j], horizontalalignment="center", color=color)

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    return figure


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

class TensorBoardSummaryWriter(object):
    """
    Class for writing different outputs to TensorBoard
    """
    def __init__(self, tensorboard_configs , run_name):
        self.tensorboard_path =   tensorboard_configs["tensorboard_path"]
        self.writer = SummaryWriter(os.path.join(self.tensorboard_path, run_name ) )
        self.write_add_metrics = tensorboard_configs["add_metrics"] 
        self.write_add_images = tensorboard_configs["add_images"]
        self.write_add_confusion_matrix = tensorboard_configs["add_confusion_matrix"]
        self.write_add_graph = tensorboard_configs["add_graph"]
        self.write_add_pr_curve = tensorboard_configs["add_pr_curve"]
        self.write_add_hparams = tensorboard_configs["add_hparams"]
        self.write_add_embedding = tensorboard_configs["add_embedding"] 

    def add_metrics(self,df, metrics_of_interest,epoch):
        """
        Add metrics to tensorboard the scalars part.
        Args:
            df(pandas dataframe): dataframe with all the recorded metrics per epoch
            metrics_of_interest(list): the metrics which should be written in tensorboard
            epoch(int): epoch
        """
        if self.write_add_metrics:
            for mt in metrics_of_interest:
                results = dict()
                for s in ["train","validation","test"]:
                    
                    # finding the value for the metric and epoch
                    indx = ((df["set"] == s) & (df["metric"] == mt)) & \
                        (df["epoch"] == (epoch + 1))
                    
                    if indx.sum() > 0:
                        results[s] = df.loc[indx,"value"].iloc[0]
                self.writer.add_scalars( mt,results ,epoch +1)

            self.writer.close()
        
    def add_images(self,  data_loader, epoch ):
        """
        Add images to tensorboard the scalars part. It outputs each channel and uses 
        one image per class for each channel
        Args:
            data_loader: data loader from pytorch 
            epoch(int): epoch
        """
        if self.write_add_images:
            # One sample per class
            idx = data_loader.train_dataset.df.groupby('class')['class'].apply(lambda s: s.sample(1))
            idx = idx.index
            nb_classes = len(data_loader.classes)
            nb_channels = len(data_loader.existing_channels)
            
            for i in range(nb_classes):
                temp_images = torch.zeros( nb_channels,
                                1,                                 
                                data_loader.reshape_size, 
                                data_loader.reshape_size  ).cpu() 
                
                temp_data = data_loader.train_dataset.__getitem__(idx[i][1]) 
                for j in range(nb_channels): 
                    temp_images[j,0,:,:] = temp_data["image"].cpu()[j,:,:]  
                
                self.writer.add_images( idx[i][0] , temp_images, epoch )
                self.writer.close()

    
    def add_confusion_matrix(self,  data_loader, epoch ):
        """
        Add confusion matrix images to tensorboard 
        Args:
            data_loader: data loader from pytorch 
            epoch(int): epoch
        """
        if self.write_add_confusion_matrix:
            # One sample per class
            validation_index = (data_loader.df["set"] == "validation")
            df_validation =  data_loader.df[validation_index].copy()
            cm = confusion_matrix(df_validation["label"].astype(int), df_validation["prediction"].astype(int))

            self.writer.add_figure("Confusion Matrix",plot_confusion_matrix(cm, data_loader.classes), global_step=epoch ) 
            

    def add_graph(self, model, data_loader ):
        """
        Add the model to tensorboard. It shows the model architecture
        Args:
            data_loader: data loader from pytorch 
            model: pytorch model
        """
        if self.write_add_graph:
            images, _ = select_n_random(data_loader.train_dataset )
            images = images.float() 

            self.writer.add_graph(model.cpu(), images.cpu())
            self.writer.close()
    
    def add_embedding(self, feature_extractor, data_loader, epoch, device):
        """
        gets the model and outpus n random images features to tensorboard projector
        Args:
            data_loader: data loader from pytorch 
            model: pytorch model
            epoch(int)
            device(str): either cpu or cuda
        """
        if self.write_add_embedding:
            images, labels = select_n_random(data_loader.train_dataset )
            labels = labels.cpu()
            images = images.to(device)
            images = images.float()
            feature_extractor = feature_extractor.to(device)
            
            # get the class labels for each image
            class_labels = [data_loader.classes[lab] for lab in labels]
            
            features = feature_extractor(images)
            features = features.reshape(features.shape[0], features.shape[1] )
            
            images_shape = (images.shape[0], 1,  images.shape[2]  , images.shape[3] )
            for j in range(images.shape[1]):
                self.writer.add_embedding(tag = "Channel " + str(j+1),
                            mat = features ,
                            metadata=class_labels,
                            label_img=images[:,j,:,:].reshape(images_shape),
                            global_step = epoch + 1)
                self.writer.close()
    
    def add_pr_curve(self, data_loader, epoch):
        """
        outputs the precision recall curve per class per epoch
        Args:
            data_loader: data loader from pytorch 
            model: pytorch model
        """
        if self.write_add_pr_curve:
            validation_index = (data_loader.df["set"] == "validation")
            df_validation =  data_loader.df[validation_index].copy()

            for k, cl in enumerate(data_loader.classes,0):
                probabilities = (df_validation.loc[:, cl + "_probability"]  ).to_numpy()
                labels = (df_validation["label"] == k).to_numpy()
                self.writer.add_pr_curve(cl,
                            labels,
                            probabilities,
                            global_step=epoch)

    def add_hparams(self, configs, best_value, best_epoch):
        """
        outputs the hyperparameters to tensorboard.
        Args:
            configs: dict
        """
        if self.write_add_hparams:
            # seperating the configs part
            model_configs = configs["model"]
            loss_configs = configs["loss"]
            optimizer_configs = configs["optimizer"]
            training_configs = configs["training"]
            data_loader_configs = configs["data_loader"] 

            hparam_dict = dict()
            hparam_dict["data_dir"] = data_loader_configs["data_dir"]

            #hparam_dict["test_data_dir"] = data_loader_configs["test_data_dir"]
            #hparam_dict["batch_size"] = data_loader_configs["batch_size"]
            #hparam_dict["model_name"] = model_configs["network"]
            #hparam_dict["optimization_method"] = optimizer_configs["optimization_method"]
            #hparam_dict["loss_function"] = loss_configs["loss_function"]
            hparam_dict["criteria"] = training_configs["call_back"]["criteria"]
            #hparam_dict["best_epoch"] = best_epoch

            #for k in optimizer_configs["optimization_parameters"]:
                #hparam_dict[k] = optimizer_configs["optimization_parameters"][k]

            metric_dict = dict()  
            metric_dict["hparam/" + hparam_dict["criteria"]] = round(best_value, 4)
            
            #for k in hparam_dict:
            #    if hparam_dict[k] == None:
            #        hparam_dict[k] = "None"
            
            #for k in metric_dict:
            #    if metric_dict[k] == None:
            #        metric_dict[k] = "None"
            
            #for k in hparam_dict:
            #    hparam_dict[k] = str(hparam_dict[k]).replace(" ", "_").lower()

            print(hparam_dict)
            print(metric_dict)
            
            self.writer.add_hparams(hparam_dict, metric_dict) 
            self.writer.close()
