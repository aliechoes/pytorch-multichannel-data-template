import torch
import torchvision
import argparse
import sys
from datetime import datetime
import os
import io
import importlib
import uuid
import logging
import random
import base64
import hashlib
import mimetypes
import json
import imageio 
import json
import numpy as np 
import pandas as pd
from data_loaders.data_loaders import DataLoaderGenerator
from machine_learning.models import get_model
from machine_learning.optimizers import get_optimizer
from machine_learning.losses import get_loss
from tensorboard_writer.tensorboard_writer import TensorBoardSummaryWriter
from train import train
import warnings
warnings.filterwarnings("ignore")

# fix random seeds for reproducibility
SEED = 123
torch.manual_seed(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
np.random.seed(SEED)

def make_folders(desired_path):
    os.mkdir(desired_path)
    return None

def load_json(file_path):
    with open(file_path, 'r') as stream:    
        return json.load(stream)

def save_json(file_path, data):
    with open(file_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=4)


def main(configs):
    """
    This is the main function which includes the main logic of the script.
    
    Args    
        configs: dictionary file with the format of the config file
    """
    data_dir = configs["data_dir"]
    mask_dir = configs["mask_dir"]
    batch_size = configs["batch_size"]
    validation_split = configs["validation_split"]
    test_split = configs["test_split"]
    tensorboard_path = configs["tensorboard_path"]
    file_extension = configs["file_extension"]
    # TODO: transfer learning
    #previous_model_address = configs["previous_model_address"]

    model_name = configs["model_name"]
    num_epochs = configs["num_epochs"]
    device = configs["device"]
    optimization_parameters = configs["optimization_parameters"]
    optimization_method = configs["optimization_method"]
    loss_function = configs["loss_function"]
    metrics_of_interest = configs["metrics_of_interest"]

    # creating a unique name for the model
    run_name = str(datetime.now()) + "_" + \
                    model_name + "_bs_" + str(batch_size)
                         
    
    # creating the tensorboard
    writer = TensorBoardSummaryWriter( os.path.join(tensorboard_path, run_name ) )

    # creating the folder for the models to be saved per epoch
    model_folder = os.path.join(tensorboard_path, run_name, "models/")
    make_folders(model_folder)

    
    # creating the dataloader
    data_loader = DataLoaderGenerator(  data_dir, 
                                        mask_dir, 
                                        file_extension,  
                                        batch_size, 
                                        validation_split, 
                                        test_split) 
    data_loader.data_frame_creator()
    
    # number of exsting channels and output classes
    number_of_channels = len(data_loader.existing_channels)
    number_of_classes = len(data_loader.nb_per_class.keys())

    # initialize the model
    model, reshape_size = get_model(  model_name, 
                        device,
                        number_of_channels ,
                        number_of_classes)
    
    data_loader.data_loader(reshape_size)

    ## load the optimzer
    optimizer = get_optimizer(   optimization_method, 
                                model, 
                                optimization_parameters) 
    
    ## load the loss
    criterion = get_loss(loss_function) 

    # creating a dataframe which will contain all the metrics per set per epoch
    metric_dataframe = pd.DataFrame(columns= ["epoch","set", "metric", "value"])
    
    
    # train the model and record the results in the metric_dataframe
    model, metric_dataframe = train( model,   
                                    data_loader, 
                                    optimizer,
                                    criterion,
                                    metric_dataframe ,    
                                    metrics_of_interest,
                                    num_epochs,
                                    writer, 
                                    model_folder,
                                    device )
                                    
    # save the dataset with train/validation/test per epoch
    output_folder = os.path.join(tensorboard_path, run_name, "output_files/")
    make_folders(output_folder)
    metric_dataframe.to_csv(os.path.join(output_folder,
                                    "aggregated_results.csv"), index = False)
    
    # save the label of all images and their predictions
    data_loader.df.to_csv(os.path.join(output_folder,
                                    "granular_results.csv"), index = False)






if __name__ == "__main__":
    parser = argparse.ArgumentParser( \
                            description='Starting the deep learning code')
    parser.add_argument('-c',\
                        '--config', \
                        help='config yaml file address', \
                        required=True, \
                        type=str)

    args = vars(parser.parse_args())
    
    configs = load_json(args['config'])
    for k in configs:
        print((k,configs[k]))
    main(configs)

