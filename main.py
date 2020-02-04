import torch
import torchvision
from torch.utils.tensorboard import SummaryWriter
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
from inputs.data_loaders import DataLoaderGenerator
from machine_learning.models import get_model
from machine_learning.optimizers import get_optimizer
from machine_learning.losses import get_loss
from outputs.tensorboard_writer import TensorBoardSummaryWriter
from train import train
import warnings
warnings.filterwarnings("ignore")

# fix random seeds for reproducibility
SEED = 123
torch.manual_seed(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
np.random.seed(SEED)

def make_folders(tb_path, run_name,desired_folder):
    os.mkdir(os.path.join(tb_path,run_name ,desired_folder))
    return None

def load_json(file_path):
    with open(file_path, 'r') as stream:    
        return json.load(stream)

def save_json(file_path, data):
    with open(file_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=4)


def main(configs):
    """
    main file
    """
    base_path = configs["base_path"]
    batch_size = configs["batch_size"]
    validation_split = configs["validation_split"]
    test_split = configs["test_split"]
    tensorboard_path = configs["tensorboard_path"]
    #file_extension = configs["file_extension"]
    #previous_model_address = configs["previous_model_address"]
    model_name = configs["model_name"]
    num_epochs = configs["num_epochs"]
    device = configs["device"]
    optimization_parameters = configs["optimization_parameters"]
    optimization_method = configs["optimization_method"]
    loss_function = configs["loss_function"]
    metrics_of_interest = configs["metrics_of_interest"]

    run_name = str(datetime.now()) + "_" + \
                    model_name + "_bs_" + str(batch_size)
                         
    

    writer = TensorBoardSummaryWriter( os.path.join(tensorboard_path, run_name ) )
    make_folders(tensorboard_path, run_name, "models/")
    model_folder = os.path.join(tensorboard_path, run_name, "models/")
    
    data_loader_generator = DataLoaderGenerator(base_path,  
                                                batch_size, 
                                                validation_split, 
                                                test_split) 
    data_loader_generator.data_frame_creator()
    

    number_of_channels = len(data_loader_generator.existing_channels)
    number_of_classes = len(data_loader_generator.nb_per_class.keys())

    model, reshape_size = get_model(  model_name, 
                        device,
                        number_of_channels ,
                        number_of_classes)
    print(model)
    data_loader_generator.data_loader_generator(reshape_size)

    ## load the optimzer
    optimizer = get_optimizer(   optimization_method, 
                                model, 
                                optimization_parameters) 
    ## load the loss
    criterion = get_loss(loss_function) 

    metric_dataframe = pd.DataFrame(columns= ["epoch","set", "metric", "value"])
    
    
    model, metric_dataframe = train( model,   
                                    data_loader_generator, 
                                    optimizer,
                                    criterion,
                                    metric_dataframe ,    
                                    metrics_of_interest,
                                    num_epochs,
                                    writer, 
                                    model_folder,
                                    device )
                                    
    make_folders(tensorboard_path, run_name, "output_files/")

    metric_dataframe.to_csv(os.path.join(tensorboard_path, run_name , 
            "output_files","aggregated_results.csv"), index = False)
    data_loader_generator.df.to_csv(os.path.join(tensorboard_path, run_name , 
            "output_files","granular_results.csv"), index = False)

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

