import torch
import torchvision
import torch.nn.functional as F
import torch.nn as nn 
from utils import *
import argparse
import time
from datetime import datetime
import os
import json
import itertools
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
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

def get_checkpoint(file_path):
    if file_path is not None:
        checkpoint = torch.load(file_path)
    else:
        checkpoint = None
    return checkpoint

def elapsed_time_print(start_time, message, epoch):
    """
    function to print the elapsed fime
    """
    elapsed_time = time.time() - start_time
    elapsed_time = time.strftime("%H:%M:%S", time.gmtime(elapsed_time))
    to_be_printed = "epoch %d: " + message 
    logging.info(10*"---")
    logging.info(to_be_printed % (epoch, elapsed_time))
    logging.info(10*"---")
    return None

def predict(configs):
    ml_configs = configs["machine_learning"]
    data_configs = configs["data"]
    device = ml_configs["device"]
    output_folder = configs["data"]["output_folder"]
    checkpoint = get_checkpoint(configs["machine_learning"]["checkpoint_path"])


    data_loader = DataLoaderGenerator(data_configs)
    data_loader.data_frame_creator()
    
    logging.info(data_loader.df)

    # number of exsting channels and output classes
    number_of_channels = len(data_loader.existing_channels)
    number_of_classes = len(data_loader.nb_per_class.keys())
    logging.info(number_of_channels, number_of_classes)
    # initialize the model
    model = get_model(  ml_configs,
                        checkpoint,
                        number_of_channels ,
                        9)

    data_loader.data_loader(model.image_size, checkpoint)
    
    for j in range(100):
        data_loader.df["prediction_" + str(j)] = -1
    # the evaluation phase
 
    logging.info("starting the evaluation")
    with torch.no_grad():  
        model.eval()
        percentage = 0
        for i, data in enumerate(data_loader.validationloader, 0): 
 
            # finding the file in the dataframe
            idx = data["idx"].cpu().numpy()   
            percentage = percentage + len(idx) / data_loader.df.shape[0]
            logging.info(round(percentage, 2))
            inputs, labels = data["image"], data["label"]
            inputs, labels = inputs.to(device) , labels.to(device)
                
            inputs = inputs.float() 
            labels = labels.reshape(labels.shape[0])
                
            outputs = model(inputs)
                
            outputs_probability = F.softmax(outputs).cpu().numpy()  
            _, predicted = torch.max(outputs.data, 1) 

            data_loader.df.loc[idx,"prediction"] = predicted.cpu().numpy() 

            logging.info("probabilities")
            for k, cl in enumerate(data_loader.classes,0):
                data_loader.df.loc[idx,cl + "_probability"] = outputs_probability[:,k]

            logging.info("starting the uncertainty")
            model.train()
            for j in range(100):
                outputs = model(inputs)   
                _, predicted = torch.max(outputs.data, 1)    
                data_loader.df.loc[idx, "prediction_" + str(j)] = predicted.cpu().numpy()
            if i % 10 == 0: 
                logging.info(
                'Eval: [{}/{} ({:.0f}%)]'.
                format(i * len(inputs), len(data_loader.validation_dataset),
                       100. * i / len(data_loader.validationloader) )) 
        
        mode = data_loader.df.loc[:, [("prediction_" + str(j)) for j in range(100) ] ].mode(axis = 1)[0]
        for i in range(0,data_loader.df.shape[0]):
            data_loader.df.loc[i,"uncertainty"] = 1. - ((data_loader.df.loc[i, [("prediction_" + str(j)) for j in range(100) ] ] == mode[i]).sum())/100.
        # 
        # save the label of all images and their predictions
        data_loader.df.to_csv(os.path.join(output_folder,
                                        "granular_results.csv"), index = False)
        for cl in data_loader.classes:
            indx = (data_loader.df["class"] == cl)
            files = data_loader.df.loc[indx, "file"]
            files = files.str.replace(configs["data"]["data_dir"], "")
            files = files.str.replace(cl, "")
            files = files.str.replace("/Exp14_Donor1_Minus_SEA_", "")
            files = files.str.replace("_Ch1.ome.tif", "")
            files.to_csv("/pstore/home/shetabs1/" + cl + ".pop", index=False)



if __name__ == "__main__":
    parser = argparse.ArgumentParser( \
                            description='Starting the deep learning code')
    parser.add_argument('-c',\
                        '--config', \
                        help='config yaml file address', \
                        default="configs/sample_config_evaluation.json", \
                        type=str)

    args = vars(parser.parse_args())
    
    configs = load_json(args['config'])
    for k in configs:
        logging.info("%s : %s \n" % (k,configs[k]))
    predict(configs)

