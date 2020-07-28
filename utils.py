import torch
import torchvision
import argparse
from datetime import datetime
import os
import multiprocessing
import psutil
import json
import itertools
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from data_loaders.data_loaders import DataLoaderGenerator
from models import get_model
from optimizers import get_optimizer
from losses import get_loss
from tensorboard_writer.tensorboard_writer import TensorBoardSummaryWriter
from train import train
import logging

# fix random seeds for reproducibility
#SEED = 123
#torch.manual_seed(SEED)
torch.backends.cudnn.deterministic = False
torch.backends.cudnn.benchmark = True
#np.random.seed(SEED)

def make_folders(desired_path):
    os.mkdir(desired_path)
    return None

def create_name(arch, optimization_method, lr):
    run_name = str(datetime.now()) +    "_" + arch + \
                                        "_" + optimization_method + \
                                        "_" + str(lr)
    logging.info("Model Name: %s \n" % run_name)
    return run_name

def load_json(file_path):
    with open(file_path, 'r') as stream:    
        return json.load(stream)

def save_json(file_path, data):
    with open(file_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=4)

def get_checkpoint(file_path):
    if file_path is not None:
        checkpoint = torch.load(file_path)
    else:
        checkpoint = None
    return checkpoint

def logger(level):
    logging_level = {
        0 : logging.WARNING,
        1 : logging.INFO,
        2 : logging.DEBUG
    }
    logging.basicConfig(level=logging_level[level])
    logging.getLogger('matplotlib.font_manager').disabled = True

def system_info():
    logging.info("CPU count: %d" % multiprocessing.cpu_count())
    logging.info("available RAM: %s GB" % round(psutil.virtual_memory().available/1000000000., 2) )
    logging.info("available number of GPUs: %d" % torch.cuda.device_count() ) 