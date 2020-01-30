import torch
import torchvision
import argparse
import sys
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
import yaml
import numpy as np 
from data_loader.data_loaders import DataLoaderGenerator

def load_yaml(file_path):
    with open(file_path, 'r') as stream:    
        return yaml.load(stream)


def main(configs):
    """
    main file
    """
    base_path = configs["base_path"]
    reshape_size = configs["reshape_size"]
    batch_size = configs["batch_size"]
    validation_split = configs["validation_split"]
    test_split = configs["test_split"]
    device = configs["device"]
    tensorboard_path = configs["tensorboard_path"]
    

    data_loader_generator = DataLoaderGenerator(base_path, 
                                                reshape_size,
                                                batch_size, 
                                                validation_split, 
                                                test_split) 
    data_loader_generator.data_loader_generator()




if __name__ == "__main__":
    parser = argparse.ArgumentParser( \
                            description='Starting the deep learning code')
    parser.add_argument('-c',\
                        '--config', \
                        help='config yaml file address', \
                        required=True, \
                        
                        type=str)
    args = vars(parser.parse_args())
    configs = load_yaml(args['config'])
    main(configs)

