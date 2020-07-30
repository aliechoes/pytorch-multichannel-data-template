from torchvision import datasets, transforms
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
import os
import glob
import pandas as pd 
import numpy as np
import os
import torch
import random
import pandas as pd
import numpy as np
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
import glob 
import sklearn
from sklearn.model_selection import train_test_split
from skimage.transform import   rescale, resize, rotate
from imageio import imread 
from data_loaders.data_sets import Dataset_Generator
import logging


def finding_classes(data_dir):
    """
    this function finds the folders in the root path and considers them
    as classes
    """
    classes = sorted(os.listdir(data_dir))
    logging.info("Classes: %s \n" % classes)
    return classes

def finding_channels(classes, data_dir):
    """
    this function finds the existing channels in the folder and returns
    a list of them
    """
    channels = ["Ch1", "Ch2", "Ch3", "Ch4", "Ch5", "Ch6", \
                               "Ch7", "Ch8", "Ch9", "Ch10", "Ch11","Ch12", \
                               "Ch13", "Ch14", "Ch15", "Ch16", "Ch17", "Ch18"]
    existing_channels = []
    for ch in channels:
        cl_path = os.path.join(data_dir, classes[0], "*_" +  ch + "*")
        cl_files = glob.glob(cl_path)
        if len(cl_files)> 1:
            existing_channels.append(ch)
    return existing_channels
        
        
def number_of_files_per_class(df ):
    """
    this function finds the number of files in each folder. It is important to
    consider that we consider all the channels togethr as on single image
    output: dictionary with keys as classes and values as number of separate images
    """

    logging.info("detected independent images per classes") 
    logging.info(df.groupby(["class", "set"])["class"].agg("count")) 
    
    return None
    
    
    
def input_dataframe_generator(data_dir, test_data_dir ,classes, representing_channel):
    """
    This functions gets the dictionary with the classes and number of files 
    per class and gives back a dataframe with these columns
    ["file"  ,"label", "class", "prediction",  
            "class0_probability" ... "classN_probability"]
    """
    
    df = pd.DataFrame(columns= ["file"  ,"label", "class", "set","uncertainty" ,"prediction"] )
    if test_data_dir != []:
        data_directory = {"train" : data_dir, "test" : test_data_dir}
    else:
        data_directory = {"train" : data_dir}
    
    for dd in data_directory:
        train_data_path = data_directory[dd]
        for tdp in train_data_path:
            label = 0
            for cl in classes:
                df_dummy = pd.DataFrame(columns= ["file" ,"label", "class", "set","prediction"]  )
                df_dummy["file"] = glob.glob(os.path.join(tdp , cl, "*_" + representing_channel + "*") ) 
                df_dummy["label"] = label
                df_dummy["class"] = cl
                df_dummy["uncertainty"] = -1.
                df_dummy["prediction"] = -1
                df_dummy["set"] = dd
                df = df.append(df_dummy, ignore_index=True)
                label = label + 1
    for cl in classes:
            df[cl+"_probability"] = -1.
    df_dummy["prediction"] = df_dummy["prediction"].astype(int)
    return df

def train_validation_test_split(df, validation_size= 0.2 , randomize = False):
    """
    This functions gets the dataframe and creates train, validation and test 
    split. it adds a new column: "set"
    """
    assert validation_size <= 1.
    assert validation_size >= 0.

    if validation_size == 1.0:
        df.loc[ df["set"]=="train" ,"set"] = "validation"
    elif validation_size > 0.:
        _, X_validation = train_test_split( df.loc[df["set"]=="train" , "set"], 
                                                stratify = df.loc[df["set"]=="train" , "set"], 
                                                test_size=validation_size, 
                                                random_state=314)
        df.loc[X_validation.index,"set"] = "validation"
    return df

def map_mean_std(mean, std, a , b):
    s = 1./(b - a)
    t = a/(a-b)
    mean_mapped = s*mean + t
    std_mapped = s*std 
    return mean_mapped, std_mapped

class DataLoaderGenerator():
    """
    data loader python generator.
    Args:
        data_configs(dict):
    """
    def __init__(self,data_configs):
        self.data_dir = data_configs["data_dir"]  
        self.test_data_dir = data_configs["test_data_dir"]  
        self.batch_size = data_configs["batch_size"]
        self.validation_split = data_configs["validation_split"]
        self.data_map = data_configs["data_map"]
        self.augmentation = data_configs["augmentation"]
        self.scaling_factor = data_configs["scaling_factor"]
        self.num_workers = data_configs["num_workers"]
        self.dynamic_range = data_configs["dynamic_range"]
        self.classes = finding_classes(self.data_dir[0] )
        self.existing_channels = finding_channels(  self.classes, 
                                                    self.data_dir[0])
        logging.info("Existing Channels: {}".format(self.existing_channels))
    
    def data_frame_creator(self):
        """
        datafame including every file, which later will be used for reading the images.
        the structure is like this:
        file	        label	class	    prediction  set
        file1_Chx.ext	0	    class_0 	0	        test
        file2_Chx.ext	1	    class_1 	0	        train
        file3_Chx.ext	0	    class_0 	0	        validation
        .               .       .           .           .
        .               .       .           .           .
        .               .       .           .           .
        """
        self.df = input_dataframe_generator(self.data_dir, 
                                            self.test_data_dir ,
                                            self.classes,
                                            self.existing_channels[0])

        self.df = train_validation_test_split(  self.df, 
                                                self.validation_split)

        number_of_files_per_class(  self.df )
        

    def calculate_statistics(self, checkpoint):
        """
        This functions creates the trainloader and calulates the statistics of
        the training set. It includes min, lower_bound, mean, std, upper_bound and 
        max values per channel
        """
        self.statistics = dict()
        if checkpoint is not None:
            logging.info("used previously calculated statistics for transfer learning")
            self.statistics["min"] = checkpoint["statistics"]["min"]
            self.statistics["lower_bound"] = checkpoint["statistics"]["lower_bound"] 
            self.statistics["mean"] = checkpoint["statistics"]["mean"]
            self.statistics["std"] = checkpoint["statistics"]["std"]
            self.statistics["upper_bound"] = checkpoint["statistics"]["upper_bound"]
            self.statistics["max"] = checkpoint["statistics"]["max"]
        else:
            train_dataset = Dataset_Generator(  self.df , 
                                                self.existing_channels , 
                                                ["train"] , 
                                                self.scaling_factor, 
                                                self.reshape_size )
            
            trainloader = DataLoader(   train_dataset, \
                                        batch_size=self.batch_size, \
                                        shuffle=True, num_workers=self.num_workers) 

            numer_of_channels = len(self.existing_channels)    

                            
            self.statistics["min"] = torch.zeros(numer_of_channels) + np.power(2,16) # 16-bit images
            self.statistics["lower_bound"] = torch.zeros(numer_of_channels)+ np.power(2,16) # 16-bit images
            self.statistics["mean"] = torch.zeros(numer_of_channels)
            self.statistics["std"] = torch.zeros(numer_of_channels)
            self.statistics["upper_bound"] = torch.zeros(numer_of_channels)
            self.statistics["max"] = torch.zeros(numer_of_channels)


            for j, data in enumerate(trainloader, 1): 
                
                if j % 100 == 1:
                    logging.info(10*"---")
                    logging.info("precentage of data %d is processed" % \
                                int( 100.*j / float(train_dataset.df.shape[0])  ) )

                data = data["image"] 
                for i in range(numer_of_channels):
                    self.statistics["min"][i] = min(data[:,i,:,:].min(), self.statistics["min"][i]   )
                    self.statistics["lower_bound"][i] = min( self.statistics["lower_bound"][i] , 
                                                    np.quantile( data[:,i,:,:], self.dynamic_range["lower_bound"] ) )

                    self.statistics["mean"][i] += data[:,i,:,:].mean()
                    self.statistics["std"][i] += data[:,i,:,:].std()

                    self.statistics["upper_bound"][i] = max(  self.statistics["upper_bound"][i], 
                                                    np.quantile( data[:,i,:,:], self.dynamic_range["upper_bound"]) ) 
                    self.statistics["max"][i] = max(data[:,i,:,:].max(), self.statistics["max"][i]    )

            self.statistics["mean"].div_(len(trainloader))
            self.statistics["std"].div_(len(trainloader))
        
        logging.info("Shifting the mean and std using %s \n" % self.dynamic_range)
        self.shift_mean_std()

    def shift_mean_std(self):
        self.statistics["mapped_mean"] = self.statistics["mean"].clone()
        self.statistics["mapped_std"] = self.statistics["std"].clone()
        for ch in range(0,len(self.existing_channels)  ): 
            a = self.statistics["lower_bound"][ch]
            b = self.statistics["upper_bound"][ch]
            mean = self.statistics["mapped_mean"][ch]
            std = self.statistics["mapped_std"][ch]
            self.statistics["mapped_mean"][ch] , self.statistics["mapped_std"][ch] = map_mean_std(mean, std, a , b)
        
        for k in self.statistics:
            logging.info("%s %s" % (k , self.statistics[k]))

    def data_loader(self, reshape_size, checkpoint):
        """
        This functions first calculates the statistics of the training dataset.
        then creates two dataset and corresponding data loader
        Args:
            reshape_size(int): size of the input image for the network
        datasets:
            train_dataset: only dataset including the training data
            validation_dataset: dataset including the whole data. But splitted in
                                train, validation and test
        """
        self.reshape_size = reshape_size
        logging.info("\nStarting to calculate the statistics...")
        self.calculate_statistics(checkpoint)
        logging.info("Calculating the statistics is finished \n")
        


        self.train_dataset = Dataset_Generator( 
                                            self.df , 
                                            self.existing_channels ,  
                                            ["train"] , 
                                            self.scaling_factor,
                                            self.reshape_size , 
                                            self.data_map, 
                                            self.statistics,
                                            self.augmentation )

        self.trainloader = DataLoader(self.train_dataset, 
                                batch_size=self.batch_size, \
                                shuffle=True, 
                                num_workers=self.num_workers)
                                
        self.validation_dataset = Dataset_Generator( 
                                            self.df , 
                                            self.existing_channels ,  
                                            ["validation", "test"] , 
                                            self.scaling_factor,
                                            self.reshape_size , 
                                            self.data_map, 
                                            self.statistics,
                                            [] )
                                            
        self.validationloader = DataLoader(self.validation_dataset, 
                                batch_size= self.batch_size, \
                                shuffle=False, 
                                num_workers=self.num_workers)