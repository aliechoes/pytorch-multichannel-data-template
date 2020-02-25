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
from sklearn.utils import shuffle
from skimage.transform import   rescale, resize, rotate
from imageio import imread 
from data_loaders.data_sets import Dataset_Generator


def finding_classes(data_dir):
    """
    this function finds the folders in the root path and considers them
    as classes
    """
    classes = os.listdir(data_dir)
    print(classes)
    return classes

def finding_channels(   classes, data_dir):
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
        
        
def number_of_files_per_class(classes, data_dir, existing_channels ):
    """
    this function finds the number of files in each folder. it is important
    as with this, we can call all the files. In Amnis, each experiment
    always starts from 0.

    output: dictionary with keys as classes and values as number of files
    """
    results = dict()
    for cl in classes:
        cl_path = os.path.join(data_dir, cl, "*") 
        cl_files = glob.glob(cl_path) 
        results[cl] = int(float(len(cl_files))/float(len(existing_channels)))
    return results
    
    
    
def input_dataframe_generator(data_dir, classes_dict, representing_channel):
    """
    This functions gets the dictionary with the classes and number of files 
    per class and gives back a dataframe with these columns
    ["file"  ,"label", "class"]
    """
    label = 0
    df = pd.DataFrame(columns= ["file"  ,"label", "class", "prediction"] )
    for cl in classes_dict:
        df_dummy = pd.DataFrame(columns= ["file" ,"label", "class", "prediction"]  )
        df_dummy["file"] = glob.glob(os.path.join(data_dir, cl, "*_" + representing_channel + "*") ) 
        df_dummy["label"] = label
        df_dummy["class"] = cl
        df_dummy["prediction"] = -1.
        df = df.append(df_dummy, ignore_index=True)
        label = label + 1
    for cl in classes_dict:
        df[cl+"_probability"] = -1.
    return df

def train_validation_test_split(df, validation_size= 0.2 , test_size = 0.3 ,
                randomize = False):
    """
    This functions gets the dataframe and creates train, validation and test 
    split. It gives back the dataframe with columns:
    ["file"  ,"label", "class", "set]
    """

    df["set"] = "train"
    X_train, X_test = train_test_split(df["label"], test_size=test_size, 
                                    random_state=314)

    X_train, X_validation = train_test_split(X_train, test_size=validation_size, 
                                    random_state=314)
    df.loc[X_validation.index,"set"] = "validation"
    df.loc[X_test.index,"set"] = "test"
    if randomize:
        df = shuffle(df)
    return df


class DataLoaderGenerator():
    def __init__(self, data_dir, batch_size, validation_split, test_split, data_map):
        self.data_dir = data_dir  
        self.batch_size = batch_size
        self.validation_split = validation_split
        self.test_split = test_split
        self.data_map = data_map
        
    def data_frame_creator(self):
        
        self.classes = finding_classes(self.data_dir)
        self.existing_channels = finding_channels(  self.classes, 
                                                    self.data_dir)

        self.nb_per_class = number_of_files_per_class(  self.classes, 
                                                        self.data_dir, 
                                                        self.existing_channels)
        print("detected files per class")
        print(self.nb_per_class)

        self.df = input_dataframe_generator(self.data_dir, 
                                            self.nb_per_class,
                                            self.existing_channels[0])

        self.df = train_validation_test_split(  self.df, 
                                                self.validation_split , 
                                                self.test_split ,
                                                randomize = False)


    def calculate_statistics(self):
        """
        This functions creates the trainloader and calulates the mean
        and standard deviation for the training set
        """
                
        train_dataset = Dataset_Generator(  self.data_dir,  
                                            self.df , 
                                            self.existing_channels , 
                                            "train" , 
                                            self.reshape_size )
        
        trainloader = DataLoader(   train_dataset, \
                                    batch_size=self.batch_size, \
                                    shuffle=True, num_workers=1) 

        numer_of_channels = len(self.existing_channels)    

        self.statistics = dict()                   
        self.statistics["min"] = torch.zeros(numer_of_channels) + np.power(2,16) # 16-bit images
        self.statistics["lower_bound"] = torch.zeros(numer_of_channels)
        self.statistics["mean"] = torch.zeros(numer_of_channels)
        self.statistics["std"] = torch.zeros(numer_of_channels)
        self.statistics["upper_bound"] = torch.zeros(numer_of_channels)
        self.statistics["max"] = torch.zeros(numer_of_channels)

        for k, data in enumerate(trainloader, 1):
            data = data["image"] 
            for i in range(numer_of_channels):
                self.statistics["min"][i] = min(data[:,i,:,:].min(), self.statistics["min"][i]   )
                self.statistics["lower_bound"][i] += np.quantile( data[:,i,:,:], .02) 
                self.statistics["mean"][i] += data[:,i,:,:].mean()
                self.statistics["std"][i] += data[:,i,:,:].std()
                self.statistics["upper_bound"][i] += np.quantile( data[:,i,:,:], .98) 
                self.statistics["max"][i] = max(data[:,i,:,:].max(), self.statistics["max"][i]    )

        self.statistics["lower_bound"] = self.statistics["lower_bound"] / float(k)
        self.statistics["mean"].div_(len(trainloader))
        self.statistics["std"].div_(len(trainloader))
        self.statistics["upper_bound"] = self.statistics["upper_bound"] / float(k)

        for k in self.statistics:
            print(k,self.statistics[k])

    def data_loader(self, reshape_size):

        self.reshape_size = reshape_size
        print("Starting to calculate the statistics...")
        self.calculate_statistics()
        print("Calculating the statistics is finished")

        self.train_dataset = Dataset_Generator(self.data_dir,  
                                            self.df , 
                                            self.existing_channels ,  
                                            "train" , 
                                            self.reshape_size , 
                                            self.data_map, 
                                            self.statistics,
                                            True )

        self.trainloader = DataLoader(self.train_dataset, 
                                batch_size=self.batch_size, \
                                shuffle=True, 
                                num_workers=1)
                                
        self.validation_dataset = Dataset_Generator(self.data_dir,  
                                            self.df , 
                                            self.existing_channels ,  
                                            None , 
                                            self.reshape_size , 
                                            self.data_map, 
                                            self.statistics,
                                            False )
                                            
        self.validationloader = DataLoader(self.validation_dataset, 
                                batch_size= self.batch_size, \
                                shuffle=False, 
                                num_workers=1)