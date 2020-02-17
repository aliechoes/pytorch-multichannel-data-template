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
import pandas as pd
import numpy as np
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
import glob 
import sklearn
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
from skimage.transform import   rescale, resize 
from imageio import imread
from torchvision import transforms

def finding_classes(base_path):
    """
    this function finds the folders in the root path and considers them
    as classes
    """
    classes = os.listdir(base_path)
    print(classes)
    return classes

def finding_channels(   classes, base_path, 
                        file_prefix = "" ,file_extension = ".jpg"):
    """
    this function finds the existing channels in the folder and returns
    a list of them
    """
    channels = ["Ch1", "Ch2", "Ch3", "Ch4", "Ch5", "Ch6", \
                               "Ch7", "Ch8", "Ch9", "Ch10", "Ch11","Ch12", \
                               "Ch13", "Ch14", "Ch15", "Ch16", "Ch17", "Ch18"]
    existing_channels = []
    for ch in channels:
        cl_path = os.path.join(base_path, classes[0], file_prefix +  "*_" + \
            ch + file_extension)
        cl_files = glob.glob(cl_path)
        if len(cl_files)> 1:
            existing_channels.append(ch)
    return existing_channels
        
        
def number_of_files_per_class(classes, base_path, existing_channels,
                        file_prefix = "" ,file_extension = ".jpg" ):
    """
    this function finds the number of files in each folder. it is important
    as with this, we can call all the files. In Amnis, each experiment
    always starts from 0.

    output: dictionary with keys as classes and values as number of files
    """
    results = dict()
    for cl in classes:
        cl_path = os.path.join(base_path, cl, "*" + file_extension) 
        cl_files = glob.glob(cl_path) 

        results[cl] = int(float(len(cl_files))/float(len(existing_channels)))
    return results
    
    
    
def input_dataframe_generator(classes_dict):
    """
    This functions gets the dictionary with the classes and number of files 
    per class and gives back a dataframe with these columns
    ["file"  ,"label", "class"]
    """
    label = 0
    df = pd.DataFrame(columns= ["file"  ,"label", "class", "prediction"] )
    for cl in classes_dict:
        df_dummy = pd.DataFrame(columns= ["file" ,"label", "class", "prediction"]  )
        df_dummy["file"] = range(1, classes_dict[cl])
        df_dummy["label"] = label
        df_dummy["class"] = cl
        df_dummy["prediction"] = -1
        df = df.append(df_dummy, ignore_index=True)
        label = label + 1
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


class Dataset_Generator(Dataset):
    """Dataset_Generator"""

    def __init__(self,  base_path, df , channels , set_type , 
                    reshape_size = 32, mean = None , std = None):
        """
        Args:
            csv_file (string): Path to the csv file with annotations.
            base_path (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.df = df.copy().reset_index(drop = True)
        
        # just using the set of interest
        if set_type is not None:
            self.df = self.df[self.df["set"]==set_type].reset_index(drop = True) 
        
        self.base_path = base_path
        self.reshape_size = reshape_size
        self.channels = channels
        self.mean = mean
        self.std = std
        
    def __len__(self):
        return self.df.shape[0]

    def __getitem__(self, idx):
        
        if torch.is_tensor(idx):
            idx = idx.tolist() 

        image = np.zeros((len(self.channels), self.reshape_size, self.reshape_size))
        #print(("idx", idx))
        for ch in range(0,len(self.channels) ): 
            img_name = os.path.join(self.base_path,self.df.loc[idx,"class"], \
                                   str(self.df.loc[idx,"file"])  +"_" + \
                                       self.channels[ch] + ".jpg")
            image_dummy = imread(img_name)
            image_dummy = resize(image_dummy , (self.reshape_size, self.reshape_size) )  
            image[ch,:,:] = image_dummy
        
        image = torch.from_numpy(image) 
        
        if self.mean is not None and self.std is not None:
            image = transforms.Normalize(self.mean,self.std)(image) 
            
        label = self.df.loc[idx,"label"]
        
        label = np.array([label]) 
        
        sample = {'image': image , 'label': torch.from_numpy(label) , "idx":idx }
        
        return sample


class DataLoaderGenerator():
    def __init__(self, base_path, batch_size, validation_split, test_split):
        self.base_path = base_path 
        self.batch_size = batch_size
        self.validation_split = validation_split
        self.test_split = test_split
        
    def data_frame_creator(self):
        
        self.classes = finding_classes(self.base_path)
        self.existing_channels = finding_channels(self.classes, self.base_path)

        self.nb_per_class = number_of_files_per_class(  self.classes, \
                                                        self.base_path, \
                                                        self.existing_channels)
        print("detected files per class")
        print(self.nb_per_class)

        self.df = input_dataframe_generator(self.nb_per_class)
        self.df = train_validation_test_split(  self.df, 
                                                self.validation_split , 
                                                self.test_split ,
                                                randomize = False)


    def calculate_statistics(self):
        """
        This functions creates the trainloader and calulates the mean
        and standard deviation for the training set
        """
        train_dataset = Dataset_Generator(    self.base_path, 
                                                self.df , 
                                                self.existing_channels , 
                                                "train" , 
                                                self.reshape_size )

        trainloader = DataLoader(   train_dataset, \
                                    batch_size=self.batch_size, \
                                    shuffle=False, num_workers=4)

        mean = 0.
        std = 0.
        nb_samples = 0.

        for data in trainloader:
            batch_samples = data["image"].size(0)
            data = data["image"].view(batch_samples, data["image"].size(1), -1)
            
            mean += data.mean(2).sum(0)
            std += data.std(2).sum(0)
            nb_samples += batch_samples
            
        train_dataset = None
        trainloader = None

        mean /= nb_samples
        std /= nb_samples
        self.mean = mean
        self.std = std

    def data_loader(self, reshape_size):

        self.reshape_size = reshape_size
        self.calculate_statistics()
                                        
        self.train_dataset = Dataset_Generator(self.base_path, 
                                            self.df , 
                                            self.existing_channels  ,  
                                            "train" , 
                                            self.reshape_size , 
                                            self.mean, 
                                            self.std )

        self.trainloader = DataLoader(self.train_dataset, 
                                batch_size=self.batch_size, \
                                shuffle=True, 
                                num_workers=4)

        self.validation_dataset = Dataset_Generator(self.base_path, 
                                            self.df , 
                                            self.existing_channels  ,  
                                            None , 
                                            self.reshape_size , 
                                            self.mean, 
                                            self.std )
                                            
        self.validationloader = DataLoader(self.validation_dataset, 
                                batch_size=1, \
                                shuffle=False, 
                                num_workers=1)