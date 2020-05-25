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
import numpy as np
from numpy import fliplr, flipud
from skimage import data
from skimage.transform import   rescale, resize, rotate
import warnings
import random
warnings.filterwarnings("ignore")


def random_crop(image, ratio = 0.8):
    """
    randomly choose the 0.8 of the height and width of an image
    """
    reshape_size = image.shape[0]
    width = int(reshape_size * ratio)
    height = int(reshape_size * ratio)
    x = random.randint(0, reshape_size - width)
    y = random.randint(0, reshape_size - height)
    image = image[y:y+height, x:x+width, :] 
    return image

def random_rotation(image ):
    """
    pick a random degree of rotation between 25 degree on 
    the left and 25 degree on the right
    """
    random_degree = random.uniform(-25, 25)
    return rotate(image, random_degree)

def random_flip(image ):
    """
    randomly choose to flip an image
    """
    random_number = np.random.random()
    if  random_number < 0.25: 
        return image  # no rotation
    elif random_number >= 0.25 and random_number < 0.50:
        return flipud(image) # vertical flip
    elif random_number >= 0.50 and random_number < 0.75:
        return fliplr(image) # horizontal flip
    else: 
        return flipud(fliplr(image)) # vertical & horizontal flip

def data_augmentation(image, aug):
    """
    combine different augmentation methods
    """
    if (aug == "random_crop") and (np.random.random() > 0.5):
        image = random_crop(image) 
    if (aug == "random_rotation") and (np.random.random() > 0.5): 
        image = random_rotation(image) 
    if (aug == "random_flip"):
        image = random_flip(image)
    return image


def map_zero_one(x, a, b):
    """
    map your vector to zero and one
    this is with the consideration that 
    min(x) <= a < b <= max(x) where it leads to:
    
    min(x)  --> 0
    a       --> 0
    b       --> 1
    max(x)  --> 1
    """
    assert b > a
    s = 1./(b - a)
    t = a/(a-b)
    y = s*x + t
    y[y>1] = 1
    y[y<0] = 0
    return y

def map_minus_one_to_one(x, a, b):
    """
    map your vector to minus one and plus one
    this is with the consideration that 
    min(x) <= a < b <= max(x) where it leads to:
    
    min(x)  --> -1
    a       --> -1
    b       --> 1
    max(x)  --> 1
    """
    assert b > a
    s = 2./(b - a)
    t = (a+b)/(a-b)
    y = s*x + t
    y[y>1] = 1
    y[y<-1] = -1
    return y

def data_mapping(image, statistics, method):
    """
    gets the image, statistics and the mapping method and according to the 
    method, chooses the right function
    Args:
        image(np.array) :   single-channel image
        statistics(dict):   includes the min, lower_bound, mean, std, upper_bound
                            and the max of the *whole training data*
        methods(list)     :   the method which will be used for mapping the image
    """
    if method == "normalize":
            image = transforms.Normalize(statistics["mean"] , statistics["std"] )(image) 
    elif method == "map_zero_one":
        for ch in range(image.shape[0]):
            a = statistics["lower_bound"][ch]
            b = statistics["upper_bound"][ch]
            image[ch,:,:] = map_zero_one(image[ch,:,:], a, b)
    elif method == "map_minus_one_to_one":
        for ch in range(image.shape[0]):
            a = statistics["lower_bound"][ch]
            b = statistics["upper_bound"][ch]
            image[ch,:,:] = map_minus_one_to_one(image[ch,:,:], a, b)
    else:
        raise Exception('Wrong mapping function')
    return image


class Dataset_Generator(Dataset):
    """Dataset_Generator"""

    def __init__(self,  data_dir,    df , channels , set_type ,
                    reshape_size = 64, data_map = [], statistics = None , augmentation = []):
        """
        Args:
            data_dir (string): Directory with all the images. 
            df(pandas df): dataframe including the files, labels etc. 
            channels(list): the existing channels in the data_dir
            set_type(str): train, test, or validation
            reshape_size(int): the reshape size for images
            data_map(list): list of data transofrmations done before the training
            statistics(dict): dict of statistcis of the training set. It can be 
                              used for data mapping
            augmentation(list): list of the augmentations done on the data
        """
        self.df = df.copy().reset_index(drop = True)
        
        # just using the set of interest
        if set_type is not None:
            self.df = self.df[self.df["set"]==set_type].reset_index(drop = True) 
        
        self.data_dir = data_dir 
        self.reshape_size = reshape_size
        self.channels = channels
        self.statistics = statistics 
        self.data_map = data_map
        self.augmentation = augmentation
        
    def __len__(self):
        return self.df.shape[0]

    def __getitem__(self, idx):
        
        if torch.is_tensor(idx):
            idx = idx.tolist() 

        image_path = str(self.df.loc[idx,"file"])

        # creating the image
        image = imread(image_path)
        image = np.zeros((image.shape[0], image.shape[1], len(self.channels)), dtype = np.float64)

        # filling the image with different channels 
        for ch in range(0,len(self.channels) ): 
            img_name = str(self.df.loc[idx,"file"]).replace(self.channels[0]+".", self.channels[ch]+".")                    
            image_dummy = imread(img_name).astype(np.float64) / 4095.
            image[:,:,ch] = image_dummy
            
        for aug in self.augmentation:
            image = data_augmentation(image,aug)
        
        image = resize(image , (self.reshape_size, self.reshape_size, len(self.channels)) ) 

        # transposeing from HWC to CHW for pytorch
        image = image.transpose(2,0,1)

        # map numpy array to tensor
        image = torch.from_numpy(np.flip(image,axis=0).copy() ) 
        
        for dm in self.data_map:
            image = data_mapping(image, self.statistics, dm)
            
        label = self.df.loc[idx,"label"]
        label = np.array([label]) 
        
        sample = {'image': image , 'label': torch.from_numpy(label) , "idx":idx }
        
        return sample
