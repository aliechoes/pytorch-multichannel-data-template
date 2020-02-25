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
    reshape_size = image.shape[0]
    width = int(reshape_size * ratio)
    height = int(reshape_size * ratio)
    x = random.randint(0, reshape_size - width)
    y = random.randint(0, reshape_size - height)
    image = image[y:y+height, x:x+width, :] 
    return image

def random_rotation(image ):
    # pick a random degree of rotation between 25% on the left and 25% on the right
    random_degree = random.uniform(-25, 25)
    return rotate(image, random_degree)

def random_flip(image ):
    # pick a random degree of rotation between 25% on the left and 25% on the right
    random_number = np.random.random()
    
    if  random_number < 0.25: 
        return image
    elif random_number >= 0.25 and random_number < 0.50:
        return flipud(image) 
    elif random_number >= 0.50 and random_number < 0.75:
        return fliplr(image)
    else: 
        return flipud(fliplr(image))



class Dataset_Generator(Dataset):
    """Dataset_Generator"""

    def __init__(self,  data_dir,  file_extension, df , channels , set_type ,
                    reshape_size = 64, mean = None , std = None, augmentation = False):
        """
        Args:
            csv_file (string): Path to the csv file with annotations.
            data_dir (string): Directory with all the images. 
        """
        self.df = df.copy().reset_index(drop = True)
        
        # just using the set of interest
        if set_type is not None:
            self.df = self.df[self.df["set"]==set_type].reset_index(drop = True) 
        
        self.data_dir = data_dir 
        self.file_extension = file_extension
        self.reshape_size = reshape_size
        self.channels = channels
        self.mean = mean
        self.std = std
        self.augmentation = augmentation
        
    def __len__(self):
        return self.df.shape[0]

    def __getitem__(self, idx):
        
        if torch.is_tensor(idx):
            idx = idx.tolist() 

        image_path = os.path.join(self.data_dir,self.df.loc[idx,"class"], \
                                   str(self.df.loc[idx,"file"])  +"_" + \
                                       self.channels[0] + self.file_extension)
        image = imread(image_path)
        image = np.zeros((image.shape[0], image.shape[1], len(self.channels)), dtype = np.float64)

        for ch in range(0,len(self.channels) ): 
            img_name = os.path.join(self.data_dir,self.df.loc[idx,"class"], \
                                   str(self.df.loc[idx,"file"])  +"_" + \
                                       self.channels[ch] + self.file_extension)
                                       
            image_dummy = imread(img_name).astype(np.float64)
            image[:,:,ch] = image_dummy
            
        if self.augmentation:
            if np.random.random() > 0.5:
                image = random_crop(image) 
            if np.random.random() > 0.5: 
                image = random_rotation(image) 
            image = random_flip(image)
        
        image = resize(image , (self.reshape_size, self.reshape_size, len(self.channels)) ) 

        image = image.transpose(2,0,1)
        image = torch.from_numpy(np.flip(image,axis=0).copy() ) 
        
        if self.mean is not None and self.std is not None:
            image = transforms.Normalize(self.mean,self.std)(image) 
            
        label = self.df.loc[idx,"label"]
        
        label = np.array([label]) 
        
        sample = {'image': image , 'label': torch.from_numpy(label) , "idx":idx }
        
        return sample
