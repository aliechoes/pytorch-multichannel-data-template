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
from skimage.transform import   rescale, resize, rotate, AffineTransform, warp
from skimage.util import crop, pad, random_noise
from imageio import imread 
import numpy as np
from numpy import fliplr, flipud
from skimage import data
from skimage.transform import   rescale, resize, rotate
import warnings
import random
warnings.filterwarnings("ignore")

def crop_pad_h_w(image_dummy,reshape_size):
    if image_dummy.shape[0] < reshape_size:
        h1_pad = ( reshape_size - image_dummy.shape[0])/2
        h1_pad = int(h1_pad)
        h2_pad =  reshape_size - h1_pad - image_dummy.shape[0]
        h1_crop = 0
        h2_crop = 0
    else:
        h1_pad = 0
        h2_pad = 0
        h1_crop = ( reshape_size - image_dummy.shape[0])/2
        h1_crop = abs(int(h1_crop))
        h2_crop = image_dummy.shape[0]- reshape_size  - h1_crop

    if image_dummy.shape[1] < reshape_size:
        w1_pad = (reshape_size - image_dummy.shape[1])/2
        w1_pad = int(w1_pad)
        w2_pad = reshape_size - w1_pad - image_dummy.shape[1]
        w1_crop = 0
        w2_crop = 0
    else:
        w1_pad = 0
        w2_pad = 0
        w1_crop = (reshape_size - image_dummy.shape[1])/2
        w1_crop = abs(int(w1_crop))
        w2_crop = image_dummy.shape[1]- reshape_size  - w1_crop

    h = [h1_crop, h2_crop, h1_pad, h2_pad]
    w = [w1_crop, w2_crop, w1_pad, w2_pad] 
    return h, w


def random_erasing(image_dummy,ratio = 0.3 ):
    
    h1 = random.randint(0, int((1 - ratio)*image_dummy.shape[0])  )
    w1 = random.randint(0, int((1 - ratio)*image_dummy.shape[1])  )
    hd = random.randint(0, int((ratio)*image_dummy.shape[0])  )
    wd = random.randint(0, int((ratio)*image_dummy.shape[1])  ) 
    
    for ch in range(image_dummy.shape[2]):
        image_dummy[h1:h1+hd, w1:w1+wd, ch  ] = image_dummy[:,:,ch].mean() 

    return image_dummy

def random_crop(image, ratio = 0.75):
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

def affine_transformation(image, ratio = 0.2 ):
    """
    pick a random degree of rotation between 25 degree on 
    the left and 25 degree on the right
    """
    random_degree = random.uniform(-15, 15) * np.pi / 180.
    random_translation_x = random.randint(0, int(image.shape[0] *ratio))
    random_translation_y = random.randint(0, int(image.shape[0] *ratio))
    
    transform = AffineTransform(translation=(random_translation_x,random_translation_y) , rotation = random_degree)
    wrapShift = warp(image,transform,mode= random.sample(["constant", "edge"], 1)[0])
    return wrapShift

def random_rotation(image ):
    """
    pick a random degree of rotation between 25 degree on 
    the left and 25 degree on the right
    """
    random_degree = random.uniform(-25, 25)
    padding_mode = random.sample(["constant", "edge"], 1)[0]
    return rotate(image, random_degree, mode = padding_mode )

def random_flip(image ):
    """
    randomly choose to flip an image
    """
    if random.randint(0,1):
        image = flipud(image) # vertical flip
    if random.randint(0,1):
        image = fliplr(image) # horizontal flip
    return image

def random_gaussian_noise(image ):
    """
    randomly choose to flip an image
    """
    sigma = image.std()
    for ch in range(image.shape[2] ):
        sigma = min(sigma, image[:,:,ch].std()  )
    image = random_noise(image, var = sigma**2)
    return image

def data_augmentation(image, aug):
    """
    combine different augmentation methods
    """
    if (aug == "random_crop") and (random.randint(0,1)):
        image = random_crop(image) 
    if (aug == "random_rotation") and (random.randint(0,1)): 
        image = random_rotation(image) 
    if (aug == "random_flip") and (random.randint(0,1)): 
        image = random_flip(image)
    if (aug == "affine_transformation") and (random.randint(0,1)):  
        image = affine_transformation(image)
    if (aug == "random_gaussian_noise") and (random.randint(0,1)):  
        image = random_gaussian_noise(image)
    if (aug == "random_erasing") and (random.randint(0,1)):  
        image = random_erasing(image) 
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
            image = transforms.Normalize(   statistics["mapped_mean"] , 
                                            statistics["mapped_std"] )(image) 
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

    def __init__(self,   df , channels , set_type , scaling_factor = 255 ,
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
        self.old_index = self.df["set"].isin(set_type)
        self.df = self.df[self.old_index].reset_index(drop = True) 
        
        self.scaling_factor = scaling_factor
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

        if self.augmentation != []:
            padding_mode = random.sample(["constant", "edge", "linear_ramp"], 1)[0]
        else:
            padding_mode = "edge"
        
        # creating the image
        image_dummy = imread(image_path)
        h, w = crop_pad_h_w(image_dummy, self.reshape_size)
        h1_crop, h2_crop, h1_pad, h2_pad =  h 
        w1_crop, w2_crop, w1_pad, w2_pad = w 
        image = np.zeros((self.reshape_size, self.reshape_size, len(self.channels)), dtype = np.float64)

        # filling the image with different channels 
        for ch in range(0,len(self.channels) ): 
            img_name = image_path.replace(self.channels[0]+".", self.channels[ch]+".")                    
            image_dummy = imread(img_name).astype(np.float64) / float(self.scaling_factor)
            image_dummy = crop(image_dummy, ((h1_crop,h2_crop),(w1_crop,w2_crop)))
            image_dummy = pad(image_dummy, ((h1_pad,h2_pad),(w1_pad,w2_pad)), padding_mode )
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
